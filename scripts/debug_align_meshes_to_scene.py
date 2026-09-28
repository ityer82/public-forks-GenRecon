"""Debug: dump every MV-SAM3D object in global scene coordinates, before and after the
pipeline's post-generation yaw alignment.

Replicates what `stage3_mvsam3d` (collect_mvsam3d_outputs.py) followed by the Stage 10 /
Stage P1 scale+translation fit (align_meshes_to_scene.py) do to each mesh, by calling those
same functions (so this script can't drift from the pipeline), and writes:

  <out_dir>/objects_unaligned.glb    MV-SAM3D meshes exactly as generated (SAM3D scale/rotation/
                                     translation + reference-view extrinsic -> world frame), then
                                     the bbox scale+translation fit onto the scene crop. No yaw step.
  <out_dir>/objects_yaw_aligned.glb  the same, with the silhouette-IoU yaw step applied first
                                     (what the pipeline currently produces).
  <out_dir>/objects_scene_crop.glb   (--include-scene-crop) the reconstructed scene crops, for reference.
  <out_dir>/alignment_report.json    per-object yaw, SAM3D tilt (result.glb up axis vs world +Z), bbox extents.

SAM3D tilt is the angle between result.glb's up axis (glTF +Y) pushed through the same transform
chain and world +Z. The yaw step never changes it, and nothing downstream corrects it.

Caveats: like the pipeline, the newest visualization dir for a label (by mtime) is used.
Writes only under --out_dir; pipeline outputs are never touched.

Usage (repo-root .venv):
    PYTHONPATH=.:mv_sam3d/mvsam3d_scripts .venv/bin/python scripts/debug_align_meshes_to_scene.py \\
        --run_dir runs/kitchen_floor
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mv_sam3d" / "mvsam3d_scripts"))
sys.path.insert(0, str(REPO / "scripts"))

import collect_mvsam3d_outputs as cmo  # noqa: E402
from align_meshes_to_scene import align_trellis_mesh_to_scene  # noqa: E402


def sam3d_tilt_deg(visualization_dir: Path, dataset: str, label: str) -> float | None:
    """Angle (deg) between result.glb's up axis (glTF +Y) pushed through transform_to_world's full
    chain and world +Z (the scene is gravity-aligned Z-up). ~0 = upright, ~180 = upside-down."""
    obj_dir = cmo.find_latest_object_dir(visualization_dir, dataset, label)
    if obj_dir is None:
        return None
    params = dict(np.load(obj_dir / "params.npz"))
    p = cmo.transform_to_world(np.array([[0, 0, 0], [0, 1, 0]], dtype=np.float32), params).astype(np.float64)
    d = p[1] - p[0]
    d /= np.linalg.norm(d)
    return float(np.degrees(np.arccos(np.clip(d[2], -1.0, 1.0))))


def dumped(scene: trimesh.Scene) -> list[trimesh.Trimesh]:
    """Geometries with their scene-graph transforms baked in."""
    return list(scene.dump(concatenate=False))


def yaw_between(raw: np.ndarray, aligned: np.ndarray) -> tuple[float, float]:
    """(yaw deg about +Z, residual) of the best rigid xy-rotation mapping raw -> aligned, both
    centered; residual is the RMS of the leftover (xy misfit and any z change)."""
    a, b = raw - raw.mean(0), aligned - aligned.mean(0)
    h = a[:, :2].T @ b[:, :2]
    yaw = np.arctan2(h[0, 1] - h[1, 0], h[0, 0] + h[1, 1])
    c, s = np.cos(yaw), np.sin(yaw)
    pred = np.column_stack([c * a[:, 0] - s * a[:, 1], s * a[:, 0] + c * a[:, 1], a[:, 2]])
    return float(np.degrees(yaw)) % 360.0, float(np.sqrt(np.mean((pred - b) ** 2)))


def fit_and_collect(mesh_glb: Path, crop_ply: Path) -> tuple[list[trimesh.Trimesh], dict]:
    scene, _transform, diag = align_trellis_mesh_to_scene(mesh_glb, crop_ply, apply_zup_correction=False)
    return dumped(scene), diag


def compare_with_run(name: str, ours: list[trimesh.Trimesh], theirs_path: Path) -> str:
    if not theirs_path.exists():
        return f"{name}: {theirs_path} not found"
    theirs = dumped(trimesh.load(str(theirs_path), force="scene"))
    a = np.vstack([g.vertices for g in ours])
    b = np.vstack([g.vertices for g in theirs])
    if a.shape == b.shape:
        return f"{name}: max vertex deviation {np.abs(a - b).max():.2e} m"
    return f"{name}: vertex counts differ ({len(a)} vs {len(b)}), bounds deviation {np.abs(np.ptp(a, 0) - np.ptp(b, 0)).max():.2e} m"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--labels", default=None, help="Comma-separated labels (default: all in labels.json)")
    ap.add_argument("--out_dir", type=Path, default=None, help="Default: <run_dir>/debug_alignment")
    ap.add_argument("--visualization_dir", type=Path, default=REPO / "mv_sam3d" / "visualization")
    ap.add_argument("--include-scene-crop", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    scene_name = run_dir.name
    dataset = f"{scene_name}_mvsam3d_input"
    mask_dir = run_dir / "stage_1_segmentation"
    shapes_dir = run_dir / "stage_3_genrecon" / "shapes"
    out_dir = (args.out_dir or run_dir / "debug_alignment").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_map = json.loads((mask_dir / "labels.json").read_text())
    labels = [x.strip() for x in args.labels.split(",")] if args.labels else list(labels_map)

    variants = {"unaligned": trimesh.Scene(), "yaw_aligned": trimesh.Scene()}
    crop_scene = trimesh.Scene()
    report = {}

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        raw_dir, aligned_dir = Path(tmp) / "raw", Path(tmp) / "aligned"
        cmo.collect_mvsam3d_outputs(args.visualization_dir, dataset, labels, raw_dir)
        cmo.collect_mvsam3d_outputs(
            args.visualization_dir, dataset, labels, aligned_dir,
            scene_pointcloud_dir=mask_dir, mvsam3d_input_dir=run_dir / "stage_2_mv_sam3d" / "input",
        )

        for label in labels:
            san = cmo.sanitize_label(label)
            crop_ply = shapes_dir / f"{san}_mesh.ply"
            raw_glb, aligned_glb = raw_dir / label / "mesh.glb", aligned_dir / label / "mesh.glb"
            if not (raw_glb.exists() and aligned_glb.exists() and crop_ply.exists()):
                print(f"[debug_align] {label}: missing mesh/crop, skipping")
                continue

            # yaw actually applied by the pipeline step, from the pre-fit world-frame meshes
            raw_v = np.vstack([g.vertices for g in dumped(trimesh.load(str(raw_glb), force="scene"))])
            ali_v = np.vstack([g.vertices for g in dumped(trimesh.load(str(aligned_glb), force="scene"))])
            yaw, resid = yaw_between(raw_v, ali_v)

            un_geoms, un_diag = fit_and_collect(raw_glb, crop_ply)
            al_geoms, al_diag = fit_and_collect(aligned_glb, crop_ply)
            for scene, geoms in ((variants["unaligned"], un_geoms), (variants["yaw_aligned"], al_geoms)):
                for i, g in enumerate(geoms):
                    scene.add_geometry(g, node_name=f"{label}_{i}", geom_name=f"{label}_{i}")

            crop = trimesh.load(str(crop_ply), force="mesh")
            if args.include_scene_crop:
                crop_scene.add_geometry(crop, node_name=label, geom_name=label)

            ext = lambda gs: np.ptp(np.vstack([g.vertices for g in gs]), axis=0)  # noqa: E731
            report[label] = {
                "applied_yaw_deg": None if resid > 1e-3 else round(yaw, 1),
                "yaw_fit_residual_m": resid,
                "sam3d_tilt_deg": sam3d_tilt_deg(args.visualization_dir, dataset, label),
                "fit_scale_unaligned": un_diag["scale"],
                "fit_scale_yaw_aligned": al_diag["scale"],
                "extent_scene_crop_xyz": np.ptp(crop.vertices, axis=0).round(3).tolist(),
                "extent_unaligned_xyz": ext(un_geoms).round(3).tolist(),
                "extent_yaw_aligned_xyz": ext(al_geoms).round(3).tolist(),
                "replication": [
                    compare_with_run("stage_2_mv_sam3d(pre-fit)", dumped(trimesh.load(str(aligned_glb), force="scene")),
                                     run_dir / "stage_2_mv_sam3d" / label / "mesh.glb"),
                    compare_with_run("shapes/glb(post-fit)", al_geoms, shapes_dir / "glb" / san / "mesh.glb"),
                ],
            }

    for name, scene in variants.items():
        scene.export(str(out_dir / f"objects_{name}.glb"))
    if args.include_scene_crop:
        crop_scene.export(str(out_dir / "objects_scene_crop.glb"))
    (out_dir / "alignment_report.json").write_text(json.dumps(report, indent=2))

    print(f"\nWrote {out_dir}/objects_unaligned.glb, objects_yaw_aligned.glb, alignment_report.json")
    print(f"{'label':14} {'yaw':>6} {'tilt':>6}  extent crop / unaligned / yaw_aligned (xyz, m)")
    for label, r in report.items():
        yaw = "n/a" if r["applied_yaw_deg"] is None else f"{r['applied_yaw_deg']:.0f}"
        tilt = "n/a" if r["sam3d_tilt_deg"] is None else f"{r['sam3d_tilt_deg']:.1f}"
        print(f"{label:14} {yaw:>6} {tilt:>6}  {r['extent_scene_crop_xyz']} / {r['extent_unaligned_xyz']} / {r['extent_yaw_aligned_xyz']}")
        for line in r["replication"]:
            print(f"{'':14} {line}")


if __name__ == "__main__":
    main()
