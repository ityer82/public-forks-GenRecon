"""Diagnostic: score candidate orientations of an MV-SAM3D mesh by silhouette IoU against
the real per-view object masks, using the DA3 cameras from <run_dir>/stage_2_mv_sam3d/input.

Standalone and read-only with respect to the pipeline: it imports helpers from
mv_sam3d/mvsam3d_scripts/collect_mvsam3d_outputs.py but never modifies or writes into the run.

Candidates scored (each after the same bbox scale+translation fit onto the segmented point
cloud that align_meshes_to_scene.py applies to a mesh, so orientation is the only difference):
  - `sam3d_only`: transform_to_world() output, no snap
  - `snap_<i>`: the 24 cube-group-seeded ICP results (the pipeline's gravity_snap_correction
    keeps the lowest 3D cost; here all 24 are kept and scored)
Silhouettes come from splatting densely-sampled surface points into each view.

Usage (repo-root .venv):
    PYTHONPATH=.:mv_sam3d/mvsam3d_scripts .venv/bin/python scripts/score_mesh_orientation.py \
        --run_dir runs/kitchen_floor --label "white chair" --out_dir <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
from trimesh.registration import icp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mv_sam3d" / "mvsam3d_scripts"))
import collect_mvsam3d_outputs as cmo  # noqa: E402
import silhouette_orientation as so  # noqa: E402


def _cube_group_rotations():
    import itertools

    rots = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            p = np.zeros((3, 3))
            for i, j in enumerate(perm):
                p[i, j] = signs[i]
            if abs(np.linalg.det(p) - 1.0) < 1e-6:
                rots.append(p)
    return rots


def _pca_frame(points, center):
    evals, evecs = np.linalg.eigh(np.cov((points - center).T))
    evecs = evecs[:, np.argsort(-evals)]
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1
    return evecs


def load_sam3d_world_points(vis_dir: Path, dataset: str, label: str, n: int = 60000):
    obj_dir = cmo.find_latest_object_dir(vis_dir, dataset, label)
    params = dict(np.load(obj_dir / "params.npz"))
    scene = trimesh.load(str(obj_dir / "result.glb"), force="scene")
    for g in scene.geometry.values():
        g.vertices = cmo.transform_to_world(np.asarray(g.vertices), params)
    mesh = scene.to_mesh()
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=0)
    return np.asarray(pts, dtype=np.float64)


def seeded_icp_candidates(mesh_pts, target_pts, n_icp=8000):
    """All 24 cube-group-seeded ICP results (matrix, 3D cost) -- same seeding as gravity_snap_correction."""
    sub = mesh_pts[np.random.default_rng(0).choice(len(mesh_pts), min(n_icp, len(mesh_pts)), replace=False)]
    t_c, s_c = target_pts.mean(0), sub.mean(0)
    tf, sf = _pca_frame(target_pts, t_c), _pca_frame(sub, s_c)
    out = []
    for perm in _cube_group_rotations():
        r0 = tf @ perm @ sf.T
        init = np.eye(4)
        init[:3, :3] = r0
        init[:3, 3] = t_c - r0 @ s_c
        m, _, cost = icp(sub, target_pts, initial=init, max_iterations=50, threshold=1e-7,
                         reflection=False, scale=False)
        out.append((m, float(cost)))
    return out


def bbox_fit(pts, target_pts):
    """Uniform scale (longest bbox side) + bbox-center translation, as in align_meshes_to_scene."""
    s_min, s_max = pts.min(0), pts.max(0)
    t_min, t_max = np.percentile(target_pts, 1, axis=0), np.percentile(target_pts, 99, axis=0)
    scale = (t_max - t_min).max() / (s_max - s_min).max()
    return (pts - (s_min + s_max) / 2) * scale + (t_min + t_max) / 2


def render_silhouette(pts, ext, K, hw, radius=2):
    h, w = hw
    cam = pts @ ext[:, :3].T + ext[:, 3]
    front = cam[:, 2] > 1e-3
    uv = cam[front] @ K.T
    uv = uv[:, :2] / uv[:, 2:3]
    ij = np.round(uv).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    img = np.zeros((h, w), np.uint8)
    img[ij[ok, 1], ij[ok, 0]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    img = cv2.dilate(img, k)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0


def load_views(run_dir: Path, scene: str, label: str, max_views: int):
    inp = run_dir / "stage_2_mv_sam3d" / "input"
    d = np.load(inp / "da3_output.npz")
    stems = [Path(str(p)).stem for p in d["image_files"]]
    views = []
    for mp in sorted((inp / label).glob("*.png")):
        if mp.stem not in stems:
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)
        mask = (m[..., 3] if m.ndim == 3 and m.shape[2] == 4 else m.max(-1) if m.ndim == 3 else m) > 0
        if mask.sum() < 400:
            continue
        i = stems.index(mp.stem)
        views.append((mp.stem, d["extrinsics"][i].astype(np.float64), d["intrinsics"][i].astype(np.float64), mask))
    if len(views) > max_views:
        idx = np.linspace(0, len(views) - 1, max_views).round().astype(int)
        views = [views[i] for i in idx]
    return views


def score(pts, views):
    ious, precs = [], []
    for _stem, ext, K, mask in views:
        sil = render_silhouette(pts, ext, K, mask.shape)
        inter = (sil & mask).sum()
        ious.append(inter / max((sil | mask).sum(), 1))
        precs.append(inter / max(sil.sum(), 1))
    return float(np.mean(ious)), float(np.mean(precs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--visualization_dir", type=Path, default=REPO / "mv_sam3d" / "visualization")
    ap.add_argument("--max_views", type=int, default=12)
    args = ap.parse_args()

    scene = args.run_dir.name
    dataset = f"{scene}_mvsam3d_input"
    san = cmo.sanitize_label(args.label)
    pc = args.run_dir / "stage_1_segmentation" / san / "point_cloud" / f"{san}.ply"
    target = np.asarray(trimesh.load(str(pc)).vertices, dtype=np.float64)

    pts = load_sam3d_world_points(args.visualization_dir, dataset, args.label)
    views = load_views(args.run_dir, scene, args.label, args.max_views)
    print(f"{args.label}: {len(views)} views, {len(pts)} surface samples, {len(target)} target points")

    cands = {"sam3d_only": (pts, None)}
    for i, (m, cost) in enumerate(seeded_icp_candidates(pts, target)):
        cands[f"snap_{i:02d}"] = ((m[:3, :3] @ pts.T).T + m[:3, 3], cost)

    rows = []
    for name, (p, cost3d) in cands.items():
        fitted = bbox_fit(p, target)
        iou, prec = score(fitted, views)
        zc = (fitted[:, 2].mean() - fitted[:, 2].min()) / (np.ptp(fitted[:, 2]) + 1e-9)
        rows.append({"name": name, "iou": iou, "precision": prec, "cost3d": cost3d,
                     "centroid_height_frac": float(zc)})
    rows.sort(key=lambda r: -r["iou"])
    print(f"{'candidate':12} {'IoU':>6} {'prec':>6} {'cost3d':>9} {'cz_frac':>7}")
    for r in rows:
        c = "-" if r["cost3d"] is None else f"{r['cost3d']:.5f}"
        print(f"{r['name']:12} {r['iou']:6.3f} {r['precision']:6.3f} {c:>9} {r['centroid_height_frac']:7.2f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{san}_orientation_scores.json").write_text(json.dumps(rows, indent=2))

    # overlays for: best-by-IoU, best-by-3D-cost (what the pipeline picks), sam3d_only
    by_cost = min((r for r in rows if r["cost3d"] is not None), key=lambda r: r["cost3d"])["name"]
    for tag, name in (("best_iou", rows[0]["name"]), ("best_cost3d", by_cost), ("sam3d_only", "sam3d_only")):
        fitted = bbox_fit(cands[name][0], target)
        tiles = []
        for stem, ext, K, mask in views[:4]:
            sil = render_silhouette(fitted, ext, K, mask.shape)
            img = cv2.imread(str(args.run_dir / f"stage_2_mv_sam3d/input/images/{stem}.jpg"))
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
            img[mask & ~sil] = (0.5 * img[mask & ~sil] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)   # mask only
            img[sil & ~mask] = (0.5 * img[sil & ~mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)   # render only
            img[sil & mask] = (0.5 * img[sil & mask] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)   # both
            tiles.append(img)
        cv2.imwrite(str(args.out_dir / f"{san}_{tag}_{name}.png"), np.hstack(tiles[:2]) if len(tiles) < 4 else
                    np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])]))


if __name__ == "__main__":
    main()
