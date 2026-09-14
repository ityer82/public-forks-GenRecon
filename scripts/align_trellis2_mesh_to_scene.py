"""Rescale/translate a TRELLIS2 object mesh into the scene's global frame.

TRELLIS2's `mesh.glb` (Stage 3 of run_full_pipeline.sh) is single-image-to-3D
output in TRELLIS2's own object-centric, view-agnostic canonical frame: it's
centered at its own bbox centroid, scaled so its longest bbox axis is ~1.0
(fit inside [-0.5, 0.5]^3, per data_toolkit/voxelize_pbr.py's training-time
normalization), and exported Y-up (glTF convention -- o_voxel/postprocess.py's
`to_glb` always applies an unconditional axis swap from TRELLIS2's own
internal Z-up frame). TRELLIS2's conditioning is image-features-only (no
elevation/azimuth/pose in or out anywhere in the pipeline), so there is no
per-object rotation information tying the mesh back to the scene.

The scene-frame reference is the object's real crop out of the reconstructed
scene mesh, e.g. shapes/<label>_mesh.ply written by extract_object_mesh.py --
already in the scene's real world frame (position, scale, orientation), with
no re-centering/re-normalization applied there.

This script does NOT attempt to estimate/search a per-object rotation (e.g.
via ICP) -- only a scale + translation are computed. It does apply one fixed,
scene-independent correction: run_full_pipeline.sh's --align-to-gravity is
always on, so the scene frame is always Z-up (vggt-omega/align_to_gravity.py:
"x/y horizontal, +z up"), while TRELLIS2 always exports Y-up glTF -- so a
constant -90-deg-about-X correction is applied first to undo that fixed
convention mismatch before computing scale/translation. Any remaining
left-right/front-back (yaw) mismatch is not corrected.

Usage:
    uv run python scripts/align_trellis2_mesh_to_scene.py \
        --trellis_glb runs/kitchen_bowl/trellis2_meshes/bowl/mesh.glb \
        --scene_mesh_ply runs/kitchen_bowl/genrecon_output/shapes/bowl_mesh.ply \
        --out_glb runs/kitchen_bowl/trellis2_meshes/bowl/mesh_aligned.glb
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData

from genrecon.utils.logger import logger

# glTF Y-up -> scene Z-up: (x, y, z) -> (x, -z, y), i.e. -90 deg about X.
ZUP_CORRECTION = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _bbox_extent(bbox_min: np.ndarray, bbox_max: np.ndarray) -> float:
    extent = float((bbox_max - bbox_min).max())
    if extent <= 0.0:
        raise ValueError(f"Degenerate bbox: min={bbox_min}, max={bbox_max}")
    return extent


def load_target_bbox(scene_mesh_ply: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(scene_mesh_ply)
    vertex = ply["vertex"]
    xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    return xyz.min(axis=0), xyz.max(axis=0)


def align_trellis_mesh_to_scene(
    trellis_glb: Path, scene_mesh_ply: Path, apply_zup_correction: bool = True
) -> tuple[trimesh.Scene, np.ndarray, dict]:
    """Returns (transformed scene, combined 4x4 transform, diagnostics dict).

    apply_zup_correction=False skips the -90-deg-about-X correction: it's only
    needed for TRELLIS2's fixed Y-up glTF export convention. A mesh that's
    already been placed in the scene's real Z-up world frame with correct
    orientation (e.g. MV-SAM3D's collect_mvsam3d_outputs.py output) must not
    get this applied again -- it would introduce a spurious 90-deg rotation.
    The scale+translation bbox-fit below is still applied either way.
    """
    target_bbox_min, target_bbox_max = load_target_bbox(scene_mesh_ply)
    target_extent = _bbox_extent(target_bbox_min, target_bbox_max)
    target_center = (target_bbox_min + target_bbox_max) / 2.0

    scene = trimesh.load(trellis_glb, force="scene")
    zup_correction = ZUP_CORRECTION if apply_zup_correction else np.eye(4)
    scene.apply_transform(zup_correction)

    source_bbox_min, source_bbox_max = scene.bounds
    source_extent = _bbox_extent(source_bbox_min, source_bbox_max)
    source_center = (source_bbox_min + source_bbox_max) / 2.0

    scale = target_extent / source_extent

    scale_translate = np.eye(4)
    scale_translate[:3, :3] = np.eye(3) * scale
    scale_translate[:3, 3] = target_center - scale * source_center
    scene.apply_transform(scale_translate)

    combined_transform = scale_translate @ zup_correction
    diagnostics = {
        "scale": scale,
        "target_bbox_min": target_bbox_min.tolist(),
        "target_bbox_max": target_bbox_max.tolist(),
        "target_center": target_center.tolist(),
        "source_bbox_min_after_zup": source_bbox_min.tolist(),
        "source_bbox_max_after_zup": source_bbox_max.tolist(),
        "source_center_after_zup": source_center.tolist(),
        "combined_transform": combined_transform.tolist(),
    }
    return scene, combined_transform, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trellis_glb", type=Path, required=True)
    parser.add_argument("--scene_mesh_ply", type=Path, required=True)
    parser.add_argument("--out_glb", type=Path, required=True)
    parser.add_argument(
        "--out_transform_json", type=Path, default=None,
        help="Defaults to <out_glb> with a .transform.json suffix.",
    )
    parser.add_argument(
        "--no-zup-correction", action="store_true",
        help="Skip the -90-deg-about-X TRELLIS2-Y-up-glTF correction (e.g. for a mesh "
        "already placed in the scene's real Z-up world frame, such as MV-SAM3D output).",
    )
    args = parser.parse_args()

    scene, transform, diagnostics = align_trellis_mesh_to_scene(
        args.trellis_glb, args.scene_mesh_ply, apply_zup_correction=not args.no_zup_correction
    )

    args.out_glb.parent.mkdir(parents=True, exist_ok=True)
    if args.out_glb.suffix.lower() == ".ply":
        # PLY has no standard image-texture linkage (trimesh's PLY exporter
        # only writes UVs, not the texture image), so MeshLab would otherwise
        # load an untextured mesh -- bake the texture into per-vertex colors
        # instead, sampled at each vertex's UV coordinate, so color survives.
        mesh = scene.to_mesh()
        if hasattr(mesh.visual, "to_color"):
            mesh.visual = mesh.visual.to_color()
        mesh.export(args.out_glb)
    else:
        scene.export(args.out_glb)
    logger.info(f"Wrote {args.out_glb} (scale={diagnostics['scale']:.4f})")

    out_transform_json = args.out_transform_json or args.out_glb.with_suffix(".transform.json")
    out_transform_json.write_text(json.dumps(diagnostics, indent=2))
    logger.info(f"Wrote {out_transform_json}")


if __name__ == "__main__":
    main()
