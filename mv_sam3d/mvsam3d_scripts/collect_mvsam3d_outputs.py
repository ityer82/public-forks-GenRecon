#!/usr/bin/env python3
"""
Collect per-object result.glb + params.npz produced by run_inference_weighted.py's
multi-object mode, transform each mesh from SAM3D's canonical space into the world
frame of the input data (using the pose parameters + the embedded reference-view
extrinsic), and write it out at a fixed, non-timestamped path for a calling pipeline
to consume (e.g. genrecon's run_full_pipeline.sh, which expects
`<out_dir>/<label>/mesh.glb`).

Must run in MV-SAM3D's own venv (needs pytorch3d, trimesh, torch).

Transform chain (validated against genrecon's own point cloud for a real scene, see
mv_sam3d_summary.md / the alignment-verification writeup for `bowl`):
  1. canonical mesh vertices (Z-up) -> Y-up
  2. apply params.npz's scale -> rotate(quaternion) -> translate
     (this lands in the reference view's PyTorch3D camera space)
  3. PyTorch3D -> OpenCV camera-space convention: (x, y, z) -> (-x, -y, z)
  4. apply inverse(params.npz['ref_extrinsic']) (camera-to-world for the reference view)

This mirrors run_inference_weighted.py's own merge_glb_with_da3_aligned(), except it
uses the *matched* reference-view extrinsic saved into params.npz (the first view
actually used for this object after mask/filename matching) instead of that function's
fallback path, which assumes the dataset's global frame 0 and applies the world-to-camera
matrix un-inverted -- both wrong whenever an object isn't visible in the dataset's first
frame, which import_from_genrecon.py's empty-mask skipping makes routine.

The above transform chain depends on MV-SAM3D's own predicted rotation quaternion and
the matched reference-view extrinsic, either of which can be wrong (bad quaternion
regression, or a mismatched reference view) -- and nothing downstream ever corrects a
bad rotation (align_meshes_to_scene.py only ever fits scale + translation for
the mvsam3d backend). If `--mvsam3d_input_dir` and `--scene_pointcloud_dir` are given,
each object's yaw (rotation about the scene's vertical axis) is additionally chosen by
silhouette IoU: the SAM3D pose and yaw-swept variants of it are rendered into the real
views and scored against the object's SAM masks -- see silhouette_orientation.py. The
SAM3D pose is kept unless a yaw improves the IoU by at least MIN_IOU_GAIN. Only yaw is
changed, so an object can never be turned upside-down by this step.
"""
import argparse
import glob
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import Transform3d, quaternion_to_matrix

try:
    from genrecon.utils.logger import logger
except ImportError:  # standalone run in MV-SAM3D's own venv, without genrecon installed
    logger = logging.getLogger(__name__)

Z_UP_TO_Y_UP = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    dtype=np.float32,
)
P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)


def find_latest_object_dir(visualization_dir: Path, dataset_name: str, label: str) -> Path | None:
    pattern = str(visualization_dir / dataset_name / "multiobject" / "*" / label)
    candidates = [Path(p) for p in glob.glob(pattern) if (Path(p) / "result.glb").exists()]
    if not candidates:
        # single-object mode falls back to this non-multiobject layout
        pattern = str(visualization_dir / dataset_name / label / "*")
        candidates = [Path(p) for p in glob.glob(pattern) if (Path(p) / "result.glb").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "result.glb").stat().st_mtime)


def transform_to_world(vertices: np.ndarray, params: dict) -> np.ndarray:
    if "ref_extrinsic" not in params:
        raise ValueError(
            "params.npz has no 'ref_extrinsic' -- re-run inference with the patched "
            "run_inference_weighted.py that saves the matched reference-view extrinsic."
        )

    scale = params["scale"].astype(np.float32).flatten()
    rotation = params["rotation"].astype(np.float32).flatten()
    translation = params["translation"].astype(np.float32).flatten()
    ref_extrinsic = np.asarray(params["ref_extrinsic"], dtype=np.float64)
    if ref_extrinsic.shape == (3, 4):
        ref_extrinsic_44 = np.eye(4, dtype=np.float64)
        ref_extrinsic_44[:3, :4] = ref_extrinsic
        ref_extrinsic = ref_extrinsic_44

    v_rot = vertices.astype(np.float32) @ Z_UP_TO_Y_UP.T

    quat_t = torch.tensor(rotation, dtype=torch.float32).unsqueeze(0)
    r_mat = quaternion_to_matrix(quat_t)
    scale_t = torch.tensor(scale, dtype=torch.float32).reshape(1, -1)
    if scale_t.shape[-1] == 1:
        scale_t = scale_t.repeat(1, 3)
    translation_t = torch.tensor(translation, dtype=torch.float32).reshape(1, 3)
    pose_transform = (
        Transform3d(dtype=torch.float32).scale(scale_t).rotate(r_mat).translate(translation_t)
    )
    pts_local = torch.from_numpy(v_rot).float().unsqueeze(0)
    pts_p3d = pose_transform.transform_points(pts_local).squeeze(0).numpy()

    pts_cv = pts_p3d @ P3D_TO_CV.T

    cam_to_world = np.linalg.inv(ref_extrinsic)
    pts_h = np.hstack([pts_cv.astype(np.float64), np.ones((len(pts_cv), 1))])
    pts_world = (cam_to_world @ pts_h.T).T[:, :3]
    return pts_world.astype(np.float32)


def sanitize_label(label: str) -> str:
    """Mirrors genrecon/pipeline/stages.py's sanitize_label (spaces/slashes -> underscores):
    genrecon's segmented point clouds (COBGS_MASK_DIR/<label>/point_cloud/<label>.ply) are
    keyed by this sanitized dirname, not the raw --classes spelling used here."""
    return label.replace(" ", "_").replace("/", "_")


# A yaw is only applied when it beats the untouched SAM3D pose's mean silhouette IoU by
# this much, so noise (few views, near-symmetric objects) doesn't move a good pose.
MIN_IOU_GAIN = 0.02
MIN_TARGET_POINTS = 200
YAW_STEP_DEG = 10.0


def collect_mvsam3d_outputs(
    visualization_dir: Path,
    dataset_name: str,
    labels: list[str],
    out_dir: Path,
    scene_pointcloud_dir: Path | None = None,
    mvsam3d_input_dir: Path | None = None,
) -> list[str]:
    """Collects per-object result.glb + params.npz into <out_dir>/<label>/mesh.glb, transformed
    into the input data's world frame. Returns the list of labels that failed/were missing
    (empty on full success) instead of calling sys.exit(1), so a caller can decide policy.

    If both `scene_pointcloud_dir` (genrecon's COBGS_MASK_DIR, i.e.
    <run_dir>/stage_1_segmentation) and `mvsam3d_input_dir` (the bridged
    input dir holding da3_output.npz and the per-label masks) are given, each object's yaw is
    chosen by silhouette IoU against the real masks via silhouette_orientation.select_best_yaw().
    Missing masks/point clouds are skipped with a warning (not a failure): the object keeps
    its MV-SAM3D-only pose.
    """
    visualization_dir = Path(visualization_dir)
    out_dir = Path(out_dir)
    scene_pointcloud_dir = Path(scene_pointcloud_dir) if scene_pointcloud_dir else None
    mvsam3d_input_dir = Path(mvsam3d_input_dir) if mvsam3d_input_dir else None

    failures = []
    for label in labels:
        obj_dir = find_latest_object_dir(visualization_dir, dataset_name, label)
        if obj_dir is None:
            print(f"[collect_mvsam3d_outputs] WARNING: no result.glb found for label '{label}', skipping")
            failures.append(label)
            continue

        params = dict(np.load(obj_dir / "params.npz"))
        scene = trimesh.load(str(obj_dir / "result.glb"), force="scene")

        try:
            for geom in scene.geometry.values():
                geom.vertices = transform_to_world(np.asarray(geom.vertices), params)
        except ValueError as e:
            print(f"[collect_mvsam3d_outputs] WARNING: {label}: {e}, skipping")
            failures.append(label)
            continue

        if scene_pointcloud_dir is not None and mvsam3d_input_dir is not None:
            _maybe_apply_silhouette_yaw(scene, label, scene_pointcloud_dir, mvsam3d_input_dir)

        label_out_dir = out_dir / label
        label_out_dir.mkdir(parents=True, exist_ok=True)
        dest = label_out_dir / "mesh.glb"
        scene.export(str(dest))
        print(f"[collect_mvsam3d_outputs] {label}: {obj_dir / 'result.glb'} -> {dest}")

    if failures:
        print(f"[collect_mvsam3d_outputs] Failed/missing labels: {failures}")
    return failures


def _maybe_apply_silhouette_yaw(
    scene: trimesh.Scene, label: str, scene_pointcloud_dir: Path, mvsam3d_input_dir: Path
) -> None:
    from silhouette_orientation import load_views, select_best_yaw

    tag = "[collect_mvsam3d_outputs][silhouette_yaw]"
    pc_path = scene_pointcloud_dir / sanitize_label(label) / "point_cloud" / f"{sanitize_label(label)}.ply"
    if not pc_path.exists():
        print(f"{tag} {label}: no reference point cloud at {pc_path}, skipping check")
        return
    loaded = trimesh.load(str(pc_path))
    if not hasattr(loaded, "vertices"):  # trimesh returns an empty Scene for a 0-vertex PLY
        print(f"{tag} {label}: reference point cloud at {pc_path} is empty, skipping check")
        return
    target_points = np.asarray(loaded.vertices, dtype=np.float64)
    if len(target_points) < MIN_TARGET_POINTS:
        print(f"{tag} {label}: reference point cloud too sparse ({len(target_points)} pts), skipping check")
        return

    views = load_views(mvsam3d_input_dir, label)
    if not views:
        print(f"{tag} {label}: no usable mask views under {mvsam3d_input_dir}, skipping check")
        return

    mesh = scene.to_mesh()
    points, _ = trimesh.sample.sample_surface(mesh, 30000, seed=0)
    result = select_best_yaw(np.asarray(points, dtype=np.float64), target_points, views, YAW_STEP_DEG)

    gain = result["iou"] - result["identity_iou"]
    applied = gain >= MIN_IOU_GAIN
    print(
        f"{tag} {label}: views={len(views)} best_yaw={result['yaw_deg']:.0f}deg iou={result['iou']:.3f} "
        f"(sam3d_pose_iou={result['identity_iou']:.3f}, gain={gain:.3f})"
    )
    logger.debug(
        f"{tag} {label}: yaw debug "
        + json.dumps(
            {
                "views": len(views),
                "yaw_deg": result["yaw_deg"],
                "iou": result["iou"],
                "identity_iou": result["identity_iou"],
                "min_iou_gain": MIN_IOU_GAIN,
                "applied": bool(applied),
                "iou_by_yaw": {str(k): v for k, v in result["iou_by_yaw"].items()},
                "matrix": result["matrix"].tolist(),
            }
        )
    )

    if not applied:
        print(f"{tag} {label}: gain below {MIN_IOU_GAIN}, keeping MV-SAM3D pose")
        return
    scene.apply_transform(result["matrix"])
    print(f"{tag} {label}: applied yaw {result['yaw_deg']:.0f}deg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualization_dir", default="visualization", help="MV-SAM3D's visualization/ dir")
    parser.add_argument("--dataset_name", required=True, help="Name of the bridged input dir (visualization/ key; <scene>_mvsam3d_input for GenRecon runs)")
    parser.add_argument("--labels", required=True, help="Comma-separated raw class labels")
    parser.add_argument("--out_dir", required=True, help="Where to write <label>/mesh.glb")
    parser.add_argument(
        "--scene_pointcloud_dir", default=None,
        help="genrecon's COBGS_MASK_DIR (<run_dir>/stage_1_segmentation): "
        "per-object segmented point clouds, used as the target for the bbox fit when scoring yaws. "
        "Needs --mvsam3d_input_dir too; omit both to keep the raw MV-SAM3D pose.",
    )
    parser.add_argument(
        "--mvsam3d_input_dir", default=None,
        help="Bridged MV-SAM3D input dir (<run>/stage_2_mv_sam3d/input: da3_output.npz + per-label "
        "masks). When given with --scene_pointcloud_dir, each object's yaw is chosen by silhouette "
        "IoU against the real masks -- see silhouette_orientation.py.",
    )
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    failures = collect_mvsam3d_outputs(
        Path(args.visualization_dir),
        args.dataset_name,
        labels,
        Path(args.out_dir),
        scene_pointcloud_dir=Path(args.scene_pointcloud_dir) if args.scene_pointcloud_dir else None,
        mvsam3d_input_dir=Path(args.mvsam3d_input_dir) if args.mvsam3d_input_dir else None,
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
