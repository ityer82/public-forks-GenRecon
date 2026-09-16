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
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import Transform3d, quaternion_to_matrix

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


def collect_mvsam3d_outputs(
    visualization_dir: Path, dataset_name: str, labels: list[str], out_dir: Path
) -> list[str]:
    """Collects per-object result.glb + params.npz into <out_dir>/<label>/mesh.glb, transformed
    into the input data's world frame. Returns the list of labels that failed/were missing
    (empty on full success) instead of calling sys.exit(1), so a caller can decide policy."""
    visualization_dir = Path(visualization_dir)
    out_dir = Path(out_dir)

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

        label_out_dir = out_dir / label
        label_out_dir.mkdir(parents=True, exist_ok=True)
        dest = label_out_dir / "mesh.glb"
        scene.export(str(dest))
        print(f"[collect_mvsam3d_outputs] {label}: {obj_dir / 'result.glb'} -> {dest}")

    if failures:
        print(f"[collect_mvsam3d_outputs] Failed/missing labels: {failures}")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualization_dir", default="visualization", help="MV-SAM3D's visualization/ dir")
    parser.add_argument("--dataset_name", required=True, help="Name of the bridged input dir (input_path.name)")
    parser.add_argument("--labels", required=True, help="Comma-separated raw class labels")
    parser.add_argument("--out_dir", required=True, help="Where to write <label>/mesh.glb")
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    failures = collect_mvsam3d_outputs(Path(args.visualization_dir), args.dataset_name, labels, Path(args.out_dir))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
