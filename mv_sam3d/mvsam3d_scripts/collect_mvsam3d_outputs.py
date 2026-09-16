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
bad rotation (align_trellis2_mesh_to_scene.py only ever fits scale + translation for
the mvsam3d backend). If `--scene_pointcloud_dir` is given, each object's world-frame
mesh is additionally checked against genrecon's own independently-reconstructed
segmented point cloud for that label (already gravity-aligned, scene-frame, and not
derived from MV-SAM3D's pose regression) and re-oriented onto it when they disagree --
see `gravity_snap_correction()`.
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from pytorch3d.transforms import Transform3d, quaternion_to_matrix
from trimesh.registration import icp

Z_UP_TO_Y_UP = np.array(
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    dtype=np.float32,
)
P3D_TO_CV = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)

# The 24 orientation-preserving (det=+1) signed permutation matrices, i.e. the
# rotational symmetry group of a cube -- used as coarse-alignment seeds in
# gravity_snap_correction() so ICP doesn't get stuck in the near-symmetric-leg
# local minima a naive identity-seeded ICP falls into for chair-like objects.
def _cube_group_rotations() -> list[np.ndarray]:
    import itertools

    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            p = np.zeros((3, 3))
            for i, j in enumerate(perm):
                p[i, j] = signs[i]
            if abs(np.linalg.det(p) - 1.0) < 1e-6:
                rotations.append(p)
    return rotations


_CUBE_GROUP_ROTATIONS = _cube_group_rotations()


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


def _pca_frame(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    cov = np.cov((points - center).T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(-evals)]
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1
    return evecs


def _leg_plane_tilt_deg(points: np.ndarray, low_frac: float = 0.15) -> float:
    """Angle (deg) between world-Z and the normal of a plane fit to the lowest `low_frac`
    of points by height -- a proxy for how far a resting-on-the-floor object's support
    plane (chair/table legs, etc.) is tilted off horizontal. Used only for diagnostics/
    logging, not for computing the correction itself."""
    z = points[:, 2]
    low = points[z <= np.quantile(z, low_frac)]
    if len(low) < 4:
        return float("nan")
    centroid = low.mean(axis=0)
    _, _, vt = np.linalg.svd(low - centroid, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))


def gravity_snap_correction(
    mesh_points: np.ndarray, target_points: np.ndarray, *, n_samples: int = 8000
) -> tuple[np.ndarray, dict]:
    """Finds the rigid rotation (about the shared centroid, no scale/reflection) that best
    registers `mesh_points` (MV-SAM3D's world-frame mesh, sampled) onto `target_points`
    (genrecon's own independently-reconstructed, gravity-aligned segmented point cloud for
    the same object) -- i.e. recovers the correct object orientation when MV-SAM3D's own
    predicted rotation/reference-view pose was wrong.

    Naive identity-seeded ICP gets stuck in local minima for near-symmetric objects (e.g.
    a 4-legged chair looks similar under several 90/180-deg flips), so this instead seeds
    ICP from all 24 candidate rotations mapping the source's PCA frame onto the target's
    PCA frame (the proper-rotation symmetries of a cube) and keeps the lowest-cost result.

    Returns (4x4 rigid transform, diagnostics dict).
    """
    if len(mesh_points) > n_samples:
        idx = np.random.default_rng(0).choice(len(mesh_points), n_samples, replace=False)
        mesh_points = mesh_points[idx]

    t_center = target_points.mean(axis=0)
    s_center = mesh_points.mean(axis=0)
    target_frame = _pca_frame(target_points, t_center)
    source_frame = _pca_frame(mesh_points, s_center)

    best_matrix, best_cost = None, np.inf
    for perm in _CUBE_GROUP_ROTATIONS:
        r0 = target_frame @ perm @ source_frame.T
        init = np.eye(4)
        init[:3, :3] = r0
        init[:3, 3] = t_center - r0 @ s_center
        matrix, _transformed, cost = icp(
            mesh_points, target_points, initial=init, max_iterations=50, threshold=1e-7,
            reflection=False, scale=False,
        )
        if cost < best_cost:
            best_matrix, best_cost = matrix, cost

    transformed = (best_matrix[:3, :3] @ mesh_points.T).T + best_matrix[:3, 3]
    identity_cost = float(np.mean(np.sum((mesh_points - target_points[
        _nearest_index(mesh_points, target_points)
    ]) ** 2, axis=1)))
    diagnostics = {
        "cost": float(best_cost),
        "rms_dist_m": float(np.sqrt(best_cost)),
        "identity_rms_dist_m": float(np.sqrt(identity_cost)),
        "rotation_angle_deg": float(
            np.degrees(np.arccos(np.clip((np.trace(best_matrix[:3, :3]) - 1) / 2, -1.0, 1.0)))
        ),
        "leg_plane_tilt_before_deg": _leg_plane_tilt_deg(mesh_points),
        "leg_plane_tilt_after_deg": _leg_plane_tilt_deg(transformed),
        "target_leg_plane_tilt_deg": _leg_plane_tilt_deg(target_points),
    }
    return best_matrix, diagnostics


def _nearest_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree

    _dist, idx = cKDTree(b).query(a, 1)
    return idx


# A correction is only trusted (and applied) when it clears both bars: the fit has to be
# tight relative to the object's own size, and it has to be a real improvement over doing
# nothing -- otherwise a wrong 24-candidate seed could win on cost without being correct,
# or the original MV-SAM3D pose may already have been fine.
MAX_RMS_FRACTION_OF_EXTENT = 0.12
MIN_IMPROVEMENT_RATIO = 0.7
MIN_TARGET_POINTS = 200
WARN_POST_CORRECTION_TILT_DEG = 10.0


def collect_mvsam3d_outputs(
    visualization_dir: Path,
    dataset_name: str,
    labels: list[str],
    out_dir: Path,
    scene_pointcloud_dir: Path | None = None,
) -> list[str]:
    """Collects per-object result.glb + params.npz into <out_dir>/<label>/mesh.glb, transformed
    into the input data's world frame. Returns the list of labels that failed/were missing
    (empty on full success) instead of calling sys.exit(1), so a caller can decide policy.

    If `scene_pointcloud_dir` is given (genrecon's COBGS_MASK_DIR, i.e.
    <genrecon_output>/segmentation_raw/masks/classes), each object is checked against its
    genrecon-reconstructed segmented point cloud at
    `scene_pointcloud_dir/<sanitized_label>/point_cloud/<sanitized_label>.ply` and rigidly
    re-oriented onto it via gravity_snap_correction() when they disagree -- see that
    function's docstring. Missing/sparse point clouds are skipped with a warning (not a
    failure): the object keeps its MV-SAM3D-only pose, same as before this was added.
    """
    visualization_dir = Path(visualization_dir)
    out_dir = Path(out_dir)
    scene_pointcloud_dir = Path(scene_pointcloud_dir) if scene_pointcloud_dir else None

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

        if scene_pointcloud_dir is not None:
            _maybe_apply_gravity_snap(scene, label, scene_pointcloud_dir)

        label_out_dir = out_dir / label
        label_out_dir.mkdir(parents=True, exist_ok=True)
        dest = label_out_dir / "mesh.glb"
        scene.export(str(dest))
        print(f"[collect_mvsam3d_outputs] {label}: {obj_dir / 'result.glb'} -> {dest}")

    if failures:
        print(f"[collect_mvsam3d_outputs] Failed/missing labels: {failures}")
    return failures


def _maybe_apply_gravity_snap(scene: trimesh.Scene, label: str, scene_pointcloud_dir: Path) -> None:
    tag = "[collect_mvsam3d_outputs][gravity_snap]"
    pc_path = scene_pointcloud_dir / sanitize_label(label) / "point_cloud" / f"{sanitize_label(label)}.ply"
    if not pc_path.exists():
        print(f"{tag} {label}: no reference point cloud at {pc_path}, skipping check")
        return

    target_points = np.asarray(trimesh.load(str(pc_path)).vertices, dtype=np.float64)
    if len(target_points) < MIN_TARGET_POINTS:
        print(f"{tag} {label}: reference point cloud too sparse ({len(target_points)} pts), skipping check")
        return

    mesh = scene.to_mesh()
    mesh_points, _ = trimesh.sample.sample_surface(mesh, min(8000, len(mesh.vertices) * 4))
    mesh_points = np.asarray(mesh_points, dtype=np.float64)

    matrix, diag = gravity_snap_correction(mesh_points, target_points)
    extent = float(np.linalg.norm(target_points.max(axis=0) - target_points.min(axis=0)))
    rms_ok = extent > 0 and diag["rms_dist_m"] < MAX_RMS_FRACTION_OF_EXTENT * extent
    improved = diag["rms_dist_m"] < MIN_IMPROVEMENT_RATIO * diag["identity_rms_dist_m"]

    print(
        f"{tag} {label}: rotation={diag['rotation_angle_deg']:.1f}deg rms={diag['rms_dist_m']:.4f}m "
        f"(identity_rms={diag['identity_rms_dist_m']:.4f}m, extent={extent:.3f}m) "
        f"leg_tilt before={diag['leg_plane_tilt_before_deg']:.1f}deg "
        f"after={diag['leg_plane_tilt_after_deg']:.1f}deg "
        f"target={diag['target_leg_plane_tilt_deg']:.1f}deg"
    )

    if not (rms_ok and improved):
        print(f"{tag} {label}: correction not applied (fit not tight/better enough), keeping MV-SAM3D pose")
        return

    scene.apply_transform(matrix)
    print(f"{tag} {label}: applied gravity-snap correction ({diag['rotation_angle_deg']:.1f}deg rotation)")
    if diag["leg_plane_tilt_after_deg"] > WARN_POST_CORRECTION_TILT_DEG:
        print(
            f"{tag} {label}: WARNING: still {diag['leg_plane_tilt_after_deg']:.1f}deg off "
            "vertical after correction, recommend manual review"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visualization_dir", default="visualization", help="MV-SAM3D's visualization/ dir")
    parser.add_argument("--dataset_name", required=True, help="Name of the bridged input dir (input_path.name)")
    parser.add_argument("--labels", required=True, help="Comma-separated raw class labels")
    parser.add_argument("--out_dir", required=True, help="Where to write <label>/mesh.glb")
    parser.add_argument(
        "--scene_pointcloud_dir", default=None,
        help="genrecon's COBGS_MASK_DIR (<genrecon_output>/segmentation_raw/masks/classes). "
        "When given, each object's pose is checked/corrected against its genrecon-reconstructed "
        "segmented point cloud -- see gravity_snap_correction(). Omit to keep the old behavior.",
    )
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    failures = collect_mvsam3d_outputs(
        Path(args.visualization_dir),
        args.dataset_name,
        labels,
        Path(args.out_dir),
        scene_pointcloud_dir=Path(args.scene_pointcloud_dir) if args.scene_pointcloud_dir else None,
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
