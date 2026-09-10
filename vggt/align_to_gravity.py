"""Rotate/translate a VGGT-Omega point cloud and camera poses into a
gravity-aligned reference frame (x/y horizontal, +z up).

Fits the dominant ground plane in the reconstructed point cloud via a
pure-numpy RANSAC (no open3d dependency) and rigidly transforms both the
point cloud and camera extrinsics into the new frame.

This module is standalone: it does not import ``demo_rerun``,
``visual_util``, or any torch-importing module in this repo, so it can be
used without loading the model.

Intended usage (not executed here)::

    # predictions_np = run_model(...)                          # demo_rerun.py
    # vertices, colors = filter_points(predictions_np, ...)     # visual_util.py
    # result = align_to_gravity(vertices, predictions_np["extrinsic"])
    # aligned_points = result.points
    # aligned_extrinsics = result.extrinsics
"""

from __future__ import annotations

import dataclasses
import os
import warnings
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path


# ---------------------------------------------------------------------------
# Rigid-transform helpers
# ---------------------------------------------------------------------------

def _to_4x4(mats: np.ndarray) -> np.ndarray:
    """Normalize (..., 3, 4) or (..., 4, 4) matrices to (..., 4, 4)."""
    mats = np.asarray(mats, dtype=np.float64)
    if mats.shape[-2:] == (4, 4):
        return mats.copy()
    if mats.shape[-2:] == (3, 4):
        out = np.zeros(mats.shape[:-2] + (4, 4), dtype=np.float64)
        out[..., 3, 3] = 1.0
        out[..., :3, :4] = mats
        return out
    raise ValueError(f"Expected matrices with trailing shape (3,4) or (4,4), got {mats.shape}")


def invert_rigid_transform(T: np.ndarray) -> np.ndarray:
    """Batched SE(3) inverse of (..., 4, 4) rigid transforms."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_t = np.swapaxes(R, -1, -2)
    t_new = -np.einsum("...ij,...j->...i", R_t, t)
    out = np.zeros_like(T)
    out[..., 3, 3] = 1.0
    out[..., :3, :3] = R_t
    out[..., :3, 3] = t_new
    return out


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    """World-space camera centers, shape (S, 3), from world-to-camera extrinsics."""
    world_to_cam = _to_4x4(extrinsics)
    cam_to_world = invert_rigid_transform(world_to_cam)
    return cam_to_world[..., :3, 3]


def apply_transform_to_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an (N, 3) array of points."""
    points = np.asarray(points, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return points @ R.T + t


def apply_transform_to_extrinsics(extrinsics: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Re-express world-to-camera extrinsics after the world frame is transformed by T.

    If p_new = T @ p_old and p_cam = E_old @ p_old, then E_new = E_old @ inv(T).
    Preserves the input's trailing shape, (3, 4) or (4, 4).
    """
    extrinsics = np.asarray(extrinsics)
    orig_rows = extrinsics.shape[-2]
    E_old = _to_4x4(extrinsics)
    T_inv = invert_rigid_transform(T)
    E_new = E_old @ T_inv[None, ...]
    if orig_rows == 3:
        return E_new[..., :3, :4]
    return E_new


# ---------------------------------------------------------------------------
# Plane fitting
# ---------------------------------------------------------------------------

def fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares plane fit ``a*x + b*y + c*z + d = 0`` via SVD.

    Returns (normal (3,) unit vector, d).
    """
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = Vt[-1]
    normal = normal / np.linalg.norm(normal)
    d = -normal @ centroid
    return normal, float(d)


def point_plane_signed_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Signed distance of each point to the plane defined by (normal, d). Assumes unit normal."""
    return points @ normal + d


def project_points_onto_plane(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Orthogonal projection of points onto the plane (normal, d)."""
    distance = point_plane_signed_distance(points, normal, d)
    return points - distance[:, None] * normal


def rotation_matrix_up_to_z(normal: np.ndarray) -> np.ndarray:
    """Rotation mapping world coords into a frame where +z is aligned with `normal`.

    v_new = R @ v_world. Axes 0/1 are in-plane tangent directions, axis 2 is
    the (assumed already up-pointing) plane normal.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)

    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 0.0, 1.0])

    t1 = np.cross(arbitrary, n)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)

    return np.stack([t1, t2, n], axis=0)


def disambiguate_normal_sign(
    normal: np.ndarray, d: float, plane_point: np.ndarray, reference_points: np.ndarray
) -> tuple[np.ndarray, float]:
    """Flip (normal, d) if needed so the normal points toward `reference_points`.

    Cameras are assumed to sit above the ground looking across the scene, so
    the mean camera center is used as the reference to disambiguate "up".
    """
    avg_dir = reference_points.mean(axis=0) - plane_point
    if normal @ avg_dir < 0:
        return -normal, -d
    return normal, d


@dataclasses.dataclass
class PlaneFitResult:
    normal: np.ndarray
    d: float
    inlier_mask: np.ndarray
    num_inliers: int


def ransac_fit_plane(
    points: np.ndarray,
    distance_threshold: float,
    num_iterations: int = 1000,
    min_samples: int = 3,
    seed: int | None = None,
) -> PlaneFitResult:
    """Pure-numpy RANSAC plane fit (replaces open3d's ``segment_plane``)."""
    rng = np.random.default_rng(seed)
    n_points = points.shape[0]
    if n_points < min_samples:
        raise ValueError(f"Need at least {min_samples} points, got {n_points}")

    best_inlier_mask = None
    best_num_inliers = -1

    for _ in range(num_iterations):
        sample_idx = rng.choice(n_points, size=min_samples, replace=False)
        sample = points[sample_idx]

        # Skip near-degenerate (collinear) samples.
        centered = sample - sample.mean(axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values[-2] < 1e-9:
            continue

        normal, d = fit_plane_svd(sample)
        distances = np.abs(point_plane_signed_distance(points, normal, d))
        inlier_mask = distances < distance_threshold
        num_inliers = int(inlier_mask.sum())

        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inlier_mask = inlier_mask

    if best_inlier_mask is None or best_num_inliers < min_samples:
        raise RuntimeError("RANSAC failed to find a valid plane (all samples degenerate)")

    # Refit via least-squares on the winning inlier set for numerical stability.
    normal, d = fit_plane_svd(points[best_inlier_mask])
    distances = np.abs(point_plane_signed_distance(points, normal, d))
    inlier_mask = distances < distance_threshold

    return PlaneFitResult(normal=normal, d=d, inlier_mask=inlier_mask, num_inliers=int(inlier_mask.sum()))


def build_gravity_alignment_transform(normal: np.ndarray, d: float, origin_point: np.ndarray) -> np.ndarray:
    """4x4 rigid transform from the old world frame into the gravity-aligned frame.

    `origin_point` must lie on the plane; it becomes the new frame's origin.
    """
    R = rotation_matrix_up_to_z(normal)
    t = -R @ origin_point
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AlignmentResult:
    points: np.ndarray
    extrinsics: np.ndarray
    transform: np.ndarray
    plane_normal: np.ndarray
    plane_d: float
    inlier_mask: np.ndarray
    num_inliers: int


def align_to_gravity(
    points: np.ndarray,
    extrinsics: np.ndarray,
    *,
    distance_threshold: float = 0.02,
    num_iterations: int = 1000,
    min_samples: int = 3,
    seed: int | None = None,
    max_ransac_points: int | None = None,
) -> AlignmentResult:
    """Rotate/translate a point cloud + camera poses into a gravity-aligned frame.

    Args:
        points: (N, 3) world-space point cloud, e.g. from ``visual_util.filter_points``.
        extrinsics: (S, 3, 4) or (S, 4, 4) world-to-camera extrinsics, OpenCV
            convention, e.g. ``predictions["extrinsic"]``.
        distance_threshold: RANSAC inlier distance threshold (in the point
            cloud's units).
        num_iterations: number of RANSAC iterations.
        min_samples: points per RANSAC minimal sample (3 for a plane).
        seed: RNG seed for reproducibility.
        max_ransac_points: if set and the cloud is larger, randomly subsample
            to this many points for the RANSAC search only (speed). The
            returned inlier mask is still computed against the full input
            `points`. No axis/height pre-filter is applied by default, since
            VGGT's world frame has no known up-axis a priori; callers with
            domain knowledge can pre-filter `points` themselves.

    Returns:
        AlignmentResult with the aligned point cloud, aligned extrinsics
        (same shape as input), the 4x4 old-world-to-new-world transform, the
        fitted plane (in the *old* frame, up-disambiguated), and the inlier
        mask/count w.r.t. the full input point cloud.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if extrinsics.shape[-2:] not in ((3, 4), (4, 4)):
        raise ValueError(f"extrinsics must have trailing shape (3,4) or (4,4), got {extrinsics.shape}")

    rng = np.random.default_rng(seed)
    if max_ransac_points is not None and points.shape[0] > max_ransac_points:
        sub_idx = rng.choice(points.shape[0], size=max_ransac_points, replace=False)
        ransac_points = points[sub_idx]
    else:
        ransac_points = points

    fit = ransac_fit_plane(
        ransac_points,
        distance_threshold=distance_threshold,
        num_iterations=num_iterations,
        min_samples=min_samples,
        seed=seed,
    )

    if fit.num_inliers < max(min_samples, int(0.01 * ransac_points.shape[0])):
        warnings.warn(
            f"align_to_gravity: RANSAC found only {fit.num_inliers} ground-plane inliers "
            f"out of {ransac_points.shape[0]} points; the fitted plane may not be reliable.",
            stacklevel=2,
        )

    centers = camera_centers_from_extrinsics(extrinsics)
    inlier_points = ransac_points[fit.inlier_mask]
    centroid = inlier_points.mean(axis=0)

    normal, d = disambiguate_normal_sign(fit.normal, fit.d, plane_point=centroid, reference_points=centers)
    origin_point = project_points_onto_plane(centroid[None, :], normal, d)[0]

    T = build_gravity_alignment_transform(normal, d, origin_point)

    aligned_points = apply_transform_to_points(points, T)
    aligned_extrinsics = apply_transform_to_extrinsics(extrinsics, T)

    full_inlier_mask = np.abs(point_plane_signed_distance(points, normal, d)) < distance_threshold

    return AlignmentResult(
        points=aligned_points,
        extrinsics=aligned_extrinsics,
        transform=T,
        plane_normal=normal,
        plane_d=d,
        inlier_mask=full_inlier_mask,
        num_inliers=int(full_inlier_mask.sum()),
    )


def rotate_horizontal(result: AlignmentResult, angle_degrees: float) -> AlignmentResult:
    """Rotate an already gravity-aligned result about the vertical (+z) axis.

    Args:
        result: an `AlignmentResult` produced by `align_to_gravity`.
        angle_degrees: rotation angle about +z, in degrees (counter-clockwise
            when viewed from +z looking down at the x/y plane).

    Returns:
        A new `AlignmentResult` with `points`/`extrinsics`/`transform`
        updated; `plane_normal`/`plane_d`/`inlier_mask`/`num_inliers` are
        carried over unchanged since they describe the RANSAC fit against
        the original input frame.
    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    T_rot = np.eye(4)
    T_rot[:2, :2] = [[cos_a, -sin_a], [sin_a, cos_a]]

    rotated_points = apply_transform_to_points(result.points, T_rot)
    rotated_extrinsics = apply_transform_to_extrinsics(result.extrinsics, T_rot)

    return AlignmentResult(
        points=rotated_points,
        extrinsics=rotated_extrinsics,
        transform=T_rot @ result.transform,
        plane_normal=result.plane_normal,
        plane_d=result.plane_d,
        inlier_mask=result.inlier_mask,
        num_inliers=result.num_inliers,
    )


# ---------------------------------------------------------------------------
# COLMAP text I/O
# ---------------------------------------------------------------------------

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert a scalar-first quaternion [qw, qx, qy, qz] to a 3x3 rotation matrix."""
    return Rotation.from_quat(qvec, scalar_first=True).as_matrix()


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a scalar-first quaternion [qw, qx, qy, qz]."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def write_points3D_ply(path: Path, vertices: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian PLY point cloud, importable by MeshLab."""
    colors_u8 = colors.clip(0, 255).astype(np.uint8)
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vertex_data["red"], vertex_data["green"], vertex_data["blue"] = colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())


def read_colmap_images_txt(path: Path) -> tuple[list[tuple[int, int, str]], np.ndarray, list[str]]:
    """Parse a COLMAP ``images.txt``.

    Returns:
        records: list of (image_id, camera_id, name), in file order.
        extrinsics: (S, 4, 4) world-to-camera matrices.
        points2d_lines: raw second line (POINTS2D) per image, kept verbatim.
    """
    records: list[tuple[int, int, str]] = []
    extrinsics: list[np.ndarray] = []
    points2d_lines: list[str] = []

    # Fixed 2 lines per image (pose line, then a POINTS2D line that is
    # usually empty). Skip stray blank lines only where a pose line is
    # expected; a blank POINTS2D line must not be filtered out, or the
    # pairing would shift and silently drop every other image.
    raw_lines = path.read_text().splitlines()
    i = 0
    while i < len(raw_lines):
        pose_line = raw_lines[i]
        if pose_line.strip() == "":
            i += 1
            continue
        points2d_line = raw_lines[i + 1] if i + 1 < len(raw_lines) else ""
        i += 2

        tokens = pose_line.split(" ", 9)
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id, name = tokens
        E = np.eye(4)
        E[:3, :3] = qvec2rotmat(np.array([float(qw), float(qx), float(qy), float(qz)]))
        E[:3, 3] = [float(tx), float(ty), float(tz)]
        records.append((int(image_id), int(camera_id), name))
        extrinsics.append(E)
        points2d_lines.append(points2d_line)

    return records, np.stack(extrinsics, axis=0), points2d_lines


def write_colmap_images_txt(
    path: Path,
    records: list[tuple[int, int, str]],
    extrinsics: np.ndarray,
    points2d_lines: list[str],
) -> None:
    """Write a COLMAP ``images.txt`` from (possibly transformed) extrinsics."""
    extrinsics = _to_4x4(extrinsics)
    with open(path, "w") as f:
        for (image_id, camera_id, name), E, points2d_line in zip(records, extrinsics, points2d_lines):
            qw, qx, qy, qz = rotmat2qvec(E[:3, :3])
            tx, ty, tz = E[:3, 3]
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {name}\n")
            f.write(f"{points2d_line}\n")


def read_colmap_points3D_txt(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Parse a COLMAP ``points3D.txt``.

    Returns:
        ids: (N,) point ids.
        xyz: (N, 3) point positions.
        rgb: (N, 3) uint8 colors.
        errors: (N,) reprojection errors.
        track_suffixes: raw trailing tokens (TRACK[] data) per line, kept verbatim.
    """
    ids, xyz, rgb, errors, track_suffixes = [], [], [], [], []
    for line in path.read_text().splitlines():
        if line.strip() == "":
            continue
        tokens = line.split(" ")
        point_id, x, y, z, r, g, b, error = tokens[:8]
        ids.append(int(point_id))
        xyz.append([float(x), float(y), float(z)])
        rgb.append([int(r), int(g), int(b)])
        errors.append(float(error))
        track_suffixes.append(" ".join(tokens[8:]))

    return np.array(ids), np.array(xyz), np.array(rgb), np.array(errors), track_suffixes


def write_colmap_points3D_txt(
    path: Path,
    ids: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    errors: np.ndarray,
    track_suffixes: list[str],
) -> None:
    """Write a COLMAP ``points3D.txt`` from (possibly transformed) point positions."""
    with open(path, "w") as f:
        for point_id, (x, y, z), (r, g, b), error, track_suffix in zip(ids, xyz, rgb, errors, track_suffixes):
            line = f"{point_id} {x} {y} {z} {r} {g} {b} {error}"
            if track_suffix:
                line += f" {track_suffix}"
            f.write(line + "\n")


def _symlink_relative(target: Path, link_path: Path) -> None:
    """Create/replace `link_path` as a symlink pointing at `target`, using a relative path."""
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(os.path.relpath(target, link_path.parent))


# ---------------------------------------------------------------------------
# Export orchestration
# ---------------------------------------------------------------------------

def export_aligned_colmap_dataset(
    input_dir: Path,
    output_dir: Path,
    result: AlignmentResult,
) -> None:
    """Write an already-computed `AlignmentResult` as a COLMAP dataset on disk.

    Mirrors `input_dir`'s directory structure into `output_dir`: `images/`
    and `sparse/0/cameras.txt` are symlinked (unaffected by a rigid world
    transform), while `sparse/0/images.txt`, `points3D.txt`, and
    `points3D.ply` are rewritten with `result`'s aligned poses/points. Does
    not itself run `align_to_gravity`/`rotate_horizontal` — pass in a result
    already produced by those.
    """
    input_sparse = input_dir / "sparse" / "0"
    output_sparse = output_dir / "sparse" / "0"
    output_sparse.mkdir(parents=True, exist_ok=True)

    _symlink_relative(input_dir / "images", output_dir / "images")
    _symlink_relative(input_sparse / "cameras.txt", output_sparse / "cameras.txt")

    ids, _, rgb, errors, track_suffixes = read_colmap_points3D_txt(input_sparse / "points3D.txt")
    records, _, points2d_lines = read_colmap_images_txt(input_sparse / "images.txt")

    write_colmap_images_txt(output_sparse / "images.txt", records, result.extrinsics, points2d_lines)
    write_colmap_points3D_txt(output_sparse / "points3D.txt", ids, result.points, rgb, errors, track_suffixes)
    write_points3D_ply(output_sparse / "points3D.ply", result.points, rgb)

    print(
        f"Exported gravity-aligned COLMAP dataset to {output_dir} "
        f"({len(records)} frames, {len(ids)} points, {result.num_inliers} ground-plane inliers)"
    )


if __name__ == "__main__":
    # Synthetic smoke test: a tilted ground plane plus a few cameras above it.
    render: bool = False
    output_dir: Path = Path("/home/gabis/Work/GitHub/COB-GS/dataset/kitchen_with_floor_aligned")
    input_dir: Path = Path("/home/gabis/Work/GitHub/COB-GS/dataset/kitchen_with_floor")
    point_cloud_data = np.loadtxt(str(input_dir/"sparse"/"0"/"points3D.txt")) 
    poses = []
    with open(input_dir/"sparse"/"0"/"images.txt", "r") as f:
        for line in f.readlines():
            if line == "\n":
                continue
            image_id, rotw, rotx, roty, rotz, tx, ty, tz, _, filename = line.split(" ")
            pose = np.eye(4)
            pose[:3, 3] = np.array([float(tx), float(ty), float(tz)])
            rot: Rotation = Rotation.from_quat([float(rotw), float(rotx), float(roty), float(rotz)], scalar_first=True)
            pose[:3, :3] = rot.as_matrix()
            poses.append(pose)
    poses = np.array(poses)
    point_cloud = point_cloud_data[:, 1:4]


    colors = point_cloud_data[:, 4:7].astype(np.uint8)
           
    result: AlignmentResult = align_to_gravity(point_cloud, poses, distance_threshold=0.05, seed=0)
    horizontal_aligned_result: AlignmentResult = rotate_horizontal(result=result, angle_degrees=-25.0)
    export_aligned_colmap_dataset(input_dir, output_dir, horizontal_aligned_result)

    if render:
        rr.init("visualize-chair", spawn=True)
        # Initial eye looks along -Z (top-down/bird's-eye view of the gravity-aligned scene).
        rr.log("world", rr.ViewCoordinates.RUB, static=True)
        #rr.log("world/pcl", rr.Points3D(point_cloud, colors=colors))
        rr.log("world/pcl_aligned", rr.Points3D(horizontal_aligned_result.points, colors=colors))

        camera_intrinsic = np.array(
            [
                [403.1188659667969, 0.0, 296.0],
                [0.0, 403.50445556640625, 224.0],
                [0.0, 0.0, 1.0],
            ]
        )
        camera_to_world_aligned = invert_rigid_transform(_to_4x4(result.extrinsics))
        for i in range(len(camera_to_world_aligned)):
            rr.log(
                f"world/camera_{i}",
                rr.Transform3D(
                    translation=camera_to_world_aligned[i, :3, 3],
                    mat3x3=camera_to_world_aligned[i, :3, :3],
                ),
            )
            rr.log(
                f"world/camera_{i}/image",
                rr.Pinhole(image_from_camera=camera_intrinsic, width=592, height=448),
            )