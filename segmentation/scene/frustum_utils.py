"""Lift a 2D detection box into a 3D frustum using sparse COLMAP points, and
reproject that frustum into other frames.

Used to gate SAM2 tracker re-detection in
detect_and_segment.py: a detected
object's box is lifted to a 3D frustum once at its seed frame, then
reprojected into later frames to estimate whether the object should still be
substantially visible there, independent of whether SAM2's mask propagation
actually found it.
"""
import numpy as np

from scene.colmap_loader import qvec2rotmat


def _project_pinhole(xyz_world, extr, intr):
    """Project world-space points xyz_world (N,3) into (extr, intr)'s pinhole
    camera. Returns (u, v, z): pixel coords and camera-space depth. u/v are
    only meaningful where z > 0 (point is in front of the camera)."""
    assert intr.model == "PINHOLE", f"Unsupported camera model: {intr.model}"
    fx, fy, cx, cy = intr.params[:4]

    R = qvec2rotmat(extr.qvec)
    t = np.asarray(extr.tvec)

    cam_pts = xyz_world @ R.T + t  # COLMAP convention: X_cam = R @ X_world + t
    z = cam_pts[:, 2]
    z_safe = np.where(z > 1e-6, z, 1e-6)
    u = fx * cam_pts[:, 0] / z_safe + cx
    v = fy * cam_pts[:, 1] / z_safe + cy
    return u, v, z


def lift_box_to_frustum(xyz_world, extr, intr, box_xyxy, box_margin=0.0,
                         depth_percentile=(5, 95), min_points=3):
    """Estimate a 3D frustum for a 2D detection box by finding the sparse
    COLMAP points that project inside it and using their depth spread as the
    frustum's near/far extent.

    Returns the 8 world-space corners (np.ndarray[8,3]: the box's 4 corners
    unprojected at z_near, then the same 4 corners unprojected at z_far), or
    None if fewer than `min_points` sparse points land inside the box (not
    enough signal to estimate its depth extent).
    """
    xyz_world = np.asarray(xyz_world)
    u, v, z = _project_pinhole(xyz_world, extr, intr)

    x0, y0, x1, y1 = box_xyxy
    x0, y0, x1, y1 = x0 - box_margin, y0 - box_margin, x1 + box_margin, y1 + box_margin
    in_box = (z > 1e-6) & (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)

    if in_box.sum() < min_points:
        return None

    z_near, z_far = np.percentile(z[in_box], depth_percentile)
    if z_far <= z_near:
        z_far = z_near + 1e-3

    fx, fy, cx, cy = intr.params[:4]
    R = qvec2rotmat(extr.qvec)
    t = np.asarray(extr.tvec)

    corners_uv = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    corners_world = []
    for depth in (z_near, z_far):
        x_cam = (corners_uv[:, 0] - cx) / fx * depth
        y_cam = (corners_uv[:, 1] - cy) / fy * depth
        cam_pts = np.stack([x_cam, y_cam, np.full(4, depth)], axis=1)
        # X_cam = R @ X_world + t  =>  X_world = R.T @ (X_cam - t); as row
        # vectors this is (cam_pts - t) @ R.
        corners_world.append((cam_pts - t) @ R)
    return np.concatenate(corners_world, axis=0)  # (8, 3)


def reproject_frustum_area_fraction(corners_world, extr, intr):
    """Project a frustum's 8 world-space corners into (extr, intr) and
    return the axis-aligned bounding-box area of the in-front, image-clipped
    corners as a fraction of the frame area (0.0 if none are in front of the
    camera). A coarse but cheap "is this object still substantially in view"
    signal -- deliberately not a tight convex hull/polygon clip."""
    width, height = intr.width, intr.height
    u, v, z = _project_pinhole(np.asarray(corners_world), extr, intr)
    in_front = z > 1e-6
    if not np.any(in_front):
        return 0.0

    u_clipped = np.clip(u[in_front], 0, width)
    v_clipped = np.clip(v[in_front], 0, height)

    area = max(0.0, u_clipped.max() - u_clipped.min()) * max(0.0, v_clipped.max() - v_clipped.min())
    return float(area / (width * height))
