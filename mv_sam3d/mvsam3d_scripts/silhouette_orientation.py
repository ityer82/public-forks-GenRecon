"""
Pick an object mesh's yaw by silhouette IoU against the real per-view object masks.

Used by collect_mvsam3d_outputs.py after the SAM3D pose has been mapped into the scene's
gravity-aligned (Z-up) world frame. Candidates are the untouched SAM3D pose plus rotations
about the vertical axis through the mesh centroid; each is rendered into the DA3 cameras
(after the same bbox scale+translation fit Stage 10 applies, so orientation is the only
difference) and scored by mean IoU with the object's SAM mask. Unlike a one-sided 3D ICP
cost against a partial point cloud, the silhouettes make an upside-down or back-to-front
pose score clearly worse.
"""
from pathlib import Path

import cv2
import numpy as np

MIN_MASK_PIXELS = 400


def load_views(input_dir: Path, label: str, max_views: int = 12) -> list[tuple]:
    """Returns [(stem, extrinsic 3x4 world->cam, intrinsic 3x3, bool mask HxW)], evenly
    subsampled to `max_views`, from <input_dir>/da3_output.npz and <input_dir>/<label>/<stem>.png
    (RGBA, alpha = object). Views whose mask is empty/tiny are skipped."""
    input_dir = Path(input_dir)
    da3 = np.load(input_dir / "da3_output.npz")
    stems = [Path(str(p)).stem for p in da3["image_files"]]
    views = []
    for mask_path in sorted((input_dir / label).glob("*.png")):
        if mask_path.stem not in stems:
            continue
        img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3 and img.shape[2] == 4:
            mask = img[..., 3] > 0
        else:
            mask = (img.max(-1) if img.ndim == 3 else img) > 0
        if mask.sum() < MIN_MASK_PIXELS:
            continue
        i = stems.index(mask_path.stem)
        views.append(
            (mask_path.stem, da3["extrinsics"][i].astype(np.float64), da3["intrinsics"][i].astype(np.float64), mask)
        )
    if len(views) > max_views:
        idx = np.linspace(0, len(views) - 1, max_views).round().astype(int)
        views = [views[i] for i in idx]
    return views


def bbox_fit(points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Uniform scale (longest bbox side) + bbox-center translation onto the target's
    (1st-99th percentile) bbox, mirroring align_meshes_to_scene.py's fit."""
    s_min, s_max = points.min(0), points.max(0)
    t_min, t_max = np.percentile(target_points, 1, axis=0), np.percentile(target_points, 99, axis=0)
    scale = (t_max - t_min).max() / max((s_max - s_min).max(), 1e-9)
    return (points - (s_min + s_max) / 2) * scale + (t_min + t_max) / 2


def render_silhouette(points: np.ndarray, ext: np.ndarray, K: np.ndarray, hw: tuple, radius: int = 2) -> np.ndarray:
    """Splats surface points into an HxW boolean silhouette (dilate + close to fill gaps)."""
    h, w = hw
    cam = points @ ext[:, :3].T + ext[:, 3]
    cam = cam[cam[:, 2] > 1e-3]
    uv = cam @ K.T
    ij = np.round(uv[:, :2] / uv[:, 2:3]).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    img = np.zeros((h, w), np.uint8)
    img[ij[ok, 1], ij[ok, 0]] = 1
    img = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0


def mean_iou(points: np.ndarray, views: list[tuple]) -> float:
    ious = []
    for _stem, ext, K, mask in views:
        sil = render_silhouette(points, ext, K, mask.shape)
        ious.append((sil & mask).sum() / max((sil | mask).sum(), 1))
    return float(np.mean(ious))


def yaw_matrix(yaw_deg: float, center: np.ndarray) -> np.ndarray:
    """4x4 rotation about the vertical (+Z) axis through `center`."""
    t = np.radians(yaw_deg)
    c, s = np.cos(t), np.sin(t)
    m = np.eye(4)
    m[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    m[:3, 3] = center - m[:3, :3] @ center
    return m


def select_best_yaw(
    world_points: np.ndarray, target_points: np.ndarray, views: list[tuple], yaw_step_deg: float = 10.0
) -> dict:
    """Scores the SAM3D pose (yaw 0) and yaw-swept variants of it; returns the best.

    Returns {"matrix": 4x4 for the best yaw, "yaw_deg", "iou", "identity_iou",
    "iou_by_yaw": {yaw_deg: iou}}. `world_points` are surface samples of the mesh in the
    scene world frame."""
    center = world_points.mean(axis=0)
    iou_by_yaw = {}
    for yaw in np.arange(0.0, 360.0, yaw_step_deg):
        m = yaw_matrix(yaw, center)
        rotated = world_points @ m[:3, :3].T + m[:3, 3]
        iou_by_yaw[float(yaw)] = mean_iou(bbox_fit(rotated, target_points), views)
    best_yaw = max(iou_by_yaw, key=iou_by_yaw.get)
    return {
        "matrix": yaw_matrix(best_yaw, center),
        "yaw_deg": best_yaw,
        "iou": iou_by_yaw[best_yaw],
        "identity_iou": iou_by_yaw[0.0],
        "iou_by_yaw": iou_by_yaw,
    }
