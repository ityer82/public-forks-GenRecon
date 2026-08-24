"""Convex-hull + reprojection-tightening helpers for a segmented object's point cloud.

Mesh-agnostic: these only need an object's point cloud and (optionally) the
scene's COLMAP cameras + per-frame segmentation masks, so they're shared by
both a post-hoc mesh crop (scripts/extract_object_mesh.py) and a
pre-generation voxel exclusion (reconstruct_scene.py's --exclude_masks_root).
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from genrecon.utils.colmap_utils import project_points
from genrecon.utils.logger import logger


def load_object_points(object_ply: Path) -> np.ndarray:
    ply = PlyData.read(object_ply)
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def padded_hull_equations(points: np.ndarray, padding: float) -> np.ndarray:
    hull = ConvexHull(points)
    equations = hull.equations.copy()  # rows [a, b, c, d]; interior iff a*x+b*y+c*z+d <= 0
    equations[:, 3] -= padding
    return equations


def padded_hull_vertices(points: np.ndarray, padding: float) -> np.ndarray:
    """Vertices of the hull after offsetting every facet outward by `padding`.

    Padding moves facet planes, not the original points, so the padded
    polytope's vertices must be re-derived from the offset half-spaces
    rather than just nudging `points` outward.
    """
    equations = padded_hull_equations(points, padding)
    interior_point = points.mean(axis=0)
    hs = HalfspaceIntersection(equations, interior_point)
    return hs.intersections


def resolve_mask_path(mask_dir: Path, frame_name: str) -> Path | None:
    """Find the mask file for `frame_name`, tolerating extension mismatches
    (e.g. COLMAP image names ending in .jpg vs. exported .png masks)."""
    exact = mask_dir / frame_name
    if exact.exists():
        return exact
    stem = Path(frame_name).stem
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_mask_resized(mask_path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Read a mask PNG as bool and resize (nearest-neighbor) to `target_hw`
    if it doesn't already match (e.g. masks produced at a downsampled scale)."""
    mask = np.array(Image.open(mask_path).convert("L"))
    if mask.shape != tuple(target_hw):
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def reprojection_overshoot(padded_vertices: np.ndarray, cam: dict, mask: np.ndarray) -> float | None:
    """Fraction of the padded hull's reprojected 2D silhouette that falls
    outside the true segmentation `mask` for this camera view. Returns None
    if too few hull vertices project on-screen to form a meaningful polygon.
    """
    pixels, in_front = project_points(padded_vertices, cam)
    W, H = cam["W"], cam["H"]
    on_screen = in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] < W) & (pixels[:, 1] >= 0) & (pixels[:, 1] < H)
    visible_pixels = pixels[in_front]
    if visible_pixels.shape[0] < 3 or on_screen.sum() == 0:
        return None

    try:
        hull2d = ConvexHull(visible_pixels)
    except QhullError:
        return None
    polygon = visible_pixels[hull2d.vertices].round().astype(np.int32)

    silhouette = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(silhouette, [polygon], color=1)
    silhouette = silhouette.astype(bool)

    silhouette_area = silhouette.sum()
    if silhouette_area == 0:
        return None
    outside = np.count_nonzero(silhouette & ~mask)
    return outside / silhouette_area


def search_hull_padding(
    points: np.ndarray,
    cameras: list[dict],
    masks_dir: Path,
    padding_min: float,
    padding_max: float,
    max_overshoot: float,
    iters: int,
) -> float:
    """Binary-search the largest padding in [padding_min, padding_max] whose
    reprojected hull silhouette overshoots the true per-frame masks by no
    more than `max_overshoot` on average. Falls back to `padding_max` if no
    camera has a usable mask, or to `padding_min` if even that overshoots.
    """
    qualifying_cams: list[tuple[dict, np.ndarray]] = []
    for cam in cameras:
        mask_path = resolve_mask_path(masks_dir, cam["name"])
        if mask_path is None:
            continue
        mask = load_mask_resized(mask_path, (cam["H"], cam["W"]))
        if not mask.any():
            continue
        qualifying_cams.append((cam, mask))

    if not qualifying_cams:
        logger.warning(f"No usable masks found in {masks_dir}; using --hull_padding={padding_max} as-is.")
        return padding_max

    def score(padding: float) -> float:
        try:
            verts = padded_hull_vertices(points, padding)
        except QhullError:
            return float("inf")
        overshoots = [
            o for cam, mask in qualifying_cams if (o := reprojection_overshoot(verts, cam, mask)) is not None
        ]
        if not overshoots:
            return float("inf")
        return float(np.mean(overshoots))

    if score(padding_max) <= max_overshoot:
        return padding_max
    if score(padding_min) > max_overshoot:
        logger.warning(
            f"Even --hull_padding_min={padding_min} overshoots masks in {masks_dir}; using it as-is."
        )
        return padding_min

    lo, hi = padding_min, padding_max
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if score(mid) <= max_overshoot:
            lo = mid
        else:
            hi = mid
    return lo
