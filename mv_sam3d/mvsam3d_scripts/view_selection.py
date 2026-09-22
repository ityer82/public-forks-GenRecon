#!/usr/bin/env python3
"""
Angular-coverage view pruning for MV-SAM3D's multi-view fusion.

MV-SAM3D's multidiffusion fusion runs a full generator forward pass per conditioning
view, at every denoising step (see sam3d_objects/pipeline/multi_view_utils.py), so
runtime scales ~linearly with the number of views used. This module selects a subset
of the available views per object that best covers the object angularly, using camera
poses (from da3_output.npz) and an estimate of the object's 3D position, so the main
diffusion pass can be run on fewer, well-spread views instead of every view with a mask.

Must run in MV-SAM3D's own venv (needs numpy, trimesh).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)


def sanitize_label(label: str) -> str:
    """Mirrors genrecon/pipeline/stages.py's and collect_mvsam3d_outputs.py's sanitize_label
    (spaces/slashes -> underscores), used to resolve the segmentation stage's per-object
    point cloud directory."""
    return label.replace(" ", "_").replace("/", "_")


def compute_object_centroid_from_pointcloud(ply_path: Path) -> Optional[np.ndarray]:
    """Load the segmentation stage's per-object point cloud and return its centroid.

    These point clouds (<scene_pointcloud_dir>/<sanitize_label(label)>/point_cloud/
    <sanitize_label(label)>.ply) are already in the same gravity-aligned, world/scene
    frame as the DA3 camera extrinsics (see collect_mvsam3d_outputs.py's gravity-snap
    docstring), so no extra alignment is needed here.

    Returns None if the file doesn't exist or fails to load, so callers can fall back
    to compute_object_centroid_from_pointmaps().
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        return None
    try:
        import trimesh

        mesh_or_cloud = trimesh.load(str(ply_path))
        vertices = np.asarray(mesh_or_cloud.vertices)
        if vertices.size == 0:
            return None
        return vertices.mean(axis=0)
    except Exception as e:  # noqa: BLE001 - best-effort fallback source
        logger.warning(f"[ViewSelection] Failed to load point cloud {ply_path}: {e}")
        return None


def compute_object_centroid_from_pointmaps(
    view_pointmaps: Sequence[np.ndarray],
    view_masks: Sequence[np.ndarray],
    camera_poses: Sequence[dict],
) -> Optional[np.ndarray]:
    """Fallback centroid estimate: back-project each view's masked pixels of its
    camera-space pointmap (pointmaps_sam3d) into world space using that view's c2w,
    pool across all views, and take the per-axis median (robust to a single noisy or
    partially-occluded view).

    Args:
        view_pointmaps: per-view (H, W, 3) camera-space point maps.
        view_masks: per-view (H, W) mask arrays (any truthy value = object pixel).
        camera_poses: per-view pose dicts as returned by
            run_inference_weighted.convert_da3_extrinsics_to_camera_poses (must contain 'c2w').
            Must be the same length and order as view_pointmaps/view_masks.
    """
    all_points = []
    for pointmap, mask, pose in zip(view_pointmaps, view_masks, camera_poses):
        pointmap = np.asarray(pointmap)
        mask = np.asarray(mask)
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            continue
        mask_bool = mask.astype(bool)
        if mask_bool.shape[:2] != pointmap.shape[:2]:
            # Sizes must line up; skip this view rather than guess a resize.
            continue
        pts_cam = pointmap[mask_bool]  # (M, 3)
        if pts_cam.size == 0:
            continue
        c2w = np.asarray(pose["c2w"])  # (4, 4)
        pts_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1))], axis=1)  # (M, 4)
        pts_world = (c2w @ pts_h.T).T[:, :3]  # (M, 3)
        all_points.append(pts_world)

    if not all_points:
        return None
    all_points = np.concatenate(all_points, axis=0)
    return np.median(all_points, axis=0)


def select_views_by_angular_coverage(
    camera_positions: Sequence[np.ndarray],
    centroid: np.ndarray,
    k: int,
) -> List[int]:
    """Greedily select k views whose viewing directions (relative to the object
    centroid) are maximally spread apart in angle, i.e. farthest-point sampling on
    the unit sphere of view directions.

    Args:
        camera_positions: world-space camera position per candidate view (index-aligned).
        centroid: world-space object centroid.
        k: number of views to select. Clamped to [1, len(camera_positions)].

    Returns:
        Selected view indices, sorted ascending (to preserve the caller's natural
        view ordering rather than the greedy pick order).
    """
    n = len(camera_positions)
    k = max(1, min(k, n))
    if k >= n:
        return list(range(n))

    positions = np.stack([np.asarray(p, dtype=np.float64) for p in camera_positions], axis=0)
    directions = positions - np.asarray(centroid, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    directions = directions / norms  # (n, 3) unit vectors

    # Pairwise angular distance matrix (radians).
    cos_sim = np.clip(directions @ directions.T, -1.0, 1.0)
    angular_dist = np.arccos(cos_sim)  # (n, n)

    selected = [0]
    remaining = set(range(n)) - {0}
    while len(selected) < k:
        # For each remaining view, its "coverage score" is the distance to its
        # nearest already-selected view; pick the one that maximizes that minimum.
        best_idx, best_score = None, -1.0
        for i in remaining:
            min_dist_to_selected = min(angular_dist[i, j] for j in selected)
            if min_dist_to_selected > best_score:
                best_score = min_dist_to_selected
                best_idx = i
        selected.append(best_idx)
        remaining.discard(best_idx)

    selected.sort()

    if len(selected) > 1:
        min_gap = min(
            angular_dist[selected[i], selected[j]]
            for i in range(len(selected))
            for j in range(i + 1, len(selected))
        )
        logger.info(
            f"[ViewSelection] Selected {len(selected)}/{n} views by angular coverage, "
            f"min pairwise angular gap = {np.degrees(min_gap):.1f} deg"
        )

    return selected
