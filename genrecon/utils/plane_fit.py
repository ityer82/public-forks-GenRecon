"""Pure-numpy plane fitting (RANSAC + least-squares), no open3d dependency.

Adapted from ../vggt-omega/align_to_gravity.py's plane-fitting helpers (used there for
gravity alignment on point clouds) -- the math here is unchanged, just extracted so it can
be applied to mesh vertices too (e.g. scripts/extract_floor_mesh.py) without depending on
that sibling repo.
"""
from __future__ import annotations

import dataclasses

import numpy as np


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
