"""Segment the flat floor region out of a cropped background mesh (background_mesh.ply)
into its own asset, so IsaacSim's convert_asset.py can give it an analytic-plane collider
(guaranteed non-fall-through) instead of the background's decimated meshSimplification
triangle-mesh proxy used for the rest of the background (see convert_asset.py's
--background-label help for why that proxy is decimated/approximate).

Runs after Stage 8's per-class cascading crop, on the final background_mesh.ply (every
--classes object already removed). Reuses extract_object_mesh.py's hull-crop machinery:
RANSAC-fits the dominant near-horizontal plane among the mesh's lowest vertices (gravity
alignment, on by default upstream, already puts the real floor near world Z=0, so the lowest
vertices are overwhelmingly floor), then treats the plane's inlier vertices as a synthetic
"object point set" fed into the same padded-hull face-membership crop used for per-class
objects.

Usage:
    uv run python scripts/extract_floor_mesh.py \
        --mesh_ply runs/<scene>/genrecon_output/shapes/background_mesh.ply \
        --out_ply runs/<scene>/genrecon_output/shapes/floor_mesh.ply \
        --remainder_out_ply runs/<scene>/genrecon_output/shapes/background_mesh.ply
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import QhullError

from extract_object_mesh import compact_mesh, crop_mesh, write_cropped_ply
from genrecon.utils.hull import padded_hull_equations
from genrecon.utils.logger import logger
from genrecon.utils.plane_fit import PlaneFitResult, point_plane_signed_distance, ransac_fit_plane


def _load_vertices(mesh_ply: Path) -> np.ndarray:
    ply = PlyData.read(mesh_ply)
    vertex = ply["vertex"]
    return np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)


def fit_floor_plane(
    verts_xyz: np.ndarray,
    *,
    z_percentile: float,
    distance_threshold: float,
    num_iterations: int,
    seed: int | None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """RANSAC-fits the dominant near-horizontal plane among the mesh's lowest vertices.

    Restricting RANSAC candidates to a low z-band (below `z_percentile`) keeps the fit from
    locking onto a wall or tabletop instead of the real floor. Returns (normal, d,
    inlier_points), with the normal flipped to point +Z (up) and `inlier_points` re-selected
    against the *full* vertex set (not just the low-z candidates), so the padded hull built
    from them still spans the floor's true footprint.
    """
    z = verts_xyz[:, 2]
    candidate_mask = z <= np.percentile(z, z_percentile)
    candidates = verts_xyz[candidate_mask]
    if len(candidates) < 3:
        raise RuntimeError(
            f"Only {len(candidates)} vertices below the {z_percentile}th z-percentile; "
            "too few to fit a floor plane."
        )

    fit: PlaneFitResult = ransac_fit_plane(
        candidates, distance_threshold=distance_threshold, num_iterations=num_iterations, seed=seed
    )
    normal, d = fit.normal, fit.d
    if normal[2] < 0:
        normal, d = -normal, -d

    up_alignment = abs(normal[2])
    if up_alignment < 0.8:
        raise RuntimeError(
            f"Fitted plane normal {normal} is not near-vertical (|z|={up_alignment:.3f} < 0.8) -- "
            "likely fit to a wall or other non-floor surface; skipping floor extraction."
        )

    full_distances = np.abs(point_plane_signed_distance(verts_xyz, normal, d))
    inlier_mask = full_distances < distance_threshold
    return normal, d, verts_xyz[inlier_mask]


def split_largest_component(
    vertex_data: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keeps only the single largest connected component of `faces` (in case another
    coplanar surface elsewhere in the background -- e.g. a shelf at the same height -- got
    swept into the same padded hull, since a 3D convex hull bridges spatially disjoint
    clusters at the same height). Returns (keep_vertex, keep_faces, discard_vertex,
    discard_faces): the discarded faces are returned (compacted) rather than dropped, so the
    caller can merge them back into the remainder mesh instead of silently losing geometry.
    """
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float64)
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1, nodes=np.arange(len(faces)))
    if len(components) <= 1:
        empty_faces = np.empty((0, 3), dtype=faces.dtype)
        discard_vertex, discard_faces = compact_mesh(vertex_data, empty_faces)
        return vertex_data, faces, discard_vertex, discard_faces

    largest = max(components, key=len)
    keep_mask = np.zeros(len(faces), dtype=bool)
    keep_mask[largest] = True

    keep_vertex, keep_faces = compact_mesh(vertex_data, faces[keep_mask])
    discard_vertex, discard_faces = compact_mesh(vertex_data, faces[~keep_mask])
    return keep_vertex, keep_faces, discard_vertex, discard_faces


def flatten_to_plane(vertex_data: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """Projects every vertex exactly onto the fitted plane, replacing the noisy raw positions
    (every vertex kept by fit_floor_plane's RANSAC inlier test just falls within
    +/-distance_threshold of the plane, so as-is the floor mesh is a bumpy band up to 2x that
    thick) with a perfectly flat surface at the analytic collision plane's own height.

    Without this, convert_asset.py's author_floor_collision() places the analytic collider at
    this mesh's bbox midpoint (its best unbiased estimate of the true height) but the *visible*
    mesh still has raw vertices scattered above and below that -- so parts of the rendered
    floor stick up above where objects are seated (they're placed to clear the analytic plane,
    not the noise), looking like objects are sunk into the floor even though they're resting
    exactly on its true collision surface."""
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float64)
    signed_distance = point_plane_signed_distance(xyz, normal, d)
    flattened = xyz - signed_distance[:, None] * normal
    out = vertex_data.copy()
    out["x"], out["y"], out["z"] = flattened[:, 0], flattened[:, 1], flattened[:, 2]
    return out


def merge_meshes(
    vertex_a: np.ndarray, faces_a: np.ndarray, vertex_b: np.ndarray, faces_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(faces_b) == 0:
        return vertex_a, faces_a
    merged_vertex = np.concatenate([vertex_a, vertex_b])
    merged_faces = np.concatenate([faces_a, faces_b + len(vertex_a)])
    return merged_vertex, merged_faces


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh_ply", type=Path, required=True)
    parser.add_argument("--out_ply", type=Path, required=True)
    parser.add_argument(
        "--remainder_out_ply", type=Path, default=None,
        help="If set, also write the input mesh with the floor's faces removed. Safe to pass the "
        "same path as --mesh_ply (e.g. to overwrite background_mesh.ply in place, chaining like "
        "Stage 8's per-class crops) -- the input is fully read before this is written.",
    )
    parser.add_argument(
        "--z_percentile", type=float, default=10.0,
        help="Percentile of mesh vertex Z used to pick RANSAC candidate vertices for the floor "
        "plane fit -- restricts the fit to the mesh's lowest vertices.",
    )
    parser.add_argument("--distance_threshold", type=float, default=0.02, help="RANSAC inlier distance threshold (world units).")
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument(
        "--hull_padding", type=float, default=0.02,
        help="Outward offset applied to the floor-inlier convex hull before cropping, same role "
        "as extract_object_mesh.py's --hull_padding.",
    )
    parser.add_argument("--remainder_margin", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def skip(reason: str) -> None:
        logger.warning(f"Skipping floor extraction for {args.mesh_ply}: {reason}")
        if args.remainder_out_ply is not None and args.remainder_out_ply != args.mesh_ply:
            args.remainder_out_ply.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.mesh_ply, args.remainder_out_ply)

    verts_xyz = _load_vertices(args.mesh_ply)

    try:
        normal, d, inlier_points = fit_floor_plane(
            verts_xyz,
            z_percentile=args.z_percentile,
            distance_threshold=args.distance_threshold,
            num_iterations=args.num_iterations,
            seed=args.seed,
        )
    except RuntimeError as e:
        skip(str(e))
        return

    remainder_padding = args.hull_padding + args.remainder_margin
    try:
        object_equations = padded_hull_equations(inlier_points, args.hull_padding)
        remainder_equations = padded_hull_equations(inlier_points, remainder_padding)
    except QhullError as e:
        skip(f"convex hull construction over the floor-plane inliers failed ({e}).")
        return

    bbox_margin = remainder_padding
    bbox_min = inlier_points.min(axis=0) - bbox_margin
    bbox_max = inlier_points.max(axis=0) + bbox_margin

    object_vertex, object_faces, remainder_vertex, remainder_faces = crop_mesh(
        args.mesh_ply, object_equations, remainder_equations, bbox_min, bbox_max,
    )
    if len(object_faces) == 0:
        skip("no mesh faces fell inside the padded floor-plane hull.")
        return

    floor_vertex, floor_faces, discard_vertex, discard_faces = split_largest_component(object_vertex, object_faces)
    remainder_vertex, remainder_faces = merge_meshes(remainder_vertex, remainder_faces, discard_vertex, discard_faces)
    floor_vertex = flatten_to_plane(floor_vertex, normal, d)

    write_cropped_ply(args.out_ply, floor_vertex, floor_faces)
    logger.info(
        f"Wrote {args.out_ply}: {len(floor_vertex)} vertices, {len(floor_faces)} faces "
        f"(plane normal {normal}, d={d:.4f})."
    )

    if args.remainder_out_ply is not None:
        write_cropped_ply(args.remainder_out_ply, remainder_vertex, remainder_faces)
        logger.info(
            f"Wrote {args.remainder_out_ply}: {len(remainder_vertex)} vertices, "
            f"{len(remainder_faces)} faces."
        )


if __name__ == "__main__":
    main()
