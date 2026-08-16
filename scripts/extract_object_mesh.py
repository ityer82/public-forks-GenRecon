"""Crop a reconstructed scene mesh down to a single object using its COB-GS point cloud.

Given the full scene mesh (shapes/mesh.ply, written by
genrecon/utils/mesh_utils.py::write_pbr_ply with per-vertex PBR properties)
and one object's point cloud (shapes/<label>.ply, plain xyz+rgb COLMAP points
from COB-GS's segment_pointcloud.py) -- both already in the same world frame
-- this builds a convex hull from the object's points, dilates it outward by
--hull_padding to compensate for the point cloud being a sparse sample of the
object's true surface, and keeps every mesh face whose three vertices all
fall inside the padded hull. Optionally also writes the exact complement
(remainder) mesh via --remainder_out_ply, so successive classes can be
cropped out of what's left of the mesh instead of the original each time.

Usage:
    uv run python scripts/extract_object_mesh.py \
        --mesh_ply runs/<scene>/genrecon_output/shapes/mesh.ply \
        --object_ply runs/<scene>/genrecon_output/shapes/printer.ply \
        --out_ply runs/<scene>/genrecon_output/shapes/printer_mesh.ply \
        --hull_padding 0.02
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, QhullError

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


def compact_mesh(vertex: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used_vertex_ids = np.unique(faces)
    remap = np.full(len(vertex), -1, dtype=np.int64)
    remap[used_vertex_ids] = np.arange(len(used_vertex_ids))

    new_vertex = vertex[used_vertex_ids]
    new_faces = remap[faces]
    return new_vertex, new_faces


def crop_mesh(
    mesh_ply: Path, equations: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ply = PlyData.read(mesh_ply)
    vertex = ply["vertex"]
    face = ply["face"]

    verts_xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)

    # The scene mesh can have millions of vertices while the object's hull is
    # tiny in comparison -- a full equations @ verts.T matmul against every
    # vertex is wastefully O(facets * total_verts). Cheaply cull to the
    # object's padded bounding box first, then only run the (still exact)
    # hull containment test against that much smaller candidate set.
    in_bbox = np.all((verts_xyz >= bbox_min) & (verts_xyz <= bbox_max), axis=1)
    candidate_ids = np.nonzero(in_bbox)[0]

    inside = np.zeros(len(vertex), dtype=bool)
    if len(candidate_ids) > 0:
        candidates = verts_xyz[candidate_ids]
        # A noisy/oversized object point cloud (e.g. a loose segmentation that
        # bleeds into the background) can produce both a hull with many
        # facets and a bbox cull that barely shrinks the candidate set --
        # facets * candidates can then be tens of millions, blowing up a
        # single dense matmul to tens of GB. Chunk over candidates to keep
        # peak memory bounded regardless of scene/hull size.
        chunk_size = max(1, int(2e8 // max(1, equations.shape[0])))
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            chunk_inside = np.all(equations[:, :3] @ chunk.T + equations[:, 3:4] <= 0.0, axis=0)
            inside[candidate_ids[start : start + chunk_size][chunk_inside]] = True

    faces = np.stack(face["vertex_indices"])
    # A face belongs to the object only if all 3 vertices are inside the
    # padded hull; every other face (fully outside, or straddling the hull
    # boundary) is remainder. This makes object/remainder an exact partition
    # of the input mesh's faces -- no gaps, no duplicated coverage.
    keep_face = inside[faces].all(axis=1)

    object_vertex, object_faces = compact_mesh(vertex.data, faces[keep_face])
    remainder_vertex, remainder_faces = compact_mesh(vertex.data, faces[~keep_face])
    return object_vertex, object_faces, remainder_vertex, remainder_faces


def write_cropped_ply(out_ply: Path, vertex_data: np.ndarray, faces: np.ndarray) -> None:
    face_data = np.empty(len(faces), dtype=[("vertex_indices", "i4", (3,))])
    face_data["vertex_indices"] = faces

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [
            PlyElement.describe(vertex_data, "vertex"),
            PlyElement.describe(face_data, "face"),
        ]
    ).write(out_ply)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_ply", type=Path, required=True)
    parser.add_argument("--object_ply", type=Path, required=True)
    parser.add_argument("--out_ply", type=Path, required=True)
    parser.add_argument(
        "--remainder_out_ply",
        type=Path,
        default=None,
        help="If set, also write the input mesh with the object's faces removed (the exact "
        "complement of --out_ply). Used to chain per-class crops so each class is cropped out "
        "of what's left of the mesh, rather than independently from the same input.",
    )
    parser.add_argument(
        "--hull_padding",
        type=float,
        default=0.02,
        help="Outward offset (world units, same scale as the COLMAP reconstruction) applied to "
        "each hull facet before cropping, to compensate for the object point cloud being a "
        "sparse sample of the object's true surface.",
    )
    args = parser.parse_args()

    def skip(reason: str) -> None:
        logger.warning(f"Skipping {args.object_ply}: {reason}")
        if args.remainder_out_ply is not None and args.remainder_out_ply != args.mesh_ply:
            args.remainder_out_ply.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.mesh_ply, args.remainder_out_ply)

    points = load_object_points(args.object_ply)
    if len(points) < 4:
        skip(f"only {len(points)} points, need >=4 to build a convex hull.")
        return

    try:
        equations = padded_hull_equations(points, args.hull_padding)
    except QhullError as e:
        skip(f"convex hull construction failed ({e}).")
        return

    bbox_min = points.min(axis=0) - args.hull_padding
    bbox_max = points.max(axis=0) + args.hull_padding
    object_vertex, object_faces, remainder_vertex, remainder_faces = crop_mesh(
        args.mesh_ply, equations, bbox_min, bbox_max
    )
    if len(object_faces) == 0:
        skip("no mesh faces fell inside the padded hull.")
        return

    write_cropped_ply(args.out_ply, object_vertex, object_faces)
    logger.info(f"Wrote {args.out_ply}: {len(object_vertex)} vertices, {len(object_faces)} faces.")

    if args.remainder_out_ply is not None:
        write_cropped_ply(args.remainder_out_ply, remainder_vertex, remainder_faces)
        logger.info(
            f"Wrote {args.remainder_out_ply}: {len(remainder_vertex)} vertices, "
            f"{len(remainder_faces)} faces."
        )


if __name__ == "__main__":
    main()
