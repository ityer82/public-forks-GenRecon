"""Remove disconnected mesh "floaters" left behind near a cropped object.

extract_object_mesh.py crops each class's object out of the scene mesh using
a tightly-padded convex hull (--hull_padding, default 0.02) that hugs the
true object boundary. In practice this leaves small disconnected mesh
fragments behind in background_mesh.ply near where the object used to sit
(e.g. hallucinated double-walled skin, partial geometry the tight crop
seam didn't fully remove).

This script does a second, more generous pass over an already-cropped
background mesh: it builds a deliberately *enlarged* convex hull around the
same object's point cloud (padding scaled to the object's own bbox size,
not a fixed world-unit value) and removes any small disconnected mesh
component whose vertices are almost entirely contained in that enlarged
hull. The single largest connected component (the main background body) is
always kept untouched, regardless of hull containment, as a safety net
against wiping real background structure.

Run after extract_object_mesh.py's crop for the same class, chaining the
same way Stage 8's per-class crops already do (--out_ply may be the same
path as --mesh_ply).

Usage:
    uv run python scripts/remove_floater_mesh.py \
        --mesh_ply runs/<scene>/genrecon_output/shapes/background_mesh.ply \
        --object_ply runs/<scene>/genrecon_output/shapes/printer.ply \
        --out_ply runs/<scene>/genrecon_output/shapes/background_mesh.ply \
        --floaters_out_ply runs/<scene>/genrecon_output/shapes/printer_floaters.ply
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData
from scipy.spatial import QhullError

from extract_object_mesh import _inside_mask, compact_mesh, write_cropped_ply
from genrecon.utils.hull import load_object_points, padded_hull_equations
from genrecon.utils.logger import logger


def find_components(vertex_data: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Returns each connected component of `faces` as an array of face indices,
    largest first."""
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float64)
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False)
    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=1, nodes=np.arange(len(faces)))
    return sorted(components, key=len, reverse=True)


def find_floater_component_indices(
    vertex_data: np.ndarray,
    faces: np.ndarray,
    components: list[np.ndarray],
    equations: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    containment_frac: float,
    max_floater_faces: int,
) -> list[int]:
    """Among all but the largest component (index 0), returns the indices (into
    `components`) of the ones to discard: at or below `max_floater_faces` faces,
    with at least `containment_frac` of their vertices inside the padded hull
    (`equations`)."""
    if len(components) <= 1:
        return []

    verts_xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float64)
    inside = _inside_mask(equations, bbox_min, bbox_max, verts_xyz)

    floater_indices = []
    for i, comp_faces in enumerate(components[1:], start=1):
        if len(comp_faces) > max_floater_faces:
            continue
        comp_vertex_ids = np.unique(faces[comp_faces])
        inside_frac = inside[comp_vertex_ids].mean()
        if inside_frac >= containment_frac:
            floater_indices.append(i)
    return floater_indices


def merge_face_groups(faces: np.ndarray, face_groups: list[np.ndarray]) -> np.ndarray:
    if not face_groups:
        return np.empty((0, 3), dtype=faces.dtype)
    return faces[np.concatenate(face_groups)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh_ply", type=Path, required=True)
    parser.add_argument("--object_ply", type=Path, required=True)
    parser.add_argument("--out_ply", type=Path, required=True)
    parser.add_argument(
        "--floaters_out_ply", type=Path, default=None,
        help="If set, also write every removed component (merged into one mesh) here, for "
        "inspection.",
    )
    parser.add_argument(
        "--search_padding_factor", type=float, default=0.2,
        help="Outward hull padding, as a fraction of the object point cloud's own bbox "
        "diagonal (padding = factor * bbox_diagonal), so the enlarged search volume scales "
        "with object/scene size instead of using a fixed world-unit value.",
    )
    parser.add_argument(
        "--containment_frac", type=float, default=0.95,
        help="Minimum fraction of a component's vertices that must fall inside the enlarged "
        "hull for that component to be treated as a floater.",
    )
    parser.add_argument(
        "--max_floater_faces", type=int, default=5000,
        help="Safety cap: only components at or below this face count are eligible for "
        "removal, regardless of hull containment. Protects larger, likely-genuine background "
        "structure that happens to fall inside the enlarged hull.",
    )
    args = parser.parse_args()

    def skip(reason: str) -> None:
        logger.warning(f"Skipping floater removal for {args.object_ply}: {reason}")
        if args.out_ply != args.mesh_ply:
            args.out_ply.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.mesh_ply, args.out_ply)

    points = load_object_points(args.object_ply)
    if len(points) < 4:
        skip(f"only {len(points)} points, need >=4 to build a convex hull.")
        return

    bbox_diagonal = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    padding = args.search_padding_factor * bbox_diagonal
    try:
        equations = padded_hull_equations(points, padding)
    except QhullError as e:
        skip(f"convex hull construction failed ({e}).")
        return

    bbox_min = points.min(axis=0) - padding
    bbox_max = points.max(axis=0) + padding

    ply = PlyData.read(args.mesh_ply)
    vertex_data = ply["vertex"].data
    faces = np.stack(ply["face"]["vertex_indices"])
    logger.info(f"{args.mesh_ply}: {len(vertex_data)} vertices, {len(faces)} faces before floater removal.")

    components = find_components(vertex_data, faces)
    floater_indices = find_floater_component_indices(
        vertex_data, faces, components, equations, bbox_min, bbox_max,
        containment_frac=args.containment_frac, max_floater_faces=args.max_floater_faces,
    )

    if not floater_indices:
        logger.info(f"{args.object_ply}: no floaters found (examined {len(components)} components).")
        write_cropped_ply(args.out_ply, vertex_data, faces)
        return

    floater_set = set(floater_indices)
    floater_faces = merge_face_groups(faces, [components[i] for i in floater_indices])
    kept_faces = merge_face_groups(faces, [c for i, c in enumerate(components) if i not in floater_set])

    kept_vertex, kept_faces = compact_mesh(vertex_data, kept_faces)
    write_cropped_ply(args.out_ply, kept_vertex, kept_faces)
    logger.info(
        f"{args.object_ply}: removed {len(floater_indices)}/{len(components)} components "
        f"({len(floater_faces)} faces). Wrote {args.out_ply}: {len(kept_vertex)} vertices, "
        f"{len(kept_faces)} faces."
    )

    if args.floaters_out_ply is not None:
        floater_vertex, floater_faces = compact_mesh(vertex_data, floater_faces)
        write_cropped_ply(args.floaters_out_ply, floater_vertex, floater_faces)
        logger.info(f"Wrote {args.floaters_out_ply}: {len(floater_vertex)} vertices, {len(floater_faces)} faces.")


if __name__ == "__main__":
    main()
