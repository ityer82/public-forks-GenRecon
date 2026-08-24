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

If --colmap_dir and --masks_dir are both given, a stage-2 search tightens
--hull_padding automatically instead of using it as a fixed value: it
binary-searches for the largest padding (within [--hull_padding_min,
--hull_padding]) whose padded hull, reprojected into each camera view that
has a real segmentation mask for this class, doesn't overshoot that mask by
more than --max_overshoot on average. This catches hulls that bleed into
background/environment geometry beyond what the true object silhouette
supports.

The object crop (--out_ply) and the remainder (--remainder_out_ply) use two
separate hulls, not one: --out_ply keeps faces fully inside the (possibly
tightened) object padding, while --remainder_out_ply drops any face with
even one vertex inside the looser --remainder_padding (default: the object
padding plus a small --remainder_margin, not the raw --hull_padding upper
bound -- when stage 2 tightens the object padding a lot, e.g. down near 0,
using the un-tightened --hull_padding for the remainder too would carve a
disconnected halo several times the object's own size out of whatever it's
sitting on). A single shared hull would force a choice between a tight
object crop that leaves a rim of the object's own boundary faces behind in
the background, or a hull loose enough to fully clear the background that
then bleeds environment geometry into the object crop.

If --mesh_ply had this object excluded from generation itself (e.g.
reconstruct_scene.py --exclude_masks_root) and so no longer contains it,
pass --object_mesh_ply pointing at a separate, unexcluded reconstruction of
the same scene (same world frame) to crop the object from instead --
--remainder_out_ply still always comes from --mesh_ply.

Usage:
    uv run python scripts/extract_object_mesh.py \
        --mesh_ply runs/<scene>/genrecon_output/shapes/mesh.ply \
        --object_ply runs/<scene>/genrecon_output/shapes/printer.ply \
        --out_ply runs/<scene>/genrecon_output/shapes/printer_mesh.ply \
        --hull_padding 0.02 \
        --colmap_dir runs/<scene>/genrecon_input/colmap \
        --masks_dir runs/<scene>/genrecon_output/segmentation_raw/masks/classes/printer/mask_bin
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pymeshfix
import trimesh
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree, QhullError

from genrecon.utils.colmap_utils import parse_colmap_cameras
from genrecon.utils.hull import load_object_points, padded_hull_equations, search_hull_padding
from genrecon.utils.logger import logger


def compact_mesh(vertex: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used_vertex_ids = np.unique(faces)
    remap = np.full(len(vertex), -1, dtype=np.int64)
    remap[used_vertex_ids] = np.arange(len(used_vertex_ids))

    new_vertex = vertex[used_vertex_ids]
    new_faces = remap[faces]
    return new_vertex, new_faces


def _inside_mask(equations: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, verts_xyz: np.ndarray) -> np.ndarray:
    # The scene mesh can have millions of vertices while the object's hull is
    # tiny in comparison -- a full equations @ verts.T matmul against every
    # vertex is wastefully O(facets * total_verts). Cheaply cull to the
    # padded bounding box first, then only run the (still exact) hull
    # containment test against that much smaller candidate set.
    in_bbox = np.all((verts_xyz >= bbox_min) & (verts_xyz <= bbox_max), axis=1)
    candidate_ids = np.nonzero(in_bbox)[0]

    inside = np.zeros(len(verts_xyz), dtype=bool)
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
    return inside


def _crop_side(mesh_ply: Path, equations: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, *, keep_inside: bool) -> tuple[np.ndarray, np.ndarray]:
    """Crop `mesh_ply` against one padded hull (`equations`).

    ``keep_inside=True`` keeps faces whose 3 vertices are *all* inside the
    hull (the "object" side); ``keep_inside=False`` keeps faces with *no*
    vertex inside the hull (the "remainder" side, everything outside it).
    """
    ply = PlyData.read(mesh_ply)
    vertex = ply["vertex"]
    face = ply["face"]
    logger.info(f"Cropping {mesh_ply}: {len(vertex.data)} vertices, {len(face.data)} faces before this crop.")

    verts_xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    inside = _inside_mask(equations, bbox_min, bbox_max, verts_xyz)

    faces = np.stack(face["vertex_indices"])
    keep_face = inside[faces].all(axis=1) if keep_inside else ~inside[faces].any(axis=1)
    return compact_mesh(vertex.data, faces[keep_face])


def crop_mesh(
    mesh_ply: Path,
    object_equations: np.ndarray,
    remainder_equations: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    *,
    object_mesh_ply: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split faces into an object crop and a remainder, using two separate
    hulls (`object_equations` tight, `remainder_equations` >= it in
    coverage). A single shared hull would force a choice between a tight
    object crop that leaves a rim of the object's own boundary faces behind
    in the remainder (straddling faces have >=1 vertex outside a tight hull,
    so they're excluded from "object" -- but with only one hull, "outside"
    also means "kept in remainder"), or a loose-enough hull to fully clear
    the remainder that then bleeds background into the object crop. Using
    `remainder_equations` for the remainder side removes anything with *any*
    vertex inside it, independent of what's kept as the object; faces caught
    in between (excluded from the tight object hull, but not fully outside
    the looser remainder hull) are dropped from both -- a thin gap rather
    than a leftover rim in either mesh.

    ``object_mesh_ply``, if given, is a *different* source mesh for the
    object crop than `mesh_ply` (e.g. a real, unexcluded reconstruction, when
    `mesh_ply` had this object excluded from generation and so no longer
    contains it) -- the remainder crop always comes from `mesh_ply`.
    """
    object_vertex, object_faces = _crop_side(
        object_mesh_ply if object_mesh_ply is not None else mesh_ply, object_equations, bbox_min, bbox_max, keep_inside=True
    )
    remainder_vertex, remainder_faces = _crop_side(mesh_ply, remainder_equations, bbox_min, bbox_max, keep_inside=False)
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


def repair_object_mesh(
    vertex_data: np.ndarray, faces: np.ndarray, *, min_component_faces: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Clean up a raw face-membership crop into a single, watertight shell.

    `_crop_side` keeps a face only if all 3 vertices fall inside the padded
    hull, which shatters the true crop seam into thousands of disconnected
    noise shards on top of the one real open boundary (e.g. an object's open
    bottom, where it was cut out of whatever it was resting on). Isaac Sim's
    convexDecomposition collision cooking is unreliable on that kind of
    open/fragmented mesh -- it can produce a lopsided decomposition (an
    asymmetric mass/inertia tensor despite a visually symmetric mesh), which
    is what makes objects swivel or fall through the floor in simulation.

    A single fan-fill pass over the crop's boundary loops (trimesh's
    `repair.fill_holes`) isn't enough here in practice: the source scene mesh
    itself carries perforations beyond the one true crop seam (thousands of
    small boundary loops, many not even simple closed curves), so it's handed
    to `pymeshfix` -- a hole-filling/self-intersection-removal repair
    specifically built for meshes this damaged -- after first dropping
    sub-`min_component_faces` disconnected noise shards (cheap pre-filter;
    `pymeshfix` also discards remaining small components on its own via
    `remove_smallest_components`). This closes holes without convex-hull-
    filling any concavity (e.g. a bowl's interior stays open), unlike
    `--collision-approximation convexHull`.
    """
    xyz = np.stack([vertex_data["x"], vertex_data["y"], vertex_data["z"]], axis=1).astype(np.float64)
    has_color = all(name in vertex_data.dtype.names for name in ("red", "green", "blue"))
    vertex_colors = None
    if has_color:
        alpha = (
            vertex_data["alpha"] if "alpha" in vertex_data.dtype.names
            else np.full(len(vertex_data), 255, dtype=np.uint8)
        )
        vertex_colors = np.stack([vertex_data["red"], vertex_data["green"], vertex_data["blue"], alpha], axis=1)

    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, vertex_colors=vertex_colors, process=False)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        big = max(components, key=lambda c: len(c.faces))
        if len(big.faces) >= min_component_faces:
            mesh = big

    fixer = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
    fixer.repair(remove_smallest_components=True)
    repaired_vertices, repaired_faces = fixer.points, fixer.faces

    repaired = trimesh.Trimesh(vertices=repaired_vertices, faces=repaired_faces, process=False)
    if not repaired.is_watertight:
        logger.warning(
            f"Repaired mesh still not watertight (euler_number={repaired.euler_number}); "
            "proceeding anyway -- downstream collision cooking (e.g. Isaac Sim "
            "convexDecomposition) may be unreliable for this object."
        )

    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4")]
    if has_color:
        dtype += [("red", "u1"), ("green", "u1"), ("blue", "u1"), ("alpha", "u1")]
    new_vertex = np.empty(len(repaired_vertices), dtype=dtype)
    new_vertex["x"] = repaired_vertices[:, 0]
    new_vertex["y"] = repaired_vertices[:, 1]
    new_vertex["z"] = repaired_vertices[:, 2]
    if has_color:
        # pymeshfix's repair can add/relocate vertices (hole caps, intersection
        # fixes), so there's no 1:1 mapping back to the original per-vertex
        # colors -- nearest-neighbor lookup against the pre-repair mesh keeps
        # colors visually close without needing that mapping.
        _, nearest = cKDTree(mesh.vertices).query(repaired_vertices)
        colors = mesh.visual.vertex_colors[nearest]
        new_vertex["red"] = colors[:, 0]
        new_vertex["green"] = colors[:, 1]
        new_vertex["blue"] = colors[:, 2]
        new_vertex["alpha"] = colors[:, 3]

    return new_vertex, np.asarray(repaired_faces, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_ply", type=Path, required=True)
    parser.add_argument("--object_ply", type=Path, required=True)
    parser.add_argument("--out_ply", type=Path, required=True)
    parser.add_argument(
        "--object_mesh_ply",
        type=Path,
        default=None,
        help="If set, crop the object (--out_ply) from this mesh instead of --mesh_ply -- "
        "e.g. a real, unexcluded reconstruction, when --mesh_ply had this object excluded "
        "from generation and so no longer contains it. The remainder (--remainder_out_ply) "
        "always comes from --mesh_ply, unaffected by this. Defaults to --mesh_ply (today's "
        "single-source behavior).",
    )
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
        "sparse sample of the object's true surface. When --colmap_dir/--masks_dir are also "
        "set, this is the upper bound of a padding search rather than a fixed value.",
    )
    parser.add_argument(
        "--colmap_dir", type=Path, default=None,
        help="COLMAP text dataset dir (cameras.txt/images.txt) for this scene. Combined with "
        "--masks_dir, enables a stage-2 search that tightens --hull_padding by reprojecting "
        "the padded hull into each camera view and comparing it against the real per-frame "
        "segmentation mask for this class.",
    )
    parser.add_argument(
        "--masks_dir", type=Path, default=None,
        help="Directory of this class's per-frame binary segmentation masks (COB-GS's "
        "mask_bin/), used by the stage-2 padding search. Requires --colmap_dir.",
    )
    parser.add_argument(
        "--hull_padding_min", type=float, default=-0.01,
        help="Lower bound of the stage-2 padding search (world units).",
    )
    parser.add_argument(
        "--max_overshoot", type=float, default=0.15,
        help="Stage-2 padding search: max allowed mean fraction of the reprojected hull "
        "silhouette that may fall outside the true mask, averaged over qualifying frames.",
    )
    parser.add_argument(
        "--search_iters", type=int, default=8,
        help="Stage-2 padding search: number of bisection iterations.",
    )
    parser.add_argument(
        "--remainder_padding", type=float, default=None,
        help="Padding used to decide what's removed from --remainder_out_ply (any mesh face "
        "with >=1 vertex inside this padded hull is dropped from the remainder), independent "
        "of the (possibly much tighter, stage-2-searched) padding used to build --out_ply. "
        "Must be >= the object padding, or a rim of the object's own boundary faces would be "
        "excluded from --out_ply yet still counted as 'outside' and left behind in the "
        "remainder. Defaults to the object padding (post-search, if stage 2 ran) plus "
        "--remainder_margin.",
    )
    parser.add_argument(
        "--remainder_margin", type=float, default=0.005,
        help="Extra padding added on top of the object padding to form the default "
        "--remainder_padding (ignored if --remainder_padding is set explicitly). Keeps the "
        "remainder cut just loose enough to fully clear the object's own boundary faces "
        "without carving a much larger halo out of whatever the object is resting on -- "
        "using the un-tightened --hull_padding for this (the old default) could remove a "
        "region several times the object's size once stage 2 tightens the object padding "
        "down near 0.",
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

    hull_padding = args.hull_padding
    if args.colmap_dir is not None and args.masks_dir is not None:
        cameras = parse_colmap_cameras(args.colmap_dir)
        hull_padding = search_hull_padding(
            points, cameras, args.masks_dir,
            args.hull_padding_min, args.hull_padding, args.max_overshoot, args.search_iters,
        )
        logger.info(f"{args.object_ply}: stage-2 search chose hull_padding={hull_padding:.4f} "
                    f"(searched [{args.hull_padding_min}, {args.hull_padding}])")

    try:
        object_equations = padded_hull_equations(points, hull_padding)
    except QhullError as e:
        skip(f"convex hull construction failed ({e}).")
        return

    remainder_padding = (
        args.remainder_padding if args.remainder_padding is not None else hull_padding + args.remainder_margin
    )
    remainder_padding = max(remainder_padding, hull_padding)
    try:
        remainder_equations = padded_hull_equations(points, remainder_padding)
    except QhullError as e:
        logger.warning(
            f"{args.object_ply}: remainder-padding hull construction failed ({e}); "
            "falling back to the object hull for the remainder cut too."
        )
        remainder_equations = object_equations
        remainder_padding = hull_padding

    # The bbox is a conservative pre-filter for the exact half-space test below,
    # so it must never shrink past the original points' bbox, and must cover
    # the larger (remainder) hull to stay a safe superset for both tests.
    bbox_margin = max(remainder_padding, 0.0)
    bbox_min = points.min(axis=0) - bbox_margin
    bbox_max = points.max(axis=0) + bbox_margin
    object_vertex, object_faces, remainder_vertex, remainder_faces = crop_mesh(
        args.mesh_ply, object_equations, remainder_equations, bbox_min, bbox_max,
        object_mesh_ply=args.object_mesh_ply,
    )
    if len(object_faces) == 0:
        skip("no mesh faces fell inside the padded hull.")
        return

    try:
        object_vertex, object_faces = repair_object_mesh(object_vertex, object_faces)
    except Exception as e:
        logger.warning(f"{args.object_ply}: mesh repair failed ({e}); writing the unrepaired crop instead.")

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
