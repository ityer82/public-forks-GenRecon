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

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

from genrecon.utils.colmap_utils import parse_colmap_cameras, project_points
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


def crop_mesh(
    mesh_ply: Path,
    object_equations: np.ndarray,
    remainder_equations: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split `mesh_ply`'s faces into an object crop and a remainder, using two
    separate hulls (`object_equations` tight, `remainder_equations` >= it in
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
    """
    ply = PlyData.read(mesh_ply)
    vertex = ply["vertex"]
    face = ply["face"]
    logger.info(f"Cropping {mesh_ply}: {len(vertex.data)} vertices, {len(face.data)} faces before this crop.")

    verts_xyz = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1).astype(np.float64)
    inside_object = _inside_mask(object_equations, bbox_min, bbox_max, verts_xyz)
    inside_remainder = _inside_mask(remainder_equations, bbox_min, bbox_max, verts_xyz)

    faces = np.stack(face["vertex_indices"])
    keep_face_object = inside_object[faces].all(axis=1)
    keep_face_remainder = ~inside_remainder[faces].any(axis=1)

    object_vertex, object_faces = compact_mesh(vertex.data, faces[keep_face_object])
    remainder_vertex, remainder_faces = compact_mesh(vertex.data, faces[keep_face_remainder])
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
        args.mesh_ply, object_equations, remainder_equations, bbox_min, bbox_max
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
