"""Segment the raw input point cloud directly using Stage-1 (Grounded-SAM2) 2D masks,
via multi-view reprojection + z-buffer visibility voting.

This never touches 3D Gaussian Splatting training - it only needs the COLMAP-format
camera poses/point cloud (see dataset_format.md) and the per-frame mask PNGs produced
by detect_and_segment.py.

Usage:
    uv run python segment_pointcloud.py --dataset dataset/tandt/truck \
        --output output/truck --text "The truck"
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw
from scipy.spatial import cKDTree, ConvexHull, QhullError

from scene.colmap_loader import (
    read_extrinsics_binary, read_extrinsics_text,
    read_intrinsics_binary, read_intrinsics_text,
    read_points3D_binary, read_points3D_text,
    qvec2rotmat,
    project_and_test_visibility,
)
from scene.ply_io import storePly


def read_colmap_model(sparse_dir):
    try:
        cam_extrinsics = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        cam_intrinsics = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except FileNotFoundError:
        cam_extrinsics = read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cam_intrinsics = read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))

    try:
        xyz, rgb, _ = read_points3D_binary(os.path.join(sparse_dir, "points3D.bin"))
    except FileNotFoundError:
        xyz, rgb, _ = read_points3D_text(os.path.join(sparse_dir, "points3D.txt"))

    return cam_extrinsics, cam_intrinsics, xyz, rgb


def resolve_mask_path(mask_dir, frame_name):
    """Find the mask file for `frame_name`, tolerating extension mismatches
    (e.g. COLMAP image names ending in .jpg vs. exported .png masks)."""
    exact = os.path.join(mask_dir, frame_name)
    if os.path.exists(exact):
        return exact
    stem = os.path.splitext(frame_name)[0]
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = os.path.join(mask_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def resolve_depth_path(depth_dir, frame_name):
    """Find the `<stem>_depth.npy` file for `frame_name` (a VGGT-Omega
    export names depth files after the image stem, e.g.
    frame_000001.png -> frame_000001_depth.npy)."""
    stem = os.path.splitext(frame_name)[0]
    candidate = os.path.join(depth_dir, f"{stem}_depth.npy")
    return candidate if os.path.exists(candidate) else None


def load_depth(path):
    """Load a VGGT-Omega depth export: metric, per-pixel, camera-space z,
    saved as (H, W, 1) float32. Returns (H, W) float32."""
    depth = np.load(path).astype(np.float32)
    return depth.squeeze(-1) if depth.ndim == 3 else depth


def resolve_depth_conf_path(depth_dir, frame_name):
    """Find the `<stem>_depth_conf.npy` file for `frame_name`, if the
    VGGT-Omega export it came from is new enough to include it (older
    exports only have `<stem>_depth.npy`)."""
    stem = os.path.splitext(frame_name)[0]
    candidate = os.path.join(depth_dir, f"{stem}_depth_conf.npy")
    return candidate if os.path.exists(candidate) else None


def load_depth_conf(path):
    """Load a VGGT-Omega per-pixel depth confidence export: (H, W) float32."""
    return np.load(path).astype(np.float32)


def depth_edge(depth, rtol=0.03, kernel_size=3):
    """Flag pixels at depth discontinuities (relative local max-min jump >
    `rtol`), where per-pixel depth is known to bleed across occlusion
    boundaries ("flying pixels"). Manual port of VGGT-Omega's own
    visual_util.py::depth_edge - duplicated here (not imported) because
    vggt-omega and COB-GS run in separate uv environments. Operates on a
    single-frame (H, W) array; returns a same-shape bool array."""
    depth = np.asarray(depth)
    original_shape = depth.shape
    depth = depth.reshape(-1, *original_shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(depth, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(depth, -np.inf)
    depth_min = np.full_like(depth, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + depth.shape[-2], x : x + depth.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(depth), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


_warned_mask_resize = False
_warned_missing_depth_conf = False


def load_mask_resized(mask_path, target_hw):
    """Read a mask PNG as bool and resize (nearest-neighbor) to `target_hw`
    (the depth map's resolution) if they don't already match, e.g. when
    masks were produced at a --resolution-downsampled scale in Stage 1."""
    global _warned_mask_resize
    mask = np.array(PILImage.open(mask_path).convert("L"))
    if mask.shape != tuple(target_hw):
        if not _warned_mask_resize:
            print(f"[warn] mask resolution {mask.shape} != depth resolution "
                  f"{target_hw}; resizing masks with nearest-neighbor "
                  f"(this warning is only printed once)")
            _warned_mask_resize = True
        mask = cv2.resize(mask, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def unproject_pixels(px, py, depth_vals, intr, extr):
    """Unproject pixel coords `px`, `py` (int arrays, (M,)) with per-pixel
    depth `depth_vals` ((M,), camera-space z) through (extr, intr) into
    world-space points (M, 3). Same formula as
    scene/frustum_utils.py::lift_box_to_frustum."""
    assert intr.model == "PINHOLE", f"Unsupported camera model: {intr.model}"
    fx, fy, cx, cy = intr.params[:4]
    R = qvec2rotmat(extr.qvec)
    t = np.asarray(extr.tvec)

    x_cam = (px.astype(np.float64) - cx) / fx * depth_vals
    y_cam = (py.astype(np.float64) - cy) / fy * depth_vals
    cam_pts = np.stack([x_cam, y_cam, depth_vals], axis=1)
    # X_cam = R @ X_world + t  =>  X_world = R.T @ (X_cam - t); as row
    # vectors this is (cam_pts - t) @ R.
    return (cam_pts - t) @ R


def vote_view(xyz, extr, intr, mask_dir, device, depth_tolerance):
    """Return (visible_mask, fg_vote) over all N points for a single camera view."""
    px, py, visible = project_and_test_visibility(xyz, extr, intr, device, depth_tolerance)
    N = xyz.shape[0]
    fg_vote = torch.zeros(N, dtype=torch.bool, device=device)
    if not visible.any():
        return visible, fg_vote

    mask_path = resolve_mask_path(mask_dir, extr.name)
    if mask_path is None:
        return torch.zeros(N, dtype=torch.bool, device=device), fg_vote

    mask = np.array(PILImage.open(mask_path).convert("L"))
    if not (mask > 0).any():
        # No detection in this frame at all - carries no signal either way,
        # so exclude it from voting instead of counting it as background.
        return torch.zeros(N, dtype=torch.bool, device=device), fg_vote

    mask_t = torch.from_numpy(mask > 0).to(device)
    vis_idx = torch.nonzero(visible, as_tuple=True)[0]
    fg_vote[vis_idx] = mask_t[py[vis_idx], px[vis_idx]]

    return visible, fg_vote


def segment_one_label(xyz, cam_extrinsics, cam_intrinsics, mask_dir, device,
                       depth_tolerance, label=None):
    """Vote foreground ratio + visibility over all cameras for a single mask directory.

    Returns (fg_ratio, labeled) over all N points: fg_ratio is votes/visible_counts
    for points visible (with a detection) in at least one view (0 elsewhere); labeled
    marks which points were visible in at least one view.
    """
    N = xyz.shape[0]
    fg_votes = torch.zeros(N, dtype=torch.float32, device=device)
    visible_counts = torch.zeros(N, dtype=torch.float32, device=device)

    prefix = f"[{label}] " if label else ""
    for i, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        visible, fg_vote = vote_view(xyz, extr, intr, mask_dir, device, depth_tolerance)
        visible_counts += visible.float()
        fg_votes += fg_vote.float()
        print(f"\r{prefix}[{i + 1}/{len(cam_extrinsics)}] {extr.name}: "
              f"{int(visible.sum())} visible points", end="")
    print()

    labeled = visible_counts > 0
    fg_ratio = torch.zeros(N, device=device)
    fg_ratio[labeled] = fg_votes[labeled] / visible_counts[labeled]

    return fg_ratio, labeled


def voxel_downsample(xyz, rgb, frame_id, voxel_size):
    """Simple grid-snap dedup: keep one point per occupied voxel cell of
    size `voxel_size`. Used to bound point count/duplication from lifting
    every foreground pixel of ~dozens of overlapping views. `frame_id` is
    carried through the same dedup indexing so per-point frame provenance
    survives downsampling."""
    keys = np.floor(xyz / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    return xyz[unique_idx], rgb[unique_idx], frame_id[unique_idx]


def lift_class_pointclouds(cam_extrinsics, cam_intrinsics, mask_dir, images_dir,
                            label_dirs, depth_dir, voxel_size,
                            depth_conf_thres=50.0, depth_edge_rtol=0.03,
                            return_frame_ids=False):
    """Directly lift each class's mask-foreground pixels to 3D world points
    via per-pixel depth (2D->3D lifting), instead of classifying the
    existing sparse COLMAP points by reprojecting them into 2D masks
    (3D->2D voting, see vote_view/segment_one_label).

    For each frame: load its depth map, load each class's mask (resized to
    the depth resolution if they don't already match), exclude pixels hit
    by more than one class simultaneously in that frame ("ambiguous" -
    same semantics as the old greedy hit_count>1 check), unproject each
    class's remaining foreground pixels using that frame's camera + depth,
    and sample RGB from the frame's image. Every frame independently
    contributes new 3D points for a class (there's no shared point
    identity to lock the way the old greedy sweep did), so points from all
    frames are simply concatenated per class, then optionally voxel-deduped.

    Raw per-pixel depth is known to be unreliable at depth discontinuities
    ("flying pixels" bleeding across occlusion boundaries) and in
    low-confidence regions - exactly the population VGGT-Omega's own
    points3D.ply export excludes via visual_util.py::filter_points (a
    depth_conf percentile threshold + depth_edge() discontinuity mask).
    Mask silhouettes are disproportionately depth edges, so lifting
    without the same filtering reproduces those artifacts as floaters.
    This mirrors that filtering in two passes: pass 1 gathers per-frame
    depth/masks/edges and the depth_conf population at candidate pixels;
    pass 2 lifts only pixels that clear both the edge and (if available)
    global confidence-percentile gate.

    Returns {class_name: (xyz (M,3) float32, rgb (M,3) uint8)}. When
    `return_frame_ids` is True, also returns a parallel
    {class_name: frame_id (M,) int32} dict (frame_id indexes into
    `frame_cache`/`cam_extrinsics` enumeration order, one id per point,
    surviving voxel dedup) and the ordered list of frame names, as a 3-tuple
    (class_points, class_frame_ids, frame_names) -- used by the center-based
    cross-frame consistency check downstream.
    """
    global _warned_missing_depth_conf
    class_names = list(label_dirs.keys())
    class_xyz = {c: [] for c in class_names}
    class_rgb = {c: [] for c in class_names}
    class_frame_id = {c: [] for c in class_names}

    frame_cache = []
    candidate_conf_values = []
    any_depth_conf = False

    for i, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]

        depth_path = resolve_depth_path(depth_dir, extr.name)
        if depth_path is None:
            print(f"\r[lift] [{i + 1}/{len(cam_extrinsics)}] {extr.name}: no depth, skipping", end="")
            continue
        depth = load_depth(depth_path)
        h, w = depth.shape

        depth_conf_path = resolve_depth_conf_path(depth_dir, extr.name)
        depth_conf = None
        if depth_conf_path is not None:
            depth_conf = load_depth_conf(depth_conf_path)
            any_depth_conf = True
        elif not _warned_missing_depth_conf:
            print(f"\n[warn] no *_depth_conf.npy found alongside depth (e.g. for {extr.name}); "
                  f"skipping confidence-based filtering, only depth-edge filtering will apply "
                  f"(this warning is only printed once)")
            _warned_missing_depth_conf = True

        edge_mask = depth_edge(depth, rtol=depth_edge_rtol)

        masks = {}
        for cls_name in class_names:
            mask_path = resolve_mask_path(os.path.join(mask_dir, label_dirs[cls_name], "mask_bin"), extr.name)
            if mask_path is None:
                continue
            mask = load_mask_resized(mask_path, (h, w))
            if not mask.any():
                # No detection in this frame at all - carries no signal, exclude.
                continue
            masks[cls_name] = mask

        ambiguous = None
        if len(masks) > 1:
            stacked = np.stack(list(masks.values()), axis=0)
            ambiguous = stacked.sum(axis=0) > 1

        frame_cache.append((extr, intr, depth, depth_conf, edge_mask, masks, ambiguous, h, w))

        if depth_conf is not None:
            for mask in masks.values():
                fg = mask & ~ambiguous if ambiguous is not None else mask
                fg = fg & ~edge_mask
                if fg.any():
                    candidate_conf_values.append(depth_conf[fg])

        print(f"\r[lift] pass 1/2 [{i + 1}/{len(cam_extrinsics)}] {extr.name}", end="")
    print()

    conf_threshold = None
    if any_depth_conf and candidate_conf_values:
        conf_threshold = np.percentile(np.concatenate(candidate_conf_values), depth_conf_thres)
        print(f"[lift] depth_conf percentile threshold ({depth_conf_thres}): {conf_threshold:.4f}")

    for i, (extr, intr, depth, depth_conf, edge_mask, masks, ambiguous, h, w) in enumerate(frame_cache):
        image = None
        for cls_name, mask in masks.items():
            fg = mask & ~ambiguous if ambiguous is not None else mask
            fg = fg & ~edge_mask
            if conf_threshold is not None:
                fg = fg & (depth_conf >= conf_threshold)
            py, px = np.nonzero(fg)
            if px.size == 0:
                continue
            depth_vals = depth[py, px].astype(np.float64)
            valid = depth_vals > 1e-6
            if not valid.any():
                continue
            px, py, depth_vals = px[valid], py[valid], depth_vals[valid]

            world_pts = unproject_pixels(px, py, depth_vals, intr, extr)

            if image is None:
                img_path = resolve_mask_path(images_dir, extr.name)
                image = np.array(PILImage.open(img_path).convert("RGB"))
                if image.shape[:2] != (h, w):
                    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            rgb = image[py, px]

            class_xyz[cls_name].append(world_pts.astype(np.float32))
            class_rgb[cls_name].append(rgb)
            class_frame_id[cls_name].append(np.full(world_pts.shape[0], i, dtype=np.int32))

        print(f"\r[lift] pass 2/2 [{i + 1}/{len(frame_cache)}] {extr.name}", end="")
    print()

    result = {}
    frame_id_result = {}
    for cls_name in class_names:
        if class_xyz[cls_name]:
            xyz_c = np.concatenate(class_xyz[cls_name], axis=0)
            rgb_c = np.concatenate(class_rgb[cls_name], axis=0)
            frame_id_c = np.concatenate(class_frame_id[cls_name], axis=0)
        else:
            xyz_c = np.zeros((0, 3), dtype=np.float32)
            rgb_c = np.zeros((0, 3), dtype=np.uint8)
            frame_id_c = np.zeros((0,), dtype=np.int32)
        if voxel_size > 0 and xyz_c.shape[0] > 0:
            xyz_c, rgb_c, frame_id_c = voxel_downsample(xyz_c, rgb_c, frame_id_c, voxel_size)
        result[cls_name] = (xyz_c, rgb_c)
        frame_id_result[cls_name] = frame_id_c
        print(f"[{cls_name}] {xyz_c.shape[0]} lifted points")

    if return_frame_ids:
        frame_names = [extr.name for extr, *_ in frame_cache]
        return result, frame_id_result, frame_names
    return result


def _forward_project_points(xyz_pts, extr, intr, device):
    """Forward-project (M,3) world points into (extr, intr)'s camera, with no
    in-bounds clipping or z-buffer test (unlike project_and_test_visibility)
    -- used to build a hull's full reprojected silhouette, which legitimately
    extends outside the image bounds/mask for a partially-out-of-frame or
    partially-occluded object. Returns (K,2) float pixel coords for the
    subset of points in front of the camera (K <= M)."""
    xyz_t = torch.from_numpy(xyz_pts).float().to(device)
    fx, fy, cx, cy = intr.params[:4]
    R = torch.from_numpy(qvec2rotmat(extr.qvec)).float().to(device)
    t = torch.from_numpy(np.array(extr.tvec)).float().to(device)
    cam_pts = xyz_t @ R.T + t
    z = cam_pts[:, 2]
    in_front = (z > 1e-6).cpu().numpy()
    z_safe = z.clamp_min(1e-6)
    u = (fx * cam_pts[:, 0] / z_safe + cx).cpu().numpy()
    v = (fy * cam_pts[:, 1] / z_safe + cy).cpu().numpy()
    return np.stack([u, v], axis=1)[in_front]


def _hull_vertices(xyz_pts, min_points):
    """3D convex hull extreme points of `xyz_pts`, or None if there are too
    few points or the point set is degenerate (coplanar/collinear --
    ConvexHull needs full-rank input)."""
    if xyz_pts.shape[0] < min_points:
        return None
    try:
        hull = ConvexHull(xyz_pts)
    except QhullError:
        return None
    return xyz_pts[hull.vertices]


def _reproject_silhouette_polygon(hull_pts, extr, intr, device):
    """2D convex-hull silhouette of a 3D hull's vertices as seen from
    (extr, intr), as a cv2.fillPoly-ready int32 polygon, or None if fewer
    than 3 vertices land in front of the camera."""
    uv = _forward_project_points(hull_pts, extr, intr, device)
    if uv.shape[0] < 3:
        return None
    polygon = cv2.convexHull(uv.astype(np.float32).reshape(-1, 1, 2))
    return polygon.astype(np.int32)


def _mask_overlap_fraction(polygon, mask):
    """Fraction of `mask`'s True pixels that fall inside `polygon`'s
    silhouette -- deliberately NOT IoU: a small mask fully contained in a
    much larger silhouette (e.g. a partially-out-of-frame or occluded view
    of the true object, where only part of it is detected) scores ~1.0,
    while a mask that doesn't overlap the silhouette at all (a different
    physical object) scores ~0."""
    mask_area = int(mask.sum())
    if mask_area == 0:
        return 0.0
    canvas = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(canvas, [polygon], 1)
    overlap = int((canvas.astype(bool) & mask).sum())
    return overlap / mask_area


def filter_by_hull_consensus(cam_extrinsics, cam_intrinsics, mask_dir, label_dirs,
                              class_points, class_frame_ids, frame_names, device,
                              overlap_threshold=0.5, min_hull_points=4):
    """RANSAC-style volumetric majority vote, replacing a single global mean
    (too easily dragged off-target by a contaminated minority of frames, and
    unable to represent a partially-visible view) with per-frame 3D convex
    hulls: every frame with enough lifted points is tried in turn as a seed
    hypothesis (its own hull), reprojected into every other frame with a
    detection and scored by _mask_overlap_fraction. The seed with the
    largest consensus (inlier count) wins -- if the true object is detected
    correctly in most frames, its seeds' hulls will agree with each other
    far more often than a wrong-instance minority's. The winning group's
    points are pooled into one final fused hull, and every frame is
    re-scored against that fused hull for the actual keep/reject decision
    (a frame not well matched by any single seed's smaller hull may still
    pass against the more complete fused one).

    Frames with no real detection for the class carry no signal and are
    skipped entirely (never scored, never rejected), matching
    vote_view/lift_class_pointclouds's existing semantics.

    Returns (filtered_class_points, rejected_frames) where filtered_class_points
    is {class_name: (xyz, rgb)} (same shape as lift_class_pointclouds's
    return) and rejected_frames is {class_name: [frame_name, ...]}.
    """
    name_to_key = {extr.name: key for key, extr in cam_extrinsics.items()}
    rejected_frames = {c: [] for c in label_dirs}
    filtered = {}

    for cls_name, (xyz_c, rgb_c) in class_points.items():
        frame_id_c = class_frame_ids[cls_name]
        if xyz_c.shape[0] == 0:
            filtered[cls_name] = (xyz_c, rgb_c)
            print(f"[hull-consensus:{cls_name}] no lifted points, skipping check")
            continue

        mask_subdir = os.path.join(mask_dir, label_dirs[cls_name], "mask_bin")

        # Preload each detected frame's mask + camera once (reused for every
        # seed's scoring pass, instead of re-reading per (seed, frame) pair).
        frame_info = {}  # frame_id -> (extr, intr, mask)
        for fid in sorted(set(frame_id_c.tolist())):
            key = name_to_key.get(frame_names[fid])
            if key is None:
                continue
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            mask_path = resolve_mask_path(mask_subdir, extr.name)
            if mask_path is None:
                continue
            mask = np.array(PILImage.open(mask_path).convert("L")) > 0
            if mask.shape != (intr.height, intr.width):
                mask = cv2.resize(mask.astype(np.uint8), (intr.width, intr.height),
                                   interpolation=cv2.INTER_NEAREST) > 0
            if not mask.any():
                continue  # no real detection -- carries no signal, leave alone
            frame_info[fid] = (extr, intr, mask)

        if not frame_info:
            filtered[cls_name] = (xyz_c, rgb_c)
            print(f"[hull-consensus:{cls_name}] no frames with a real detection, skipping check")
            continue

        frame_points = {fid: xyz_c[frame_id_c == fid] for fid in frame_info}

        best_seed, best_inliers = None, set()
        for seed_fid, seed_pts in frame_points.items():
            hull_pts = _hull_vertices(seed_pts, min_hull_points)
            if hull_pts is None:
                continue
            inliers = set()
            for fid, (extr, intr, mask) in frame_info.items():
                polygon = _reproject_silhouette_polygon(hull_pts, extr, intr, device)
                if polygon is not None and _mask_overlap_fraction(polygon, mask) >= overlap_threshold:
                    inliers.add(fid)
            if len(inliers) > len(best_inliers):
                best_seed, best_inliers = seed_fid, inliers

        if best_seed is None:
            print(f"[hull-consensus:{cls_name}] WARNING: no frame had enough points "
                  f"(>= {min_hull_points}) to seed a hull; skipping consensus check.")
            filtered[cls_name] = (xyz_c, rgb_c)
            continue

        # Fuse the winning group's points into one final hull, then
        # re-score every frame against it for the actual decision.
        fused_pts = np.concatenate([frame_points[fid] for fid in best_inliers], axis=0)
        fused_hull_pts = _hull_vertices(fused_pts, min_hull_points)
        if fused_hull_pts is None:
            fused_hull_pts = _hull_vertices(frame_points[best_seed], min_hull_points)

        final_keep = set()
        for fid, (extr, intr, mask) in frame_info.items():
            polygon = _reproject_silhouette_polygon(fused_hull_pts, extr, intr, device)
            score = _mask_overlap_fraction(polygon, mask) if polygon is not None else 0.0
            if score >= overlap_threshold:
                final_keep.add(fid)
            print(f"\r[hull-consensus:{cls_name}] {frame_names[fid]}: score={score:.2f} "
                  f"{'OK' if fid in final_keep else 'REJECT'}", end="")
        print()

        bad_fids = set(frame_info) - final_keep
        for fid in bad_fids:
            rejected_frames[cls_name].append(frame_names[fid])

        print(f"[hull-consensus:{cls_name}] seed={frame_names[best_seed]}, "
              f"{len(frame_info)} frame(s) with a detection, {len(bad_fids)} rejected, "
              f"{len(final_keep)} kept")

        if bad_fids:
            keep_point = ~np.isin(frame_id_c, np.array(sorted(bad_fids), dtype=np.int32))
            filtered[cls_name] = (xyz_c[keep_point], rgb_c[keep_point])
        else:
            filtered[cls_name] = (xyz_c, rgb_c)

    return filtered, rejected_frames


def quarantine_rejected_frames(mask_dir, label_dirs, rejected_frames):
    """Non-destructively blank a class's mask_bin/mask_overlay PNGs for each
    frame flagged by filter_by_hull_consensus: move the original aside
    into a sibling mask_bin_rejected/ (mask_overlay_rejected/) dir, then
    write an all-zero PNG of the same size/mode at the original path, so
    every downstream consumer that reads mask_bin/<frame>.png directly from
    disk (apply_segmentation_mask.py, export_rgba_masks.py,
    extract_object_mesh.py/reconstruct_scene.py's search_hull_padding, and
    this script's own export_debug_projections) sees "no detection" there
    with no code changes needed on their end. Idempotent across reruns."""
    for cls_name, frame_names in rejected_frames.items():
        if not frame_names:
            continue
        class_dir = os.path.join(mask_dir, label_dirs[cls_name])
        for subdir_name in ("mask_bin", "mask_overlay"):
            src_dir = os.path.join(class_dir, subdir_name)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(class_dir, f"{subdir_name}_rejected")
            for frame_name in frame_names:
                src_path = resolve_mask_path(src_dir, frame_name)
                if src_path is None:
                    continue
                dst_path = os.path.join(dst_dir, os.path.basename(src_path))
                if os.path.exists(dst_path):
                    continue  # already quarantined in a previous run
                os.makedirs(dst_dir, exist_ok=True)
                img = PILImage.open(src_path)
                img.load()
                blank = PILImage.new(img.mode, img.size, 0)
                os.rename(src_path, dst_path)
                blank.save(src_path)
        print(f"[quarantine:{cls_name}] blanked {len(frame_names)} frame(s) on disk "
              f"under mask_bin/ (+ mask_overlay/ if present)")


def write_rejection_manifest(mask_dir, rejected_frames):
    """Write <mask_dir>/hull_consistency_rejections.json ({class_name:
    [frame_name, ...]}) as a durable audit trail -- quarantine_rejected_frames
    otherwise silently rewrites mask PNGs in place. Always written (even
    empty) so its absence is never ambiguous with "check didn't run"."""
    manifest = {cls: frames for cls, frames in rejected_frames.items() if frames}
    path = os.path.join(mask_dir, "hull_consistency_rejections.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    if manifest:
        total = sum(len(v) for v in manifest.values())
        print(f"[hull-consensus] wrote rejection manifest ({total} frame(s) across "
              f"{len(manifest)} class(es)) -> {path}")


def validate_lifted_points(cam_extrinsics, cam_intrinsics, mask_dir, label_dirs,
                            class_points, device, depth_tolerance):
    """Cross-frame consistency filter, mirroring the previous 3D-point
    validate_assignments pass, now applied to lifted points instead of the
    sparse COLMAP cloud. Voxel-dedup in lift_class_pointclouds already gave
    each surviving point a fixed, single world-space identity; here that
    identity is reprojected into every camera frame (vote_view, same as
    before) and dropped if it's visible in a frame where its class has a
    real (non-empty) detection but the point falls outside that class's
    mask there.

    Returns a new {class_name: (xyz, rgb)} dict (does not mutate input).
    """
    validated = {}
    for cls_name, (xyz_c, rgb_c) in class_points.items():
        if xyz_c.shape[0] == 0:
            validated[cls_name] = (xyz_c, rgb_c)
            continue

        pts = torch.from_numpy(xyz_c).float().to(device)
        mask_subdir = os.path.join(mask_dir, label_dirs[cls_name], "mask_bin")
        violations = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)

        for i, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            visible, fg_vote = vote_view(pts, extr, intr, mask_subdir, device, depth_tolerance)
            violations |= visible & ~fg_vote
            print(f"\r[validate:{cls_name}] [{i + 1}/{len(cam_extrinsics)}] {extr.name}: "
                  f"{int(violations.sum())} violations so far", end="")
        print()

        keep = (~violations).cpu().numpy()
        validated[cls_name] = (xyz_c[keep], rgb_c[keep])
        print(f"[{cls_name}] {xyz_c.shape[0]} lifted, {int((~keep).sum())} reassigned to "
              f"background, {int(keep.sum())} kept")

    return validated


def assign_background_by_proximity(xyz_np, class_points, radius):
    """A sparse COLMAP point is background if it isn't within `radius` of
    any lifted class point. Computed via a scipy cKDTree nearest-neighbor
    query, which avoids materializing an (N, M) pairwise distance matrix
    (the dense torch.cdist approach OOMs once N, the sparse COLMAP cloud
    size, gets into the millions)."""
    all_class_pts = [xyz for xyz, _rgb in class_points.values() if xyz.shape[0] > 0]
    N = xyz_np.shape[0]
    if not all_class_pts:
        return np.ones(N, dtype=bool)

    class_pts = np.concatenate(all_class_pts, axis=0)
    tree = cKDTree(class_pts)
    min_dist, _ = tree.query(xyz_np, k=1, workers=-1)

    return min_dist > radius


def export_debug_projections(cam_extrinsics, cam_intrinsics, mask_dir, label_dirs,
                              class_points, device, depth_tolerance,
                              point_color=(255, 0, 0), point_radius=2):
    """For each frame and class, forward-project that class's lifted points
    back into the frame and draw them onto a copy of the class's mask_bin
    frame, saved under mask_proj, to visually verify lifted points land
    inside the mask."""
    class_tensors = {
        cls_name: torch.from_numpy(xyz).float().to(device)
        for cls_name, (xyz, _rgb) in class_points.items() if xyz.shape[0] > 0
    }

    for i, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        frame_stem = os.path.splitext(extr.name)[0]

        for cls_name, dirname in label_dirs.items():
            xyz_c = class_tensors.get(cls_name)
            if xyz_c is None:
                continue

            class_dir = os.path.join(mask_dir, dirname)
            mask_bin_path = os.path.join(class_dir, "mask_bin", f"{frame_stem}.png")
            if not os.path.exists(mask_bin_path):
                continue

            img = PILImage.open(mask_bin_path)
            if not (np.array(img.convert("L")) > 0).any():
                # No detection in this frame at all - drawing points on a
                # blank mask always looks like a miss, so skip it.
                continue

            px, py, visible = project_and_test_visibility(xyz_c, extr, intr, device, depth_tolerance)
            if not visible.any():
                continue

            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            xs = px[visible].cpu().numpy()
            ys = py[visible].cpu().numpy()
            for x, y in zip(xs, ys):
                draw.ellipse([int(x) - point_radius, int(y) - point_radius,
                              int(x) + point_radius, int(y) + point_radius],
                             fill=point_color)

            out_dir = os.path.join(class_dir, "mask_proj")
            os.makedirs(out_dir, exist_ok=True)
            img.save(os.path.join(out_dir, f"{frame_stem}.png"))

        print(f"\r[debug projections] [{i + 1}/{len(cam_extrinsics)}] {extr.name}", end="")
    print()


def main():
    parser = argparse.ArgumentParser(description="Segment the raw input point cloud using 2D masks")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset/<scene> (contains sparse/0/)")
    parser.add_argument("--output", type=str, required=True, help="Path to output/<scene>")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated candidate sub-classes for open-world, "
                              "class-based segmentation (must match --classes passed to "
                              "detect_and_segment.py). When set, a separate "
                              "object.ply is written per class, plus one shared background.ply.")
    parser.add_argument("--pc_mask_threshold", type=float, default=0.3,
                         help="Minimum foreground vote ratio to count a point as object. "
                              "Only used in single-class mode (no --classes), which classifies "
                              "the existing sparse COLMAP points via multi-view voting. "
                              "Multi-class mode instead lifts every mask-foreground pixel to a "
                              "new 3D point directly via per-pixel depth (see --depth_dir).")
    parser.add_argument("--depth_tolerance", type=float, default=0.1,
                         help="Relative tolerance for z-buffer front-most-point visibility test "
                              "(single-class voting and multi-class debug-projection visibility)")
    parser.add_argument("--depth_dir", type=str, default=None,
                         help="Directory of <frame_stem>_depth.npy per-pixel metric depth maps "
                              "(e.g. a VGGT-Omega --export-for-3dgs output's depth/ dir), used to "
                              "directly lift each class's 2D mask pixels to 3D world points. "
                              "Required when --classes is set.")
    parser.add_argument("--bg_assign_radius", type=float, default=0.02,
                         help="Multi-class mode: a sparse COLMAP point becomes background.ply "
                              "unless it lies within this world-space distance of a lifted class "
                              "point.")
    parser.add_argument("--voxel_size", type=float, default=0.005,
                         help="Multi-class mode: grid-snap dedup cell size applied to each "
                              "class's lifted points (0 disables) to bound point count/duplication "
                              "across overlapping views.")
    parser.add_argument("--depth_conf_thres", type=float, default=50.0,
                         help="Multi-class mode: percentile (0-100) of per-pixel depth_conf "
                              "(from <depth_dir>/<stem>_depth_conf.npy) below which a candidate "
                              "pixel is dropped before lifting. Mirrors VGGT-Omega's own "
                              "--conf-thres used to build points3D.ply. Ignored (with a warning) "
                              "if depth_conf files aren't present in --depth_dir.")
    parser.add_argument("--depth_edge_rtol", type=float, default=0.03,
                         help="Multi-class mode: relative local depth-jump threshold above which "
                              "a pixel is treated as a depth discontinuity and excluded from "
                              "lifting (mirrors VGGT-Omega's visual_util.py::depth_edge, used the "
                              "same way to build points3D.ply).")
    parser.add_argument("--skip_hull_consistency_check", action="store_true",
                         help="Multi-class mode: skip the per-class RANSAC volumetric-hull "
                              "consensus check (builds a 3D convex hull per frame, finds the "
                              "subgroup of frames whose hulls mutually agree via reprojection "
                              "overlap with each frame's mask, and quarantines frames outside "
                              "that majority group -- catches a wrong-instance track reseed "
                              "contaminating a class's point cloud).")
    parser.add_argument("--hull_overlap_threshold", type=float, default=0.5,
                         help="Hull consistency check: minimum fraction of a frame's real mask "
                              "that must be explained by (fall inside) a hull's reprojected "
                              "silhouette for that frame to count as consistent with the hull.")
    parser.add_argument("--hull_min_points", type=int, default=4,
                         help="Hull consistency check: minimum lifted points a frame needs to "
                              "seed a 3D convex hull hypothesis (a frame with fewer points can "
                              "still be scored against the winning fused hull, just can't serve "
                              "as a seed itself).")
    parser.add_argument("--mask_dir_override", type=str, default=None,
                         help="Read per-frame masks directly from this directory instead of "
                              "the derived <output>/masks/<text> (single-class mode only, i.e. "
                              "no --classes). Useful for testing against a specific class's "
                              "mask exports, e.g. <output>/masks/<text>/<class>/mask_bin.")
    parser.add_argument("--out_dir_override", type=str, default=None,
                         help="Single-class mode only: write object.ply/background.ply "
                              "directly into this directory instead of the derived "
                              "<mask_dir>/ply_pointcloud. Multi-class mode always writes "
                              "each class's point cloud under <mask_dir>/<class>/point_cloud/ "
                              "and background under <mask_dir>/background/point_cloud/, "
                              "ignoring this flag.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sparse_dir = os.path.join(args.dataset, "sparse", "0")
    mask_dir = args.mask_dir_override or os.path.join(args.output, "masks", args.text)

    cam_extrinsics, cam_intrinsics, xyz_np, rgb_np = read_colmap_model(sparse_dir)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    N = xyz.shape[0]

    if args.classes is None:
        out_dir = Path(args.out_dir_override) if args.out_dir_override else Path(mask_dir) / "ply_pointcloud"
        out_dir.mkdir(parents=True, exist_ok=True)

        fg_ratio, labeled = segment_one_label(
            xyz, cam_extrinsics, cam_intrinsics, mask_dir, device, args.depth_tolerance)
        is_object = labeled & (fg_ratio > args.pc_mask_threshold)
        is_background = labeled & ~is_object

        print(f"Points: {N} total, {int(is_object.sum())} object, "
              f"{int(is_background.sum())} background, {int((~labeled).sum())} unlabeled "
              f"(never visible in any view)")

        object_mask = is_object.cpu().numpy()
        background_mask = is_background.cpu().numpy()

        storePly(str(out_dir / "object.ply"), xyz_np[object_mask], rgb_np[object_mask])
        storePly(str(out_dir / "background.ply"), xyz_np[background_mask], rgb_np[background_mask])
        print(f"Wrote {out_dir / 'object.ply'} and {out_dir / 'background.ply'}")
        return

    if not args.depth_dir:
        parser.error("--depth_dir is required when --classes is set")

    with open(os.path.join(mask_dir, "labels.json")) as f:
        label_dirs = json.load(f)  # {class_name: sanitized_dirname}

    images_dir = os.path.join(args.dataset, "images")

    # Direct 2D->3D lifting: unproject each class's mask-foreground pixels
    # to new world-space points via per-pixel depth, instead of classifying
    # the existing sparse COLMAP points via 3D->2D voting.
    class_points, class_frame_ids, frame_names = lift_class_pointclouds(
        cam_extrinsics, cam_intrinsics, mask_dir, images_dir, label_dirs,
        args.depth_dir, args.voxel_size,
        depth_conf_thres=args.depth_conf_thres, depth_edge_rtol=args.depth_edge_rtol,
        return_frame_ids=True)

    # RANSAC volumetric-hull consensus check: build a 3D convex hull per
    # frame, find the majority subgroup of frames whose hulls mutually agree
    # (via reprojection overlap with each frame's mask), and drop (+
    # quarantine on disk) any frame outside that group -- catches a
    # wrong-instance track reseed (e.g. a text-only re-detection locking
    # onto a different physical object of the same class) without the cost/
    # fragility of validating every individual point (see validate_lifted_points
    # below, which is disabled for exactly that reason).
    if not args.skip_hull_consistency_check:
        class_points, rejected_frames = filter_by_hull_consensus(
            cam_extrinsics, cam_intrinsics, mask_dir, label_dirs, class_points,
            class_frame_ids, frame_names, device,
            overlap_threshold=args.hull_overlap_threshold,
            min_hull_points=args.hull_min_points)
        quarantine_rejected_frames(mask_dir, label_dirs, rejected_frames)
        write_rejection_manifest(mask_dir, rejected_frames)

    # Voxel-dedup above gave each surviving lifted point a fixed identity;
    # cross-check that identity against every frame's mask, same as the old
    # validate_assignments pass on the sparse COLMAP cloud.
    # Disabled: the all-frames-must-agree rule pruned 95% of points on the
    # toy scene (2558 -> 125), too aggressive for dense per-pixel lifted
    # points vs. the old sparse COLMAP cloud it was designed for.
    # class_points = validate_lifted_points(
    #     cam_extrinsics, cam_intrinsics, mask_dir, label_dirs, class_points,
    #     device, args.depth_tolerance)

    export_debug_projections(
        cam_extrinsics, cam_intrinsics, mask_dir, label_dirs, class_points,
        device, args.depth_tolerance)

    for cls_name, (xyz_c, rgb_c) in class_points.items():
        dirname = label_dirs[cls_name]
        cls_out_dir = Path(mask_dir) / dirname / "point_cloud"
        cls_out_dir.mkdir(parents=True, exist_ok=True)
        cls_ply_path = cls_out_dir / f"{dirname}.ply"
        storePly(str(cls_ply_path), xyz_c, rgb_c)
        print(f"[{cls_name}] {xyz_c.shape[0]} object points -> {cls_ply_path}")

    # Background: the sparse COLMAP points not spatially claimed by any
    # class's lifted point cloud.
    background_mask = assign_background_by_proximity(
        xyz_np, class_points, args.bg_assign_radius)
    background_out_dir = Path(mask_dir) / "background" / "point_cloud"
    background_out_dir.mkdir(parents=True, exist_ok=True)
    background_ply_path = background_out_dir / "background.ply"
    storePly(str(background_ply_path), xyz_np[background_mask], rgb_np[background_mask])
    print(f"Wrote shared {background_ply_path} ({int(background_mask.sum())} points)")


if __name__ == "__main__":
    main()
