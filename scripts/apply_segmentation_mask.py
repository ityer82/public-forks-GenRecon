"""Turn COB-GS's per-class 2D masks + background point cloud into GenRecon inputs.

Given the per-frame class masks produced by COB-GS's
grounded_sam2_stable_tracking.py (one binary PNG per frame per class, listed
in labels.json) and the background.ply produced by its
segment_pointcloud.py, this writes:

  - an RGBA copy of each input frame with alpha=0 on any detected class
    pixel (so GenRecon's existing alpha-aware image loader masks it out),
  - a COLMAP-text points3D.txt containing only the background points.

Usage:
    uv run python scripts/apply_segmentation_mask.py \
        --images_dir runs/<scene>/vggt_export/images \
        --masks_root COB-GS/output/<scene>/masks/<label> \
        --background_ply COB-GS/output/<scene>/masks/<label>/background/point_cloud/background.ply \
        --out_rgb_dir runs/<scene>/segmentation/masked_rgb \
        --out_points3d runs/<scene>/segmentation/filtered_colmap/points3D.txt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image


def resolve_mask_path(mask_dir: Path, frame_stem: str) -> Path | None:
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = mask_dir / f"{frame_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def foreground_mask_for_frame(class_mask_dirs: list[Path], frame_stem: str, size: tuple[int, int]) -> np.ndarray:
    foreground = np.zeros((size[1], size[0]), dtype=bool)
    for mask_dir in class_mask_dirs:
        mask_path = resolve_mask_path(mask_dir, frame_stem)
        if mask_path is None:
            continue
        mask = np.array(Image.open(mask_path).convert("L").resize(size, Image.NEAREST))
        foreground |= mask > 0
    return foreground


def write_masked_images(images_dir: Path, class_mask_dirs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        foreground = foreground_mask_for_frame(class_mask_dirs, image_path.stem, image.size)
        alpha = np.where(foreground, 0, 255).astype(np.uint8)
        rgba = np.dstack([np.array(image), alpha])
        # Keep the source filename (incl. extension) so GenRecon's basename
        # lookup in images.txt still resolves, even though JPEG can't hold an
        # alpha channel — PIL identifies image format from file content, not
        # extension, so a PNG-encoded file with a .jpg name still loads fine.
        Image.fromarray(rgba, mode="RGBA").save(out_dir / image_path.name, format="PNG")


def write_background_points3d(background_ply: Path, out_points3d: Path) -> None:
    cloud = trimesh.load(background_ply, process=False)
    xyz = np.asarray(cloud.vertices)
    if cloud.visual is not None and getattr(cloud.visual, "vertex_colors", None) is not None:
        rgb = np.asarray(cloud.visual.vertex_colors)[:, :3]
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)

    out_points3d.parent.mkdir(parents=True, exist_ok=True)
    with out_points3d.open("w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR\n")
        for i, ((x, y, z), (r, g, b)) in enumerate(zip(xyz, rgb)):
            f.write(f"{i} {x} {y} {z} {int(r)} {int(g)} {int(b)} 0.0\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--masks_root", type=Path, required=True)
    parser.add_argument("--background_ply", type=Path, required=True)
    parser.add_argument("--out_rgb_dir", type=Path, required=True)
    parser.add_argument("--out_points3d", type=Path, required=True)
    args = parser.parse_args()

    labels = json.loads((args.masks_root / "labels.json").read_text())
    class_mask_dirs = [args.masks_root / dirname / "mask_bin" for dirname in labels.values()]

    write_masked_images(args.images_dir, class_mask_dirs, args.out_rgb_dir)
    write_background_points3d(args.background_ply, args.out_points3d)


if __name__ == "__main__":
    main()
