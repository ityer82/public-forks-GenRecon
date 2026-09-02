"""Export one best-view RGBA image per class from COB-GS segmentation output.

Given the per-frame class masks produced by COB-GS's
grounded_sam2_stable_tracking.py (one binary PNG per frame per class, listed
in labels.json), this picks -- for each class -- the frame that best balances
mask size and centering (largest foreground fraction, weighted down the
further its centroid sits from the image center), and composites that frame
with its mask into an RGBA PNG via
make_rgba.composite_rgba, suitable for passing to image-conditioned
generators (e.g. TRELLIS.2's generate.py --input). The RGBA image is written
to masks_root/<class>/mask_rgba/<frame_stem>.png, alongside COB-GS's own
per-class mask_bin/mask_overlay/mask_proj output directories.

Usage:
    uv run python scripts/export_rgba_masks.py \
        --images_dir runs/<scene>/vggt_export/images \
        --masks_root runs/<scene>/genrecon_output/segmentation_raw/masks/classes
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from apply_segmentation_mask import resolve_mask_path
from make_rgba import composite_rgba
from genrecon.utils.logger import logger


def best_frame_for_class(mask_dir: Path) -> tuple[str, float] | None:
    """Return (frame_stem, score) for the mask that best balances size and centering.

    Scores each frame's mask by coverage_ratio * centeredness, where centeredness
    is 1.0 when the mask's centroid sits exactly at the image center and falls off
    linearly to 0.0 at the image corners (normalized by the half-diagonal). This
    favors frames where the object is both large (most detail) and unoccluded/
    fully in view (a mask cut off at an edge pulls its centroid away from center).
    """
    best_stem = None
    best_score = -1.0
    for mask_path in sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg")) + sorted(mask_dir.glob("*.jpeg")):
        mask_array = np.array(Image.open(mask_path).convert("L"))
        fg = mask_array > 0
        fg_count = int(fg.sum())
        if fg_count == 0:
            continue
        ratio = float(fg_count) / mask_array.size

        h, w = mask_array.shape
        ys, xs = np.nonzero(fg)
        centroid_y, centroid_x = ys.mean(), xs.mean()
        center_y, center_x = h / 2.0, w / 2.0
        dist = float(np.hypot(centroid_y - center_y, centroid_x - center_x))
        half_diag = float(np.hypot(center_y, center_x))
        centeredness = 1.0 - min(dist / half_diag, 1.0) if half_diag > 0 else 1.0

        score = ratio * centeredness
        if score > best_score:
            best_score = score
            best_stem = mask_path.stem
    if best_stem is None:
        return None
    return best_stem, best_score


def export_rgba_masks(images_dir: Path, masks_root: Path) -> None:
    labels = json.loads((masks_root / "labels.json").read_text())

    for label, dirname in labels.items():
        class_dir = masks_root / dirname
        mask_dir = class_dir / "mask_bin"
        best = best_frame_for_class(mask_dir)
        if best is None:
            logger.warning(f"{label}: no mask frames found under {mask_dir}, skipping")
            continue
        frame_stem, score = best

        image_path = resolve_mask_path(images_dir, frame_stem)
        if image_path is None:
            logger.warning(f"{label}: source image for frame '{frame_stem}' not found under {images_dir}, skipping")
            continue
        mask_path = resolve_mask_path(mask_dir, frame_stem)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        rgba = composite_rgba(image, mask, resize_mask="mask")

        out_dir = class_dir / "mask_rgba"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{frame_stem}.png"
        rgba.save(out_path)
        logger.info(f"{label}: selected frame '{frame_stem}' (score {score:.3f}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--masks_root", type=Path, required=True)
    args = parser.parse_args()

    export_rgba_masks(args.images_dir, args.masks_root)


if __name__ == "__main__":
    main()
