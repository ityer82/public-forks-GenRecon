"""Export one best-view RGBA image per class from COB-GS segmentation output.

Given the per-frame class masks produced by COB-GS's
grounded_sam2_stable_tracking.py (one binary PNG per frame per class, listed
in labels.json), this picks -- for each class -- the frame where the object
occupies the largest fraction of the image (i.e. is seen with the most
detail), and composites that frame with its mask into an RGBA PNG via
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
    """Return (frame_stem, coverage_ratio) for the mask with the largest foreground fraction."""
    best_stem = None
    best_ratio = -1.0
    for mask_path in sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg")) + sorted(mask_dir.glob("*.jpeg")):
        mask_array = np.array(Image.open(mask_path).convert("L"))
        ratio = float((mask_array > 0).sum()) / mask_array.size
        if ratio > best_ratio:
            best_ratio = ratio
            best_stem = mask_path.stem
    if best_stem is None:
        return None
    return best_stem, best_ratio


def export_rgba_masks(images_dir: Path, masks_root: Path) -> None:
    labels = json.loads((masks_root / "labels.json").read_text())

    for label, dirname in labels.items():
        class_dir = masks_root / dirname
        mask_dir = class_dir / "mask_bin"
        best = best_frame_for_class(mask_dir)
        if best is None:
            logger.warning(f"{label}: no mask frames found under {mask_dir}, skipping")
            continue
        frame_stem, ratio = best

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
        logger.info(f"{label}: selected frame '{frame_stem}' (coverage {ratio:.1%}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--masks_root", type=Path, required=True)
    args = parser.parse_args()

    export_rgba_masks(args.images_dir, args.masks_root)


if __name__ == "__main__":
    main()
