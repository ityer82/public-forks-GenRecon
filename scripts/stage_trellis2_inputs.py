"""Stage per-class RGBA masks into a flat directory for TRELLIS.2 batch generation.

export_rgba_masks.py writes one RGBA PNG per class to
masks_root/<class>/mask_rgba/<frame_stem>.png. TRELLIS.2's generate.py takes
a directory of images and derives each output asset's name from the input
filename's stem, so this symlinks each class's RGBA file into a flat
directory named <label>.png -- giving generate.py's per-class output
directories the class label instead of an arbitrary frame timestamp.

Usage:
    uv run python scripts/stage_trellis2_inputs.py \
        --masks_root runs/<scene>/genrecon_output/segmentation_raw/masks/classes \
        --out_dir runs/<scene>/trellis2_input
"""
import argparse
import json
import sys
from pathlib import Path


def stage_trellis2_inputs(masks_root: Path, out_dir: Path) -> None:
    labels = json.loads((masks_root / "labels.json").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, dirname in labels.items():
        mask_rgba_dir = masks_root / dirname / "mask_rgba"
        candidates = sorted(mask_rgba_dir.glob("*.png"))
        if not candidates:
            print(f"[stage_trellis2_inputs] {label}: no RGBA mask found under {mask_rgba_dir}, skipping", file=sys.stderr)
            continue
        if len(candidates) > 1:
            candidates.sort(key=lambda p: p.stat().st_mtime)
            print(f"[stage_trellis2_inputs] {label}: {len(candidates)} RGBA masks found under {mask_rgba_dir}, using most recent", file=sys.stderr)
        source = candidates[-1].resolve()

        dest = out_dir / f"{label}.png"
        dest.unlink(missing_ok=True)
        dest.symlink_to(source)
        print(f"[stage_trellis2_inputs] {label}: {dest} -> {source}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masks_root", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    stage_trellis2_inputs(args.masks_root, args.out_dir)


if __name__ == "__main__":
    main()
