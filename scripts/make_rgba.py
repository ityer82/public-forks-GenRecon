"""Composite a source image and a foreground mask into a single RGBA image.

Takes an RGB(A) source image and a separately-produced mask image (white/light =
foreground, black/dark = background) and writes an RGBA PNG with the mask as the
alpha channel, suitable for passing directly as --input to generate.py.

Usage:
    uv run --no-sync make_rgba.py --image images/T.png --mask images/T_mask.png --output images/T_rgba.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="Path to the source RGB(A) image.")
    parser.add_argument(
        "--mask",
        required=True,
        help="Path to the mask image (white/bright = keep foreground, black/dark = transparent "
        "background). Accepts a grayscale image, an RGB image (luminance is used), or an RGBA/LA "
        "image (its own alpha channel is used).",
    )
    parser.add_argument("--output", required=True, help="Path to write the output RGBA image to (use a .png extension).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional 0-255 cutoff to binarize the mask (values >= threshold become fully opaque, "
        "others fully transparent). By default the mask's grayscale values are used as-is, "
        "preserving soft/antialiased edges (TRELLIS.2 accepts non-binary alpha).",
    )
    parser.add_argument(
        "--resize-mask",
        choices=["error", "mask", "image"],
        default="error",
        help="How to handle a size mismatch between --image and --mask: 'error' fails fast (default), "
        "'mask' resizes the mask to match the image, 'image' resizes the image to match the mask.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert the mask (treat black as foreground / white as background), for masks produced "
        "with the opposite convention.",
    )
    return parser.parse_args()


def load_mask_array(mask: Image.Image) -> np.ndarray:
    """Derive a single-channel (H, W) uint8 array from a mask image of any mode."""
    if mask.mode in ("RGBA", "LA"):
        channel = "using alpha channel"
        array = np.array(mask.split()[-1])
    elif mask.mode in ("L", "1"):
        channel = "using grayscale values"
        array = np.array(mask.convert("L"))
    else:
        channel = "using luminance"
        array = np.array(mask.convert("L"))
    print(f"[load] mask ({mask.width}x{mask.height}, mode={mask.mode}) -> {channel}")
    return array


def composite_rgba(
    image: Image.Image,
    mask: Image.Image,
    threshold: float | None = None,
    resize_mask: str = "error",
    invert: bool = False,
) -> Image.Image:
    """Composite an already-loaded RGB image and mask image into an RGBA image.

    Mirrors main()'s CLI behavior (size mismatch handling, threshold binarization,
    invert, uniform-mask warning) so it can be reused as a library call.
    """
    mask_array = load_mask_array(mask)

    if invert:
        mask_array = 255 - mask_array

    if image.size != mask.size:
        if resize_mask == "error":
            raise ValueError(
                f"Image size {image.size} does not match mask size {mask.size}. "
                "Pass resize_mask='mask'|'image' to resize one to match the other."
            )
        elif resize_mask == "mask":
            mask_array = np.array(
                Image.fromarray(mask_array).resize(image.size, Image.Resampling.LANCZOS)
            )
            print(f"[resize] mask -> {image.size}")
        else:
            image = image.resize(mask.size, Image.Resampling.LANCZOS)
            print(f"[resize] image -> {mask.size}")

    if threshold is not None:
        mask_array = np.where(mask_array >= threshold, 255, 0).astype(np.uint8)

    if mask_array.min() == mask_array.max():
        print(
            f"[warn] mask is uniform (all values == {mask_array.min()}); this defeats compositing -- "
            "TRELLIS.2 treats an all-255 alpha channel as 'no mask' and will run its own background "
            "removal, while an all-0 alpha channel produces a fully transparent, unusable image.",
            file=sys.stderr,
        )

    rgba_array = np.dstack([np.array(image), mask_array])
    return Image.fromarray(rgba_array, mode="RGBA")


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"No such file: {args.image}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.mask):
        print(f"No such file: {args.mask}", file=sys.stderr)
        sys.exit(1)

    try:
        image = Image.open(args.image).convert("RGB")
        print(f"[load] image {args.image} ({image.width}x{image.height})")

        mask = Image.open(args.mask)

        result = composite_rgba(
            image, mask, threshold=args.threshold, resize_mask=args.resize_mask, invert=args.invert
        )

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        result.save(args.output)
        print(f"[write] {args.output} ({result.width}x{result.height}, RGBA)")
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
