"""
Import genrecon's VGGT-Omega Reconstruction into MV-SAM3D

Converts a completed genrecon run (camera poses, depth maps, images, and
per-object masks produced by public-forks-GenRecon's VGGT-Omega
reconstruction + segmentation stages) into the input format
run_inference_weighted.py expects (a da3_output.npz plus an images/masks
directory), so MV-SAM3D can run its 3D generation directly on that
geometry without running Depth Anything 3 at all.

Expected genrecon run directory layout:
    <genrecon_run>/
      vggt_export/
        images/frame_NNNNNN.jpg
        depth/frame_NNNNNN_depth.npy        (H, W, 1) float32
        sparse/0/cameras.txt                COLMAP text, PINHOLE per frame
        sparse/0/images.txt                 COLMAP text, w2c poses
      genrecon_output/segmentation_raw/masks/classes/
        labels.json                         {class_name: dir_name}
        <class>/mask_bin/frame_NNNNNN.png   binary uint8 mask, every frame

Usage:
    python scripts/import_from_genrecon.py \
        --genrecon_run /path/to/public-forks-GenRecon/runs/kitchen_multi_object \
        --output_dir ./data/kitchen_multi_object

    # Only a subset of classes:
    python scripts/import_from_genrecon.py \
        --genrecon_run /path/to/runs/kitchen_multi_object \
        --output_dir ./data/kitchen_multi_object \
        --objects bowl,cup
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def depth_to_pointmap(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth map to pointmap (3D coordinates in camera space).

    Standard camera space (same convention MV-SAM3D's scripts/run_da3.py uses):
        - x: right, y: down, z: forward (positive depth)
    SAM3D's compute_pointmap() applies the PyTorch3D coordinate transform
    internally, so we must NOT do it here.

    Args:
        depth: (H, W) depth map, values are distances from camera
        intrinsics: (3, 3) camera intrinsic matrix

    Returns:
        pointmap: (H, W, 3)
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    return np.stack([x, y, z], axis=-1)  # (H, W, 3)


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Scalar-first quaternion (w, x, y, z) -> (3, 3) rotation matrix."""
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def parse_cameras_txt(path: Path) -> Dict[int, Tuple[np.ndarray, int, int]]:
    """Parse COLMAP cameras.txt -> {camera_id: (K (3,3), width, height)}."""
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        camera_id = int(parts[0])
        model = parts[1]
        width, height = int(parts[2]), int(parts[3])
        if model != "PINHOLE":
            raise ValueError(f"Unsupported camera model '{model}' in {path} (expected PINHOLE)")
        fx, fy, cx, cy = (float(p) for p in parts[4:8])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        cameras[camera_id] = (K, width, height)
    return cameras


def parse_images_txt(path: Path) -> List[dict]:
    """
    Parse COLMAP images.txt -> list of entries sorted by IMAGE_ID, each:
        {image_id, extrinsics (3,4) w2c, camera_id, name}
    Each image occupies two lines; the second (2D points) line is skipped. That second line is
    frequently blank in this repo's exports (no reprojected 2D-3D correspondences), so blank
    lines are kept here (only comment lines are dropped) to preserve the 2-line pairing -- an
    earlier version filtered blank lines before chunking by 2, which silently discarded every
    other image whenever a points2D line was empty.
    """
    lines = [l for l in path.read_text().splitlines() if not l.strip().startswith("#")]

    entries = []
    for i in range(0, len(lines), 2):
        if not lines[i].strip():
            continue
        parts = lines[i].split()
        image_id = int(parts[0])
        qw, qx, qy, qz = (float(p) for p in parts[1:5])
        tx, ty, tz = (float(p) for p in parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]

        R = quat_wxyz_to_rotmat(np.array([qw, qx, qy, qz]))
        t = np.array([tx, ty, tz], dtype=np.float64)
        extrinsics = np.concatenate([R, t[:, None]], axis=1)  # (3, 4), world-to-camera

        entries.append(
            {
                "image_id": image_id,
                "extrinsics": extrinsics,
                "camera_id": camera_id,
                "name": name,
            }
        )

    entries.sort(key=lambda e: e["image_id"])
    return entries


def load_labels(segmentation_dir: Path) -> Dict[str, str]:
    labels_path = segmentation_dir / "masks" / "classes" / "labels.json"
    if not labels_path.exists():
        return {}
    return json.loads(labels_path.read_text())


def import_from_genrecon(genrecon_run: Path, output_dir: Path, objects: List[str] | None = None) -> None:
    """Converts a completed genrecon run into MV-SAM3D's input format (a da3_output.npz plus
    an images/masks directory). `objects` is a list of raw class names to export masks for
    (default: all classes in labels.json)."""
    genrecon_run = Path(genrecon_run)
    output_dir = Path(output_dir)
    vggt_export = genrecon_run / "vggt_export"
    segmentation_dir = genrecon_run / "genrecon_output" / "segmentation_raw"

    if not vggt_export.exists():
        raise FileNotFoundError(f"vggt_export/ not found under {genrecon_run}")

    print(f"[import_from_genrecon] Reading cameras from {vggt_export / 'sparse/0/cameras.txt'}")
    cameras = parse_cameras_txt(vggt_export / "sparse" / "0" / "cameras.txt")

    print(f"[import_from_genrecon] Reading poses from {vggt_export / 'sparse/0/images.txt'}")
    image_entries = parse_images_txt(vggt_export / "sparse" / "0" / "images.txt")
    print(f"[import_from_genrecon] Found {len(image_entries)} frames")

    depth_dir = vggt_export / "depth"
    images_src_dir = vggt_export / "images"

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dst_dir = output_dir / "images"
    images_dst_dir.mkdir(parents=True, exist_ok=True)

    all_depth = []
    all_pointmaps = []
    all_pointmaps_sam3d = []
    all_extrinsics = []
    all_intrinsics = []
    all_image_files = []

    W_last = H_last = None

    for entry in image_entries:
        stem = Path(entry["name"]).stem
        K, W, H = cameras[entry["camera_id"]]
        W_last, H_last = W, H

        depth_path = depth_dir / f"{stem}_depth.npy"
        depth = np.load(depth_path).astype(np.float64)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape != (H, W):
            raise ValueError(f"Depth shape {depth.shape} != camera size ({H}, {W}) for {stem}")

        pointmap = depth_to_pointmap(depth, K)
        pointmap_sam3d = pointmap.transpose(2, 0, 1)  # (3, H, W)

        all_depth.append(depth.astype(np.float32))
        all_pointmaps.append(pointmap.astype(np.float32))
        all_pointmaps_sam3d.append(pointmap_sam3d.astype(np.float32))
        all_extrinsics.append(entry["extrinsics"].astype(np.float32))
        all_intrinsics.append(K.astype(np.float32))

        src_image = images_src_dir / entry["name"]
        dst_image = images_dst_dir / entry["name"]
        shutil.copy2(src_image, dst_image)
        all_image_files.append(str(dst_image))

    da3_output = {
        "depth": np.stack(all_depth, axis=0),
        "pointmaps": np.stack(all_pointmaps, axis=0),
        "pointmaps_sam3d": np.stack(all_pointmaps_sam3d, axis=0),
        "extrinsics": np.stack(all_extrinsics, axis=0),
        "intrinsics": np.stack(all_intrinsics, axis=0),
        "image_files": np.array(all_image_files),
        "process_res": W_last,
    }
    da3_output_path = output_dir / "da3_output.npz"
    np.savez(da3_output_path, **da3_output)
    print(f"[import_from_genrecon] Wrote {da3_output_path} ({len(image_entries)} frames, {W_last}x{H_last})")

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    # labels.json maps raw label -> sanitized on-disk directory name (e.g.
    # "plastic cup" -> "plastic_cup"), since genrecon's segmentation stage
    # replaces spaces/slashes when creating per-class directories. object_names
    # stays in raw-label form throughout (both as the --objects default and as
    # the output folder name, matching what --mask_prompt is later given) --
    # only the on-disk mask_bin lookup below needs the sanitized name.
    labels = load_labels(segmentation_dir)
    if objects:
        object_names = [o.strip() for o in objects if o.strip()]
    else:
        object_names = list(labels.keys())

    if not object_names:
        print("[import_from_genrecon] No object classes found/requested; skipping mask export.")
    else:
        print(f"[import_from_genrecon] Exporting masks for: {object_names}")

    for class_name in object_names:
        dir_name = labels.get(class_name, class_name)
        mask_bin_dir = segmentation_dir / "masks" / "classes" / dir_name / "mask_bin"
        if not mask_bin_dir.exists():
            print(f"[import_from_genrecon]   WARNING: no mask_bin/ for class '{class_name}' (dir '{dir_name}') at {mask_bin_dir}, skipping")
            continue

        class_dst_dir = output_dir / class_name
        class_dst_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for entry in image_entries:
            stem = Path(entry["name"]).stem
            mask_path = mask_bin_dir / f"{stem}.png"
            if not mask_path.exists():
                continue

            mask = Image.open(mask_path).convert("L")
            if not mask.getbbox():
                # Object not visible in this frame (all-zero mask). Skip
                # writing a mask file entirely so load_images_and_masks_from_path
                # drops this frame for this class, instead of writing an
                # empty mask that later divides by zero when MV-SAM3D tries
                # to crop around a (nonexistent) foreground bounding box.
                continue

            image = Image.open(images_src_dir / entry["name"]).convert("RGB")
            if mask.size != image.size:
                mask = mask.resize(image.size, Image.NEAREST)

            rgba = Image.merge("RGBA", (*image.split(), mask))
            rgba.save(class_dst_dir / f"{stem}.png")
            n_written += 1

        print(f"[import_from_genrecon]   {class_name}: wrote {n_written}/{len(image_entries)} masks")

    # ------------------------------------------------------------------
    # Print ready-to-run command
    # ------------------------------------------------------------------
    mask_prompt = ",".join(object_names) if object_names else "<mask_prompt>"
    print("\n[import_from_genrecon] Done. Run MV-SAM3D with:\n")
    print(
        f"python run_inference_weighted.py \\\n"
        f"  --input_path {output_dir} \\\n"
        f"  --mask_prompt {mask_prompt} \\\n"
        f"  --da3_output {da3_output_path}"
    )


def main():
    parser = argparse.ArgumentParser(description="Import a genrecon run into MV-SAM3D's input format.")
    parser.add_argument("--genrecon_run", required=True, help="Path to genrecon run dir, e.g. runs/kitchen_multi_object")
    parser.add_argument("--output_dir", required=True, help="Output directory (becomes --input_path for MV-SAM3D)")
    parser.add_argument(
        "--objects",
        default=None,
        help="Comma-separated class names to export masks for (default: all classes in labels.json)",
    )
    args = parser.parse_args()

    objects = [o.strip() for o in args.objects.split(",") if o.strip()] if args.objects else None
    import_from_genrecon(Path(args.genrecon_run), Path(args.output_dir), objects)


if __name__ == "__main__":
    main()
