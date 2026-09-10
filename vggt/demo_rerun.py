# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os

import numpy as np
import rerun as rr
import torch
from PIL import Image

from align_to_gravity import align_to_gravity, rotate_horizontal
from visual_util import filter_points
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera


def load_model(checkpoint_path: str) -> VGGTOmega:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run VGGT-Omega.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to("cuda")


def run_model(
    image_folder: str,
    model: VGGTOmega,
    image_resolution: int,
    skip_frames: int = -1,
    max_frames: int = -1,
) -> dict:
    print(f"Processing images from {image_folder}")

    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    if len(image_names) == 0:
        raise FileNotFoundError(f"No images found in {image_folder}")

    if max_frames != -1:
        image_names = image_names[:max_frames]
    if skip_frames != -1:
        image_names = image_names[::skip_frames]

    images = load_and_preprocess_images(image_names, image_resolution=image_resolution).to("cuda")
    print(f"Preprocessed images shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    predictions_np = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().numpy()
            if value.shape[0] == 1:
                value = value[0]
            predictions_np[key] = value

    predictions_np["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions_np["depth"],
        predictions_np["extrinsic"],
        predictions_np["intrinsic"],
    )
    predictions_np["source_image_names"] = [os.path.basename(p) for p in image_names]

    torch.cuda.empty_cache()
    return predictions_np


def apply_gravity_alignment(
    predictions: dict,
    align_to_gravity_flag: bool,
    rotate_horizontal_deg: float,
) -> dict:
    """Rotate/translate predictions' point map and extrinsics into a gravity-aligned frame.

    A non-zero `rotate_horizontal_deg` implies gravity alignment, since the
    rotation is defined on top of the fitted ground plane.
    """
    if not align_to_gravity_flag and rotate_horizontal_deg == 0.0:
        return predictions

    points_shape = predictions["world_points_from_depth"].shape
    result = align_to_gravity(
        points=predictions["world_points_from_depth"].reshape(-1, 3),
        extrinsics=predictions["extrinsic"],
    )
    if rotate_horizontal_deg != 0.0:
        result = rotate_horizontal(result, angle_degrees=rotate_horizontal_deg)

    predictions["world_points_from_depth"] = result.points.reshape(points_shape).astype(np.float32)
    predictions["extrinsic"] = result.extrinsics.astype(np.float32)
    return predictions


def unproject_depth_map_to_point_map(depth_map: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    depth = depth_map[..., 0]
    num_frames, height, width = depth.shape

    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    x = np.broadcast_to(x[None], (num_frames, height, width))
    y = np.broadcast_to(y[None], (num_frames, height, width))

    fx = intrinsic[:, 0, 0][:, None, None]
    fy = intrinsic[:, 1, 1][:, None, None]
    cx = intrinsic[:, 0, 2][:, None, None]
    cy = intrinsic[:, 1, 2][:, None, None]

    camera_points = np.stack(
        [
            (x - cx) / fx * depth,
            (y - cy) / fy * depth,
            depth,
        ],
        axis=-1,
    )

    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    return np.einsum(
        "sij,shwj->shwi",
        np.transpose(rotation, (0, 2, 1)),
        camera_points - translation[:, None, None, :],
    )


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a scalar-first quaternion [qw, qx, qy, qz]."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def export_colmap_dataset(
    predictions: dict,
    output_dir: str,
    conf_thres: float,
    max_points: int,
    save_depth: bool = False,
) -> None:
    """Write a COLMAP-text dataset (images/ + sparse/0/) for 3D Gaussian Splatting."""
    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))
    num_frames, height, width = images.shape[:3]

    # Always re-encode as JPEG regardless of the source format, since downstream
    # consumers (e.g. COB-GS/Grounded-SAM-2) hardcode ".jpg" in places.
    image_names = [f"{os.path.splitext(name)[0]}.jpg" for name in predictions["source_image_names"]]
    for name, image in zip(image_names, images):
        Image.fromarray((image * 255).clip(0, 255).astype(np.uint8)).save(os.path.join(images_dir, name), quality=95)

    with open(os.path.join(sparse_dir, "cameras.txt"), "w") as f:
        for i in range(num_frames):
            fx, fy = intrinsic[i, 0, 0], intrinsic[i, 1, 1]
            cx, cy = intrinsic[i, 0, 2], intrinsic[i, 1, 2]
            f.write(f"{i + 1} PINHOLE {width} {height} {fx} {fy} {cx} {cy}\n")

    with open(os.path.join(sparse_dir, "images.txt"), "w") as f:
        for i in range(num_frames):
            qw, qx, qy, qz = rotmat2qvec(extrinsic[i, :3, :3])
            tx, ty, tz = extrinsic[i, :3, 3]
            f.write(f"{i + 1} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {i + 1} {image_names[i]}\n")
            f.write("\n")

    vertices, colors = filter_points(
        predictions,
        conf_thres=conf_thres,
        max_points=max_points,
    )
    with open(os.path.join(sparse_dir, "points3D.txt"), "w") as f:
        for point_id, ((x, y, z), (r, g, b)) in enumerate(zip(vertices, colors), start=1):
            f.write(f"{point_id} {x} {y} {z} {r} {g} {b} 0.0\n")

    ply_path = os.path.join(sparse_dir, "points3D.ply")
    write_points3D_ply(ply_path, vertices, colors)

    depth_dir = None
    if save_depth:
        depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)
        depth = predictions["depth"]
        depth_conf = predictions["depth_conf"]
        for i, name in enumerate(image_names):
            stem = os.path.splitext(name)[0]
            np.save(os.path.join(depth_dir, f"{stem}_depth.npy"), depth[i])
            np.save(os.path.join(depth_dir, f"{stem}_depth_conf.npy"), depth_conf[i])

    print(
        f"Exported COLMAP dataset to {output_dir} "
        f"({num_frames} frames, {len(vertices)} points, PLY: {ply_path})"
        + (f", depth+depth_conf maps: {depth_dir}" if depth_dir is not None else "")
    )


def write_points3D_ply(path: str, vertices: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary little-endian PLY point cloud, importable by MeshLab."""
    colors_u8 = colors.clip(0, 255).astype(np.uint8)
    vertex_dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    vertex_data = np.empty(len(vertices), dtype=vertex_dtype)
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    vertex_data["red"], vertex_data["green"], vertex_data["blue"] = colors_u8[:, 0], colors_u8[:, 1], colors_u8[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())


def log_to_rerun(
    predictions: dict,
    conf_thres: float,
    mask_black_bg: bool,
    mask_white_bg: bool,
    show_cam: bool,
    mask_sky: bool,
    image_folder: str,
    max_points: int,
) -> None:
    vertices, colors = filter_points(
        predictions,
        conf_thres=conf_thres,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        image_dir=image_folder,
        max_points=max_points,
    )
    rr.log("world/points", rr.Points3D(positions=vertices, colors=colors))

    if not show_cam:
        return

    extrinsic = predictions["extrinsic"]
    intrinsic = predictions["intrinsic"]
    images = predictions["images"]
    if images.ndim == 4 and images.shape[1] == 3:
        images = np.transpose(images, (0, 2, 3, 1))

    for i in range(len(extrinsic)):
        world_to_camera = np.eye(4)
        world_to_camera[:3, :4] = extrinsic[i]
        camera_to_world = np.linalg.inv(world_to_camera)

        rr.log(
            f"world/camera_{i}",
            rr.Transform3D(translation=camera_to_world[:3, 3], mat3x3=camera_to_world[:3, :3]),
        )
        rr.log(
            f"world/camera_{i}/image",
            rr.Pinhole(
                image_from_camera=intrinsic[i],
                width=images.shape[2],
                height=images.shape[1],
            ),
        )
        rr.log(f"world/camera_{i}/image", rr.Image((images[i] * 255).clip(0, 255).astype(np.uint8)))


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-Omega rerun visualization script")
    parser.add_argument("image_folder", help="Path to a folder of images to reconstruct.")
    parser.add_argument("--checkpoint", default="checkpoints/vggt_omega/ckpts/vggt_omega_1b_512.pt", help="Local VGGT-Omega checkpoint path.")
    parser.add_argument("--image-resolution", type=int, default=512, help="Input image resolution. Default: 512.")
    parser.add_argument("--conf-thres", type=float, default=50.0, help="Confidence threshold percentile (0-100). Default: 50.")
    parser.add_argument("--max-points-k", type=int, default=1000, help="Max points to display, in thousands. Default: 1000.")
    parser.add_argument("--skip-frames", type=int, default=-1, help="Retain only every n-th frame. Default: -1 (no effect).")
    parser.add_argument("--max-frames", type=int, default=-1, help="Limit the number of loaded frames. Default: -1 (no effect).")
    parser.add_argument("--show-cam", action=argparse.BooleanOptionalAction, default=True, help="Show camera frustums.")
    parser.add_argument("--mask-sky", action=argparse.BooleanOptionalAction, default=False, help="Filter sky points.")
    parser.add_argument("--mask-black-bg", action=argparse.BooleanOptionalAction, default=False, help="Filter black background points.")
    parser.add_argument("--mask-white-bg", action=argparse.BooleanOptionalAction, default=False, help="Filter white background points.")
    parser.add_argument(
        "--export-for-3dgs",
        default=None,
        help="If set, write a COLMAP-text dataset (images/ + sparse/0/) to this directory for 3D Gaussian Splatting.",
    )
    parser.add_argument(
        "--save-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If set (with --export-for-3dgs), also save each frame's depth map and per-pixel "
        "depth confidence as .npy files under <export-for-3dgs>/depth/ "
        "(<stem>_depth.npy, <stem>_depth_conf.npy). Pass --no-save-depth to disable. Default: True.",
    )
    parser.add_argument(
        "--align-to-gravity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Align the point cloud and camera poses to gravity (fit ground plane, rotate to +z up).",
    )
    parser.add_argument(
        "--rotate-horizontal-deg",
        type=float,
        default=0.0,
        help="Additional rotation (degrees) about the vertical axis after gravity alignment. Implies --align-to-gravity if non-zero.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args.checkpoint)
    predictions = run_model(
        args.image_folder,
        model,
        args.image_resolution,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
    )
    predictions = apply_gravity_alignment(predictions, args.align_to_gravity, args.rotate_horizontal_deg)

    if args.export_for_3dgs is not None:
        export_colmap_dataset(
            predictions,
            args.export_for_3dgs,
            conf_thres=args.conf_thres,
            max_points=int(args.max_points_k * 1000),
            save_depth=args.save_depth,
        )
    elif args.save_depth:
        print("Warning: --save-depth has no effect without --export-for-3dgs.")

    rr.init("vggt-omega", spawn=True)
    log_to_rerun(
        predictions,
        conf_thres=args.conf_thres,
        mask_black_bg=args.mask_black_bg,
        mask_white_bg=args.mask_white_bg,
        show_cam=args.show_cam,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        max_points=int(args.max_points_k * 1000),
    )


if __name__ == "__main__":
    main()
