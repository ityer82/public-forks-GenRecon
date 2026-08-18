"""Shared COLMAP-text camera parsing and point projection helpers."""
from pathlib import Path

import numpy as np
import torch


def parse_colmap_cameras(colmap_dir: Path) -> list[dict]:
    """Parse a PINHOLE-only COLMAP text dataset into per-image camera params.

    Returns a list of dicts with keys "name", "W", "H", "w2c" (4,4 OpenCV
    world-to-camera torch.Tensor), "intrinsics" (3,3 normalized torch.Tensor).
    """
    cam_table: dict[int, dict] = {}
    with (colmap_dir / "cameras.txt").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id, model = int(parts[0]), parts[1]
            if model != "PINHOLE":
                raise ValueError(
                    f"Unsupported COLMAP camera model {model!r} for camera {cam_id}; "
                    "this script only handles PINHOLE (the model vggt-omega exports)."
                )
            W, H = int(float(parts[2])), int(float(parts[3]))
            fx, fy, cx, cy = (float(x) for x in parts[4:8])
            cam_table[cam_id] = {"W": W, "H": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

    cameras: list[dict] = []
    with (colmap_dir / "images.txt").open("r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME (NAME may contain spaces)
        parts = line.split(maxsplit=9)
        qw, qx, qy, qz = (float(x) for x in parts[1:5])
        tx, ty, tz = (float(x) for x in parts[5:8])
        cam_id = int(parts[8])
        name = Path(parts[9]).name

        cam = cam_table[cam_id]
        W, H = cam["W"], cam["H"]
        intrinsics = torch.tensor(
            [
                [cam["fx"] / W, 0.0, cam["cx"] / W],
                [0.0, cam["fy"] / H, cam["cy"] / H],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )

        # COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation (column-vector convention).
        R_w2c = torch.tensor(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=torch.float32,
        )
        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = torch.tensor([tx, ty, tz], dtype=torch.float32)

        cameras.append({"name": name, "W": W, "H": H, "w2c": w2c, "intrinsics": intrinsics})
        i += 2  # skip the POINTS2D line that follows each image header

    return cameras


def project_points(points_world: np.ndarray, cam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project world-space points into a camera's pixel space.

    `cam` is one entry from `parse_colmap_cameras` (w2c: (4,4), intrinsics:
    (3,3) normalized by W/H). Returns (pixels (N,2) float, in_front (N,) bool)
    -- pixels are only meaningful where in_front is True (no in-bounds check;
    callers that need on-screen points should also clip against cam["W"]/["H"]).
    """
    w2c = cam["w2c"].numpy() if isinstance(cam["w2c"], torch.Tensor) else np.asarray(cam["w2c"])
    intrinsics = (
        cam["intrinsics"].numpy() if isinstance(cam["intrinsics"], torch.Tensor) else np.asarray(cam["intrinsics"])
    )
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    cam_pts = points_world @ R.T + t
    z = cam_pts[:, 2]
    in_front = z > 1e-6

    z_safe = np.where(in_front, z, 1.0)
    x_n = cam_pts[:, 0] / z_safe
    y_n = cam_pts[:, 1] / z_safe

    fx_n, cx_n = intrinsics[0, 0], intrinsics[0, 2]
    fy_n, cy_n = intrinsics[1, 1], intrinsics[1, 2]
    u = (fx_n * x_n + cx_n) * cam["W"]
    v = (fy_n * y_n + cy_n) * cam["H"]

    pixels = np.stack([u, v], axis=1)
    return pixels, in_front
