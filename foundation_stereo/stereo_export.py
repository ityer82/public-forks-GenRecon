"""FoundationStereo depth for a rectified stereo pair, packaged as a drop-in replacement for the
VGGT-Omega prediction step of Stage 0 (see vggt/demo_rerun.py::run_model).

`run_stereo` returns a `predictions` dict with the same keys/shapes as `run_model`, so
`apply_gravity_alignment` and `export_colmap_dataset` from vggt/demo_rerun.py consume it unchanged.
The world frame is the left camera's frame (X right, Y down, Z forward, metres); the right camera
sits `baseline` metres along +X, i.e. its world-to-camera extrinsic is [I | -baseline, 0, 0].

The model code under core/ is vendored from NVlabs/FoundationStereo (NVIDIA license, see the
headers). First model load also downloads `timm/edgenext_small` from Hugging Face and uses a DINOv2
checkpoint from ~/.cache/torch/hub; neither is vendored here.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

FS_DIR = Path(__file__).resolve().parent
if str(FS_DIR) not in sys.path:
    sys.path.insert(0, str(FS_DIR))

VALID_ITERS = 32
Z_FAR = 10.0  # metres, matches FoundationStereo's run_demo.py default
LR_CONSISTENCY_PX = 1.0  # max |d_left - d_right(warped)| for a pixel to get depth_conf 1
STEREO_CONF_SECTIONS = {"SVGA": (960, 600), "FHD": (1920, 1080), "FHD1200": (1920, 1200)}


def find_stereo_pair(image_folder: Path) -> tuple[Path, Path]:
    """The single image whose name contains 'left' and the single one containing 'right'."""
    images = sorted(p for p in Path(image_folder).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    left = [p for p in images if "left" in p.name.lower()]
    right = [p for p in images if "right" in p.name.lower()]
    if len(left) != 1 or len(right) != 1:
        raise ValueError(
            f"--stage-0-use-stereo needs exactly one 'left' and one 'right' image in {image_folder}, "
            f"found left={[p.name for p in left]}, right={[p.name for p in right]}."
        )
    return left[0], right[0]


def _parse_conf_sections(conf_path: Path) -> dict[str, dict[str, float]]:
    sections: dict[str, dict[str, float]] = {}
    current = None
    for line in Path(conf_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        header = re.fullmatch(r"\[(.+)\]", line)
        if header:
            current = sections.setdefault(header.group(1), {})
        elif "=" in line and current is not None:
            key, value = line.split("=", 1)
            try:
                current[key.strip()] = float(value)
            except ValueError:
                pass
    return sections


def parse_zed_conf(conf_path: Path, width: int, height: int) -> tuple[np.ndarray, float]:
    """(K (3,3), baseline in metres) for images of size `width` x `height`.

    Uses the left-camera focal length of the calibration section whose aspect ratio matches the image,
    scaled by width / section_width, with the principal point at the image centre (the images are
    assumed rectified, so distortion, TY/TZ and the small rotation terms are ignored). The [STEREO]
    Baseline is in millimetres.
    """
    sections = _parse_conf_sections(conf_path)
    aspect = width / height
    for name, (sec_w, sec_h) in STEREO_CONF_SECTIONS.items():
        left = sections.get(f"LEFT_CAM_{name}")
        if left is not None and abs(sec_w / sec_h - aspect) < 0.01:
            break
    else:
        raise ValueError(f"No LEFT_CAM_* section in {conf_path} matches an image aspect ratio of {aspect:.3f}.")
    focal = 0.5 * (left["fx"] + left["fy"]) * width / sec_w
    baseline_mm = sections.get("STEREO", {}).get("Baseline")
    if baseline_mm is None:
        raise ValueError(f"No [STEREO] Baseline in {conf_path}.")
    K = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K, baseline_mm / 1000.0


def load_model(checkpoint_path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run FoundationStereo.")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    from core.foundation_stereo import FoundationStereo

    cfg = OmegaConf.load(f"{os.path.dirname(checkpoint_path)}/cfg.yaml")
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    cfg["valid_iters"] = VALID_ITERS
    cfg["hiera"] = 0
    model = FoundationStereo(OmegaConf.create(cfg))
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    return model.cuda().eval()


@torch.no_grad()
def _disparity(model, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Left-view disparity (H, W) in pixels for uint8 RGB (H, W, 3) images."""
    from core.utils.utils import InputPadder

    height, width = left.shape[:2]
    img0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)
    with torch.autocast("cuda", enabled=True):
        disp = model.forward(img0, img1, iters=VALID_ITERS, test_mode=True)
    disp = padder.unpad(disp.float())
    return disp.cpu().numpy().reshape(height, width)


def _warp_horizontal(values: np.ndarray, x_src: np.ndarray) -> np.ndarray:
    """values[y, round(x_src[y, x])] with NaN where the source column is out of range."""
    height, width = values.shape
    xi = np.rint(x_src).astype(np.int64)
    inside = (xi >= 0) & (xi < width)
    rows = np.broadcast_to(np.arange(height)[:, None], values.shape)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    out[inside] = values[rows[inside], xi[inside]]
    return out


def run_stereo(left_path: Path, right_path: Path, model, conf_path: Path) -> dict:
    """Predictions dict compatible with vggt/demo_rerun.py::run_model (2 frames: left, right)."""
    from demo_rerun import unproject_depth_map_to_point_map

    left = np.asarray(Image.open(left_path).convert("RGB"))
    right = np.asarray(Image.open(right_path).convert("RGB"))
    if left.shape != right.shape:
        raise ValueError(f"Left/right image sizes differ: {left.shape} vs {right.shape}")
    height, width = left.shape[:2]
    K, baseline = parse_zed_conf(conf_path, width, height)

    disp_left = _disparity(model, left, right)
    # Right-view disparity: run the flipped, swapped pair (flip(right) acts as the "left" image).
    disp_right = _disparity(model, np.ascontiguousarray(right[:, ::-1]), np.ascontiguousarray(left[:, ::-1]))[:, ::-1]

    xx = np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], (height, width))
    # A left pixel x sees the same point at x - d in the right image; a right pixel x at x + d in the left.
    x_in_right = xx - disp_left
    x_in_left = xx + disp_right
    consistent_left = np.abs(disp_left - _warp_horizontal(disp_right, x_in_right)) < LR_CONSISTENCY_PX
    consistent_right = np.abs(disp_right - _warp_horizontal(disp_left, x_in_left)) < LR_CONSISTENCY_PX

    def to_depth(disp, in_view):
        valid = np.isfinite(disp) & (disp > 0) & in_view
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(valid, K[0, 0] * baseline / np.where(valid, disp, 1.0), 0.0)
        depth[depth > Z_FAR] = 0.0
        return depth.astype(np.float32)

    depth = np.stack([to_depth(disp_left, x_in_right >= 0), to_depth(disp_right, x_in_left <= width - 1)])
    consistent = np.stack([consistent_left, consistent_right])
    depth_conf = ((depth > 0) & consistent).astype(np.float32)

    extrinsic = np.zeros((2, 3, 4), dtype=np.float32)
    extrinsic[:, :3, :3] = np.eye(3)
    extrinsic[1, 0, 3] = -baseline
    predictions = {
        "images": (np.stack([left, right]).astype(np.float32) / 255.0).transpose(0, 3, 1, 2),
        "depth": depth[..., None],
        "depth_conf": depth_conf,
        "extrinsic": extrinsic,
        "intrinsic": np.stack([K, K]),
        "source_image_names": [Path(left_path).name, Path(right_path).name],
    }
    predictions["world_points_from_depth"] = unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    ).astype(np.float32)
    return predictions
