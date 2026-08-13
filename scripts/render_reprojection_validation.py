"""Render the reconstructed mesh from the original COLMAP camera poses and
compare against the original photos, for visual reconstruction QA.

Given a world-frame vertex-colored ``mesh.ply`` (as written by
``inference/transform_to_original.py::save_mesh_to_original``) and the COLMAP
``cameras.txt``/``images.txt`` used to reconstruct it, this renders one
synthetic view per registered camera and writes:

  - the raw synthetic renders to ``--out_synth_dir``,
  - the original photo and its synthetic render concatenated side by side
    to ``--out_compare_dir``,

both keyed by the original image filename.

Usage:
    uv run python scripts/render_reprojection_validation.py \
        --mesh_ply runs/<scene>/output/mesh.ply \
        --colmap_dir runs/<scene>/scene/colmap \
        --images_dir runs/<scene>/scene/rgb \
        --out_synth_dir runs/<scene>/output/synth_views \
        --out_compare_dir runs/<scene>/output/compare_views
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from genrecon.renderers.mesh_renderer import MeshRenderer
from genrecon.representations.mesh import Mesh

_SEPARATOR_WIDTH_PX = 4


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


def load_colored_mesh(mesh_ply: Path, device: str) -> Mesh:
    scene = trimesh.load(mesh_ply, process=False)
    vertices = torch.tensor(np.asarray(scene.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(scene.faces), dtype=torch.int32, device=device)
    colors = np.asarray(scene.visual.vertex_colors)[:, :3].astype(np.float32) / 255.0
    vertex_attrs = torch.tensor(colors, dtype=torch.float32, device=device)
    return Mesh(vertices, faces, vertex_attrs)


def bbox_near_far(mesh: Mesh) -> tuple[float, float]:
    bbox_diag = (mesh.vertices.max(dim=0).values - mesh.vertices.min(dim=0).values).norm().item()
    bbox_diag = max(bbox_diag, 1e-3)
    return 0.01 * bbox_diag, 4.0 * bbox_diag


def render_view(renderer: MeshRenderer, mesh: Mesh, cam: dict) -> np.ndarray:
    device = mesh.device
    renderer.rendering_options.resolution = (cam["H"], cam["W"])
    out = renderer.render(
        mesh,
        cam["w2c"].to(device),
        cam["intrinsics"].to(device),
        return_types=["attr", "mask"],
    )
    attr = out["attr"].permute(1, 2, 0)  # (3, H, W) -> (H, W, 3)
    mask = out["mask"].unsqueeze(-1)  # (H, W) -> (H, W, 1)
    composited = attr * mask + 1.0 * (1.0 - mask)  # white background
    img = (composited.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu().numpy()
    return img


def make_side_by_side(original: np.ndarray, synthetic: np.ndarray) -> np.ndarray:
    h = original.shape[0]
    separator = np.full((h, _SEPARATOR_WIDTH_PX, 3), 255, dtype=np.uint8)
    return np.concatenate([original, separator, synthetic], axis=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh_ply", type=Path, required=True)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--images_dir", type=Path, required=True)
    parser.add_argument("--out_synth_dir", type=Path, required=True)
    parser.add_argument("--out_compare_dir", type=Path, required=True)
    parser.add_argument("--ssaa", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_synth_dir.mkdir(parents=True, exist_ok=True)
    args.out_compare_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = load_colored_mesh(args.mesh_ply, device)
    near, far = bbox_near_far(mesh)
    renderer = MeshRenderer(rendering_options={"near": near, "far": far, "ssaa": args.ssaa}, device=device)

    cameras = parse_colmap_cameras(args.colmap_dir)
    print(f"[render_reprojection_validation] {len(cameras)} camera(s) found in {args.colmap_dir}")

    n_rendered, n_missing = 0, 0
    for cam in cameras:
        image_path = args.images_dir / cam["name"]
        if not image_path.is_file():
            print(f"[render_reprojection_validation] skipping {cam['name']}: no matching original image")
            n_missing += 1
            continue

        synth = render_view(renderer, mesh, cam)
        Image.fromarray(synth).save(args.out_synth_dir / cam["name"])

        original = np.array(Image.open(image_path).convert("RGB"))
        Image.fromarray(make_side_by_side(original, synth)).save(args.out_compare_dir / cam["name"])
        n_rendered += 1

    print(
        f"[render_reprojection_validation] rendered {n_rendered} view(s) -> {args.out_synth_dir} "
        f"and {args.out_compare_dir} ({n_missing} skipped for missing originals)"
    )


if __name__ == "__main__":
    main()
