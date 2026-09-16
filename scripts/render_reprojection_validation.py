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
from genrecon.utils.colmap_utils import parse_colmap_cameras
from genrecon.utils.logger import logger

_SEPARATOR_WIDTH_PX = 4


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
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5_000_000,
        help=(
            "Max faces rasterized per nvdiffrast call. Meshes with tens of millions of "
            "faces can crash the CUDA rasterizer (Cuda error 700) if rasterized in one "
            "shot; chunking avoids that. Set to 0 to disable chunking."
        ),
    )
    return parser


def run_reprojection_validation(
    mesh_ply: Path,
    colmap_dir: Path,
    images_dir: Path,
    out_synth_dir: Path,
    out_compare_dir: Path,
    *,
    ssaa: int = 2,
    chunk_size: int = 5_000_000,
) -> None:
    out_synth_dir.mkdir(parents=True, exist_ok=True)
    out_compare_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mesh = load_colored_mesh(mesh_ply, device)
    near, far = bbox_near_far(mesh)
    renderer = MeshRenderer(
        rendering_options={
            "near": near,
            "far": far,
            "ssaa": ssaa,
            "chunk_size": chunk_size or None,
        },
        device=device,
    )

    cameras = parse_colmap_cameras(colmap_dir)
    logger.info(f"{len(cameras)} camera(s) found in {colmap_dir}")

    n_rendered, n_missing = 0, 0
    try:
        for cam in cameras:
            image_path = images_dir / cam["name"]
            if not image_path.is_file():
                logger.warning(f"skipping {cam['name']}: no matching original image")
                n_missing += 1
                continue

            synth = render_view(renderer, mesh, cam)
            Image.fromarray(synth).save(out_synth_dir / cam["name"])

            original = np.array(Image.open(image_path).convert("RGB"))
            Image.fromarray(make_side_by_side(original, synth)).save(out_compare_dir / cam["name"])
            n_rendered += 1
    finally:
        del renderer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info(
        f"rendered {n_rendered} view(s) -> {out_synth_dir} "
        f"and {out_compare_dir} ({n_missing} skipped for missing originals)"
    )


def main() -> None:
    args = build_parser().parse_args()
    run_reprojection_validation(
        args.mesh_ply,
        args.colmap_dir,
        args.images_dir,
        args.out_synth_dir,
        args.out_compare_dir,
        ssaa=args.ssaa,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
