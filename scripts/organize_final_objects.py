"""Copy per-run reconstruction outputs into a single final_objects/ folder.

Pulls together the artifacts that end up scattered across
genrecon_output/segmentation_raw/, genrecon_output/shapes/, trellis2_input/
and trellis2_meshes/ into one self-contained directory: the background point
cloud and mesh at the root, and a subfolder per class holding that class's
detection preview, point cloud, cropped mesh, TRELLIS.2 input image, and
TRELLIS.2 mesh (if TRELLIS.2 reconstruction was run).

Usage:
    uv run python scripts/organize_final_objects.py \
        --shapes_dir runs/<scene>/genrecon_output/shapes \
        --segmentation_raw_dir runs/<scene>/genrecon_output/segmentation_raw \
        --trellis2_input_dir runs/<scene>/trellis2_input \
        --trellis2_meshes_dir runs/<scene>/trellis2_meshes \
        --out_dir runs/<scene>/final_objects
"""
import argparse
import json
import shutil
from pathlib import Path

from genrecon.utils.logger import logger


def _copy(source: Path, dest: Path, label: str, description: str) -> None:
    if not source.exists():
        logger.warning(f"{label}: {description} not found at {source}, skipping")
        return
    shutil.copy2(source, dest)
    logger.info(f"{label}: {description} {source} -> {dest}")


def organize_final_objects(
    shapes_dir: Path,
    segmentation_raw_dir: Path,
    trellis2_input_dir: Path,
    trellis2_meshes_dir: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy(shapes_dir / "background.ply", out_dir / "background.ply", "background", "point cloud")
    _copy(shapes_dir / "background_mesh.ply", out_dir / "background_mesh.ply", "background", "mesh")

    labels_json = segmentation_raw_dir / "masks" / "classes" / "labels.json"
    labels = json.loads(labels_json.read_text())

    for label, dirname in labels.items():
        class_dir = out_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)

        # shapes_dir/segmentation_raw_dir artifacts are keyed by the sanitized
        # `dirname` (e.g. "plastic_cup"); trellis2_input_dir/trellis2_meshes_dir
        # are keyed by the raw `label` (e.g. "plastic cup") -- see labels.json.
        _copy(segmentation_raw_dir / f"detection_{dirname}.png", class_dir / "detection.png", label, "detection")
        _copy(shapes_dir / f"{dirname}.ply", class_dir / "pointcloud.ply", label, "point cloud")
        _copy(shapes_dir / f"{dirname}_mesh.ply", class_dir / "mesh.ply", label, "mesh")
        _copy(trellis2_input_dir / f"{label}.png", class_dir / "trellis_input.png", label, "trellis input")

        trellis_mesh = trellis2_meshes_dir / label / "mesh.glb"
        if trellis_mesh.exists():
            _copy(trellis_mesh, class_dir / "trellis_mesh.glb", label, "trellis mesh")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes_dir", type=Path, required=True)
    parser.add_argument("--segmentation_raw_dir", type=Path, required=True)
    parser.add_argument("--trellis2_input_dir", type=Path, required=True)
    parser.add_argument("--trellis2_meshes_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    organize_final_objects(
        args.shapes_dir,
        args.segmentation_raw_dir,
        args.trellis2_input_dir,
        args.trellis2_meshes_dir,
        args.out_dir,
    )


if __name__ == "__main__":
    main()
