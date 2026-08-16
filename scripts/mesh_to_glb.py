"""Convert cropped object meshes in a shapes/ dir to per-object GLBs.

Given a genrecon_output/shapes/ directory (written by run_full_pipeline.sh's Stage
3.6/3.7), converts every <label>_mesh.ply (the per-object crops written by
scripts/extract_object_mesh.py, e.g. chair_mesh.ply, background_mesh.ply) to
<out_dir>/<label>/mesh.glb. The whole-scene mesh.ply and the plain per-class point
clouds (<label>.ply, e.g. chair.ply) are not touched -- only the *_mesh.ply crops.

The <label>/mesh.glb subdirectory layout (rather than a flat directory of .glb files)
matches what IsaacSim's convert_asset.py expects for batch conversion (its
find_glb_assets() looks for <input_dir>/<asset_name>/mesh.glb).

Usage:
    uv run python scripts/mesh_to_glb.py \
        --shapes_dir runs/<scene>/genrecon_output/shapes \
        --out_dir runs/<scene>/genrecon_output/shapes/glb
"""
import argparse
from pathlib import Path

import trimesh

from genrecon.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes_dir", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Defaults to <shapes_dir>/glb.",
    )
    args = parser.parse_args()
    out_dir = args.out_dir if args.out_dir is not None else args.shapes_dir / "glb"

    def skip(mesh_ply: Path, reason: str) -> None:
        logger.warning(f"Skipping {mesh_ply}: {reason}")

    mesh_plys = sorted(args.shapes_dir.glob("*_mesh.ply"))
    if not mesh_plys:
        logger.warning(f"No *_mesh.ply files found in {args.shapes_dir}.")
        return

    for mesh_ply in mesh_plys:
        label = mesh_ply.stem[: -len("_mesh")]
        mesh = trimesh.load(mesh_ply, process=False)
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                skip(mesh_ply, "empty scene (no geometry).")
                continue
            mesh = trimesh.util.concatenate(mesh.dump())
        if len(mesh.faces) == 0:
            skip(mesh_ply, "zero faces.")
            continue

        out_glb = out_dir / label / "mesh.glb"
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_glb)
        logger.info(f"Wrote {out_glb}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces.")


if __name__ == "__main__":
    main()
