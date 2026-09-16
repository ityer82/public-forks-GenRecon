"""
Run chunked GLB conversion on saved reconstruct_scene outputs.

Loads to_glb_inputs.pt + chunk_inputs.pt produced by reconstruct_scene.py,
runs global remesh + per-chunk bake, writes a multi-primitive scene.glb.

Usage:
    python chunked_to_glb.py \
        --inputs path/to/to_glb_inputs.pt \
        --chunk_inputs path/to/chunk_inputs.pt \
        --output_dir Xabl/chunked_v1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from genrecon.utils.logger import logger
from inference.chunked_glb import chunked_to_glb


def load_pt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def run_chunked_to_glb(
    inputs: Path,
    chunk_inputs: Path,
    output_dir: Path,
    *,
    texture_size: int = 4096,
    simplify_threshold: int = 2_000_000,
    skip_fill_holes: bool = False,
    hole_perim: float = 1.0,
    skip_remesh: bool = False,
    remesh_res: int | None = None,
    remesh_band: float = 1.0,
    remesh_project: float = 0.9,
    smooth_iters: int = 0,
    smooth_lambda: float = 0.5,
    smooth_mu: float = -0.53,
    smooth_feature_angle: float = 25.0,
    dump_geometry_plys: bool = False,
    chunks_dir: Path | None = None,
    no_chunk_cache: bool = False,
):
    """Loads to_glb_inputs.pt/chunk_inputs.pt and runs chunked GLB conversion, returning the
    resulting trimesh.Scene (or None in --dump_geometry_plys mode). Unlike the original
    CLI (which swallowed any exception here and returned exit 0 on failure -- a real bug,
    since set -euo pipefail could never catch it), this lets exceptions propagate so a caller
    can correctly abort on failure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if no_chunk_cache:
        chunks_save_dir = None
    elif chunks_dir is not None:
        chunks_save_dir = chunks_dir
    else:
        chunks_save_dir = output_dir / "chunks"

    to_glb_inputs = load_pt(inputs)
    chunk_inputs_data = load_pt(chunk_inputs)

    logger.info(
        f"Loaded {inputs}: "
        f"vertices {tuple(to_glb_inputs['vertices'].shape)}, "
        f"faces {tuple(to_glb_inputs['faces'].shape)}, "
        f"attr_volume {tuple(to_glb_inputs['attr_volume'].shape)}, "
        f"coords {tuple(to_glb_inputs['coords'].shape)}"
    )
    logger.info(
        f"Loaded {chunk_inputs}: "
        f"{len(chunk_inputs_data['chunk_indices'])} chunks, "
        f"chunk_size_world={chunk_inputs_data['chunk_size_world']:.4f}"
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    try:
        scene = chunked_to_glb(
            vertices_world=to_glb_inputs["vertices"],
            faces=to_glb_inputs["faces"],
            attr_volume=to_glb_inputs["attr_volume"],
            coords=to_glb_inputs["coords"],
            attr_layout=to_glb_inputs["attr_layout"],
            aabb_world=to_glb_inputs["aabb"],
            voxel_size_world=to_glb_inputs["voxel_size"],
            chunk_centers_world=chunk_inputs_data["chunk_centers_world"],
            chunk_size_world=chunk_inputs_data["chunk_size_world"],
            chunk_indices=chunk_inputs_data["chunk_indices"],
            do_fill_holes=not skip_fill_holes,
            hole_perim=hole_perim,
            do_remesh=not skip_remesh,
            remesh_res=remesh_res,
            remesh_band=remesh_band,
            remesh_project=remesh_project,
            smooth_iters=smooth_iters,
            smooth_lambda=smooth_lambda,
            smooth_mu=smooth_mu,
            smooth_feature_angle=smooth_feature_angle,
            simplify_threshold=simplify_threshold,
            texture_size=texture_size,
            dump_geometry_dir=output_dir if dump_geometry_plys else None,
            chunks_save_dir=chunks_save_dir,
            verbose=True,
        )
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(f"total time: {elapsed:.1f}s")
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"VRAM peak: {peak_gb:.2f} GB")

    if dump_geometry_plys:
        logger.info(f"dump-geometry mode: PLYs written to {output_dir}")
        return None

    out_file = output_dir / "scene.glb"
    logger.info(f"writing {out_file} ...")
    scene.export(str(out_file))
    return scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Path to to_glb_inputs.pt")
    parser.add_argument("--chunk_inputs", required=True, help="Path to chunk_inputs.pt")
    parser.add_argument("--output_dir", required=True, help="Directory to write scene.glb")
    parser.add_argument("--texture_size", type=int, default=4096)
    parser.add_argument("--simplify_threshold", type=int, default=2_000_000)
    parser.add_argument(
        "--skip_fill_holes",
        action="store_true",
        help="Skip the global cumesh.fill_holes step.",
    )
    parser.add_argument("--hole_perim", type=float, default=1.0)
    parser.add_argument(
        "--skip_remesh",
        action="store_true",
        help="Skip the global narrow-band DC remesh step.",
    )
    parser.add_argument("--remesh_res", type=int, default=None)
    parser.add_argument("--remesh_band", type=float, default=1.0)
    parser.add_argument("--remesh_project", type=float, default=0.9)
    parser.add_argument(
        "--smooth_iters",
        type=int,
        default=0,
        help="Feature-preserving Taubin λ-μ smoothing iterations applied to the "
        "(possibly skip-remeshed) global mesh before chunk partition. 0 = disabled.",
    )
    parser.add_argument(
        "--smooth_lambda",
        type=float,
        default=0.5,
        help="Taubin λ (positive smoothing factor). Typical range 0.3–0.6.",
    )
    parser.add_argument(
        "--smooth_mu",
        type=float,
        default=-0.53,
        help="Taubin μ (negative anti-shrinkage factor). Should satisfy |μ| > |λ|.",
    )
    parser.add_argument(
        "--smooth_feature_angle",
        type=float,
        default=25.0,
        help="Dihedral angle threshold (deg). Vertices on edges above this — plus all "
        "boundary / non-manifold edges — are locked and not smoothed.",
    )
    parser.add_argument(
        "--dump_geometry_plys",
        action="store_true",
        help="Diagnostic mode: dump global_mesh.ply + per-chunk chunk_NNN_geom.ply into "
        "output_dir and skip op.to_glb (no decimation, xatlas, or texture bake). Lets you "
        "inspect the pre-bake geometry and iterate on smoothing/fill_holes/remesh in seconds.",
    )
    parser.add_argument(
        "--chunks_dir",
        default=None,
        help="Directory to save per-chunk GLBs as they're baked. Already-present chunks are "
        "loaded from disk on rerun (resumable). Default: <output_dir>/chunks. "
        "Pass --no_chunk_cache to disable.",
    )
    parser.add_argument(
        "--no_chunk_cache",
        action="store_true",
        help="Disable per-chunk GLB save / resume entirely (in-memory only).",
    )
    args = parser.parse_args()

    run_chunked_to_glb(
        Path(args.inputs),
        Path(args.chunk_inputs),
        Path(args.output_dir),
        texture_size=args.texture_size,
        simplify_threshold=args.simplify_threshold,
        skip_fill_holes=args.skip_fill_holes,
        hole_perim=args.hole_perim,
        skip_remesh=args.skip_remesh,
        remesh_res=args.remesh_res,
        remesh_band=args.remesh_band,
        remesh_project=args.remesh_project,
        smooth_iters=args.smooth_iters,
        smooth_lambda=args.smooth_lambda,
        smooth_mu=args.smooth_mu,
        smooth_feature_angle=args.smooth_feature_angle,
        dump_geometry_plys=args.dump_geometry_plys,
        chunks_dir=Path(args.chunks_dir) if args.chunks_dir is not None else None,
        no_chunk_cache=args.no_chunk_cache,
    )


if __name__ == "__main__":
    main()
