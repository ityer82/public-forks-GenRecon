"""Lightweight GroundedSAM2-based segmentation pipeline: mask extraction +
point-cloud segmentation only, vendored (copied, not linked) from COB-GS's
main_light.py/detect_and_segment.py/segment_pointcloud.py for GenRecon's
own segmentation stage.

Runs Stage 1 (2D mask extraction via Grounded-SAM-2) and Stage 1.5 (direct
point-cloud segmentation via mask reprojection).

Usage (class-based/open-world mode: separate masks/point clouds per sub-class):
    uv run python main_light.py --scene kitchen --text classes \
        --classes "apple,banana,orange" --dataset_root path/to/vggt_export
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def run_stage(name, cmd, skip=False, skip_reason=None, read_from_drive=False):
    if read_from_drive and skip:
        print(f"[skip] {name}: {skip_reason} already exists")
        return
    print(f"[run]  {name}: {shlex.join(cmd)}")
    subprocess.run(cmd, check=True)


def stage1_done(mask_dir, multi_class):
    if not multi_class:
        return mask_dir.exists()
    return (mask_dir / "labels.json").exists()


def stage_done(mask_dir, multi_class):
    if not multi_class:
        return (mask_dir / "ply_pointcloud" / "object.ply").exists()
    manifest = mask_dir / "labels.json"
    if not manifest.exists():
        return False
    label_dirs = json.loads(manifest.read_text()).values()
    return (
        (mask_dir / "background" / "point_cloud" / "background.ply").exists()
        and all((mask_dir / dirname / "point_cloud" / f"{dirname}.ply").exists()
                for dirname in label_dirs)
    )


def run_segmentation(
    scene: str,
    dataset_root: str,
    output_root: str,
    classes: str | None,
    *,
    text: str = "The truck",
    resolution: int = -1,
    pc_mask_threshold: float = 0.3,
    depth_tolerance: float = 0.1,
    depth_dir: str | None = None,
    bg_assign_radius: float = 0.02,
    voxel_size: float = 0.005,
    depth_conf_thres: float = 50.0,
    depth_edge_rtol: float = 0.03,
    skip_mask: bool = False,
    skip_pc_segment: bool = False,
    skip_hull_consistency_check: bool = False,
    read_from_drive: bool = False,
    flat_output: bool = False,
    detector_backend: str = "groundingdino",
    detection_vlm_model: str = "gemma4:31b",
    detection_ollama_host: str | None = None,
    detection_num_sample_frames: int = 8,
    detection_box_padding_frac: float = 0.05,
) -> None:
    """Runs Stage 1 (2D mask extraction) and Stage 1.5 (point-cloud segmentation) for a scene.
    Dispatches to detect_and_segment.py/segment_pointcloud.py as real subprocesses (the actual
    CUDA-heavy GroundedSAM2 work) -- unchanged from the CLI's own behavior, just callable
    directly with typed args instead of building argv by hand."""
    output_path = output_root if flat_output else f"{output_root}/{scene}"
    label = text
    mask_dir = Path(output_path) / "masks" / label
    py = ["uv", "run", "python"]
    multi_class = classes is not None
    classes_flag = ["--classes", classes] if multi_class else []

    detector_flags = [
        "--detector_backend", detector_backend,
        "--detection_vlm_model", detection_vlm_model,
        "--detection_num_sample_frames", str(detection_num_sample_frames),
        "--detection_box_padding_frac", str(detection_box_padding_frac),
    ] + (["--detection_ollama_host", detection_ollama_host] if detection_ollama_host else [])

    if not skip_mask:
        run_stage(
            "Stage 1: Mask extraction",
            py + [str(SCRIPT_DIR / "detect_and_segment.py"),
                  "--dataset_root", dataset_root, "--output", output_root, "--scene", scene,
                  "--text", label, "--resolution", str(resolution), "--frame_idx", "0"] + classes_flag
            + (["--flat_output"] if flat_output else []) + detector_flags,
            skip=stage1_done(mask_dir, multi_class),
            skip_reason=mask_dir,
            read_from_drive=read_from_drive,
        )

    if not skip_pc_segment:
        if multi_class and not depth_dir:
            raise ValueError("depth_dir is required for Stage 1.5 in class-based/open-world mode")
        depth_flag = (["--depth_dir", depth_dir, "--bg_assign_radius", str(bg_assign_radius),
                        "--voxel_size", str(voxel_size),
                        "--depth_conf_thres", str(depth_conf_thres),
                        "--depth_edge_rtol", str(depth_edge_rtol)]
                       + (["--skip_hull_consistency_check"] if skip_hull_consistency_check else [])
                       if multi_class else [])
        run_stage(
            "Stage 1.5: Point-cloud segmentation",
            py + [str(SCRIPT_DIR / "segment_pointcloud.py"), "--dataset", dataset_root, "--output", output_path,
                  "--text", label, "--pc_mask_threshold", str(pc_mask_threshold),
                  "--depth_tolerance", str(depth_tolerance)] + classes_flag + depth_flag,
            skip=stage_done(mask_dir, multi_class),
            skip_reason=mask_dir,
            read_from_drive=read_from_drive,
        )

    print("Pipeline complete.")


def main():
    parser = argparse.ArgumentParser(description="Lightweight GroundedSAM2 segmentation pipeline (mask extraction + point-cloud segmentation)")
    parser.add_argument("--scene", type=str, default="truck")
    parser.add_argument("--text", type=str, default="The truck")
    parser.add_argument("--classes", type=str, default=None,
                         help="Comma-separated candidate sub-classes for open-world, "
                              "class-based segmentation (e.g. \"apple,banana,orange\"). "
                              "When set, --text is only used as the output-directory "
                              "label and each detected object is tracked/saved under "
                              "its matched class instead of one merged mask.")
    parser.add_argument("--dataset_root", type=str, default="dataset",
                         help="Base directory containing this scene's images/ and "
                              "sparse/0/ (e.g. a VGGT-Omega --export-for-3dgs output dir). "
                              "Passed straight through to detect_and_segment.py/"
                              "segment_pointcloud.py, no <scene> subdir is appended.")
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--pc_mask_threshold", type=float, default=0.3,
                         help="Foreground vote threshold for Stage 1.5 point-cloud segmentation "
                              "(single-class mode only)")
    parser.add_argument("--depth_tolerance", type=float, default=0.1,
                         help="Relative z-buffer tolerance for Stage 1.5 point-cloud segmentation")
    parser.add_argument("--depth_dir", type=str, default=None,
                         help="Directory of <frame_stem>_depth.npy per-pixel depth maps (e.g. a "
                              "VGGT-Omega --export-for-3dgs output's depth/ dir). Required for "
                              "Stage 1.5 in class-based mode (--classes), "
                              "which directly lifts 2D mask pixels to 3D points via this depth.")
    parser.add_argument("--bg_assign_radius", type=float, default=0.02,
                         help="Class-based mode: world-space distance within which a sparse COLMAP "
                              "point is considered claimed by a lifted class point cloud (otherwise "
                              "background)")
    parser.add_argument("--voxel_size", type=float, default=0.005,
                         help="Class-based mode: grid-snap dedup cell size for lifted class points "
                              "(0 disables)")
    parser.add_argument("--depth_conf_thres", type=float, default=50.0,
                         help="Class-based mode: percentile (0-100) of per-pixel depth_conf below "
                              "which a candidate pixel is dropped before lifting (mirrors "
                              "VGGT-Omega's own points3D.ply confidence filtering)")
    parser.add_argument("--depth_edge_rtol", type=float, default=0.03,
                         help="Class-based mode: relative local depth-jump threshold above which a "
                              "pixel is excluded as a depth discontinuity before lifting")
    parser.add_argument("--skip_mask", action="store_true")
    parser.add_argument("--skip_pc_segment", action="store_true",
                         help="Skip Stage 1.5: direct point-cloud segmentation via mask reprojection")
    parser.add_argument("--skip_hull_consistency_check", action="store_true",
                         help="Stage 1.5, class-based mode: skip the per-class RANSAC volumetric-"
                              "hull consensus check (see segment_pointcloud.py)")
    parser.add_argument("--read_from_drive", action="store_true",
                         help="Skip a stage if its output already exists on disk. "
                              "Default behavior is to always rerun and overwrite "
                              "existing outputs.")
    parser.add_argument("--flat_output", action="store_true",
                         help="Skip the <scene> path segment under --output_root: output_path "
                              "becomes exactly --output_root instead of --output_root/<scene>. "
                              "Useful when --output_root is already scene-specific (e.g. driven "
                              "by an external per-scene pipeline).")
    parser.add_argument("--detector_backend", choices=["groundingdino", "gemma"], default="groundingdino",
                         help="Box-detection backend for Stage 1. 'gemma' uses a local "
                              "Ollama-served Gemma vision model instead of Grounding DINO -- "
                              "see segmentation/gemma_detection_utils.py.")
    parser.add_argument("--detection_vlm_model", type=str, default="gemma4:31b",
                         help="Ollama model tag used when --detector_backend gemma.")
    parser.add_argument("--detection_ollama_host", type=str, default=None,
                         help="Ollama base URL override for --detector_backend gemma.")
    parser.add_argument("--detection_num_sample_frames", type=int, default=8,
                         help="--detector_backend gemma, --classes mode only: number of "
                              "evenly-spaced frames sent to Gemma for the initial scan.")
    parser.add_argument("--detection_box_padding_frac", type=float, default=0.05,
                         help="--detector_backend gemma only: outward box padding fraction "
                              "applied before SAM2 prompting.")
    args = parser.parse_args()

    try:
        run_segmentation(
            args.scene,
            args.dataset_root,
            args.output_root,
            args.classes,
            text=args.text,
            resolution=args.resolution,
            pc_mask_threshold=args.pc_mask_threshold,
            depth_tolerance=args.depth_tolerance,
            depth_dir=args.depth_dir,
            bg_assign_radius=args.bg_assign_radius,
            voxel_size=args.voxel_size,
            depth_conf_thres=args.depth_conf_thres,
            depth_edge_rtol=args.depth_edge_rtol,
            skip_mask=args.skip_mask,
            skip_pc_segment=args.skip_pc_segment,
            skip_hull_consistency_check=args.skip_hull_consistency_check,
            read_from_drive=args.read_from_drive,
            flat_output=args.flat_output,
            detector_backend=args.detector_backend,
            detection_vlm_model=args.detection_vlm_model,
            detection_ollama_host=args.detection_ollama_host,
            detection_num_sample_frames=args.detection_num_sample_frames,
            detection_box_padding_frac=args.detection_box_padding_frac,
        )
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
