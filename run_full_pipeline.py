#!/usr/bin/env python3
"""End-to-end pipeline: VGGT-Omega pose/depth prediction -> [optional GroundedSAM2 3D
segmentation] -> GenRecon reconstruction -> GLB bake.

Python port of run_full_pipeline.sh: same CLI surface, but every non-Isaac, non-TRELLIS.2 stage
runs in-process as a typed Python function call instead of a subprocess with a hand-built argv
string (see genrecon/pipeline/stages.py). Isaac Sim (../IsaacSim) remains a subprocess, since it
still runs in its own separate uv env.

Usage:
    uv run python run_full_pipeline.py <image_folder> <scene_name> [options...]
    uv run python run_full_pipeline.py <video_file> <scene_name> --video [options...]

Example:
    uv run python run_full_pipeline.py /path/to/images food2_vggt --classes "banana,bowl" --mode pick-and-place \\
        --pick_place_target banana --place-target bowl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GENRECON_DIR = Path(__file__).resolve().parent
ISAACSIM_DIR = GENRECON_DIR.parent / "IsaacSim"
MVSAM3D_VENDOR_DIR = GENRECON_DIR / "mv_sam3d"
VGGT_CHECKPOINT = GENRECON_DIR / "checkpoints" / "vggt_omega" / "ckpts" / "vggt_omega_1b_512.pt"
FOUNDATION_STEREO_CHECKPOINT = GENRECON_DIR / "checkpoints" / "foundation_stereo" / "11-33-40" / "model_best_bp2.pth"
COLLISION_APPROXIMATION = "convexDecomposition"
RUN_GLB = False
SIMPLIFY_THRESHOLD = 250_000
TEXTURE_SIZE = 2048
USE_TRELLIS = True
MODES = ("pick-and-place", "full-scene", "full-scene-with-robot")
FULL_SCENE_MODES = ("full-scene", "full-scene-with-robot")
VIDEO_FPS = 5
VIDEO_WIDTH, VIDEO_HEIGHT = 960, 540  # matches ../fisheye/data/food
DISCOVERY_HF_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # one-shot object labels + boxes (Stage 0b)


def _place_offset(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--place-offset must be DX,DY,DZ (got {s!r})")
    return (parts[0], parts[1], parts[2])


def extract_video_frames(video: Path, out_dir: Path) -> int:
    """ffmpeg: sample `video` at VIDEO_FPS, resize to VIDEO_WIDTHxVIDEO_HEIGHT, write JPEGs to out_dir."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("--video requires ffmpeg on PATH.")
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-vf", f"fps={VIDEO_FPS},scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}",
            "-q:v", "2", str(out_dir / "frame_%06d.jpg"),
        ],
        check=True,
    )
    return len(list(out_dir.glob("frame_*.jpg")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_folder", type=Path, help="Folder of images (or a video file with --video).")
    parser.add_argument("scene_name", type=str)

    parser.add_argument(
        "--video", action="store_true", default=False,
        help=f"Treat the first positional argument as a video file: extract frames at {VIDEO_FPS} fps, "
        f"resized to {VIDEO_WIDTH}x{VIDEO_HEIGHT}, into a temporary folder (deleted when the run "
        "ends) that Stage 0 reads instead. All later stages use Stage 0's outputs.",
    )
    parser.add_argument("--num_imgs_per_scene", type=int, default=32)
    parser.add_argument("--skip-frames", dest="skip_frames", type=int, default=-1)
    parser.add_argument(
        "--max-frames", dest="max_frames", type=int, default=-1,
        help="Only use the first N images (sorted by filename) from image_folder. Applied before "
        "--skip-frames. -1 (default) uses all images.",
    )
    parser.add_argument("--rotate-horizontal-deg", type=float, default=0.0)
    parser.add_argument(
        "--classes", type=str, default=None,
        help="Comma-separated object labels (e.g. \"banana,bowl\"). pick-and-place mode: "
        "conditions Stage 0b's Qwen3-VL pass to look for exactly these objects instead of "
        "freely discovering everything on the table (see --mode). full-scene mode: required, "
        "passed to Grounding DINO.",
    )
    parser.add_argument(
        "--mode", dest="mode", choices=list(MODES), default="pick-and-place",
        help="Run mode. 'pick-and-place' (default): a Qwen3-VL pass (Stage 0b) always retrieves "
        "object labels + boxes -- conditioned on --classes if given, else freely discovering "
        "every object on the table -- and a random pick/place pair is chosen. "
        "'full-scene': Grounding DINO detector; requires --classes; full-scene reconstruction. "
        "'full-scene-with-robot': like full-scene, then a robot-collision demo against a "
        "randomly chosen class (or --robot-target if given).",
    )
    parser.add_argument(
        "--detection-box-padding-frac", dest="detection_box_padding_frac", type=float, default=0.05,
        help="pick-and-place mode only: outward padding applied to each Stage 0b-discovered box "
        "(as a fraction of its own width/height) before it's used to prompt SAM2. Mitigates "
        "an undershoot failure mode found in testing where a box that doesn't fully enclose "
        "its object makes SAM2 truncate the mask at the wrong edge.",
    )
    parser.add_argument("--mvsam3d-stage1-steps", dest="mvsam3d_stage1_steps", type=int, default=25,
                        help="MV-SAM3D stage 1 (shape) inference steps. The upstream MV-SAM3D default is 50.")
    parser.add_argument("--mvsam3d-stage2-steps", dest="mvsam3d_stage2_steps", type=int, default=12,
                        help="MV-SAM3D stage 2 (texture) inference steps. The upstream MV-SAM3D default is 25.")
    parser.add_argument("--mvsam3d-top-k-views", dest="mvsam3d_top_k_views", type=int, default=5,
                        help="Prune to the k views that best cover each object angularly before "
                             "MV-SAM3D's main diffusion pass. Pass 0 or a value >= the scene's "
                             "view count to disable pruning (use every view with a mask).")
    parser.add_argument("--mvsam3d-align-shape-latents", dest="mvsam3d_align_shape_latents",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="MV-SAM3D stage 1: align each view's canonical frame (cube rotation found "
                             "from a single-view pre-pass) to the reference view before fusing, "
                             "fixing crossed slabs when views disagree on the canonical frame. "
                             "On by default; pass --no-mvsam3d-align-shape-latents to disable.")
    parser.add_argument("--skip_isaac", dest="run_usd", action="store_false", default=True)
    parser.add_argument(
        "--friction-table-path", type=Path,
        default=GENRECON_DIR / "configs" / "materials" / "friction_table.example.yaml",
    )
    parser.add_argument("--ollama-model", dest="ollama_model", default="llama3.1:8b")
    parser.add_argument("--start-from-stage", type=int, default=0)
    parser.add_argument("--stop-after-stage", type=int, default=999)
    parser.add_argument("--max_chunks_per_group", type=int, default=None)
    parser.add_argument("--max_inflated_voxels", type=int, default=None)
    parser.add_argument("--depth_conf_thres", type=float, default=50.0)
    parser.add_argument("--depth_edge_rtol", type=float, default=0.03)
    parser.add_argument("--fix_num_chunks", type=int, default=16)
    parser.add_argument("--robot-target", dest="robot_target", default=None)
    # Lowered from demo_rerun.py's own CLI default (50.0, a percentile filter that discards half
    # of every exported point cloud by construction) -- see vggt/demo_rerun.py's --conf-thres help.
    parser.add_argument("--vggt-conf-thres", dest="vggt_conf_thres", type=float, default=20.0)
    parser.add_argument(
        "--vggt-viewer", dest="vggt_viewer", action="store_true", default=False,
        help="Spawn the rerun point-cloud viewer after Stage 0's export (background thread). Off "
        "by default -- nothing downstream reads from it, and it adds nothing to a headless run.",
    )
    parser.add_argument(
        "--stage-0-use-stereo", dest="stage_0_use_stereo", action="store_true", default=False,
        help="Run FoundationStereo instead of VGGT-Omega in Stage 0. image_folder must hold exactly "
        "one left and one right image (filenames containing 'left'/'right') from a rectified stereo "
        "pair, plus a ZED .conf calibration (see --stereo-conf). pick-and-place mode only. "
        "Everything after Stage 0 is unchanged.",
    )
    parser.add_argument(
        "--stereo-conf", dest="stereo_conf", type=Path, default=None,
        help="ZED .conf calibration file for --stage-0-use-stereo. Default: the single *.conf in image_folder.",
    )
    parser.add_argument("--pick_place_target", default=None)
    parser.add_argument("--place-offset", dest="place_offset", type=_place_offset, default=(0.3, 0.0, 0.0))
    parser.add_argument("--place-target", dest="place_target", default=None)
    parser.add_argument("--place-target-clearance", dest="place_target_clearance", type=float, default=0.05)
    parser.add_argument("--gripper-open-width", dest="gripper_open_width", type=float, default=0.06)
    parser.add_argument("--approach-side", dest="approach_side", choices=["neg-x", "pos-x", "neg-y", "pos-y"], default="neg-y")
    parser.add_argument("--ai-scene-agent", dest="ai_scene_agent", action="store_true", default=False)
    parser.add_argument("--scene-agent-ollama-model", dest="scene_agent_ollama_model", default="qwen2.5:7b")
    parser.add_argument(
        "--room", dest="room", action="store_true", default=False,
        help="Wrap the table in compose_isaac_scene.py's static kitchen room backdrop "
        "(requires --table, which is on by default in compose_isaac_scene.py).",
    )
    parser.add_argument(
        "--room-asset-dir", dest="room_asset_dir", type=Path, default=None,
        help="Override compose_isaac_scene.py's --room-asset-dir (default: assets/room "
        "relative to the IsaacSim script).",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, logger) -> tuple[list[str], bool]:
    """Ports run_full_pipeline.sh's cross-field validation (lines ~168-259). Returns the parsed
    --classes list and the effective use_trellis flag (derived from USE_TRELLIS, overridden per
    the same rules bash silently applied). Calls parser.error() (exit 2) on invalid combinations,
    matching bash's exit-1-with-message behavior closely enough for a CLI tool."""
    if args.video:
        if not args.image_folder.is_file():
            parser.error(f"Video file not found: {args.image_folder}")
        if args.stage_0_use_stereo:
            parser.error("--video cannot be combined with --stage-0-use-stereo.")
    elif not args.image_folder.is_dir():
        parser.error(f"Image folder not found: {args.image_folder}")

    full_scene_mode = args.mode in FULL_SCENE_MODES

    if args.stage_0_use_stereo:
        if args.mode != "pick-and-place":
            parser.error("--stage-0-use-stereo is only valid with --mode pick-and-place.")
        for flag, used in (
            ("--skip-frames", args.skip_frames != -1),
            ("--max-frames", args.max_frames != -1),
            ("--vggt-viewer", args.vggt_viewer),
        ):
            if used:
                parser.error(f"{flag} cannot be combined with --stage-0-use-stereo.")
        if args.start_from_stage <= 0 and not FOUNDATION_STEREO_CHECKPOINT.is_file():
            parser.error(f"FoundationStereo checkpoint not found at {FOUNDATION_STEREO_CHECKPOINT}.")
        if args.stereo_conf is None:
            confs = sorted(args.image_folder.glob("*.conf"))
            if len(confs) != 1:
                parser.error(
                    f"--stage-0-use-stereo needs --stereo-conf, or exactly one *.conf in {args.image_folder} "
                    f"(found {len(confs)})."
                )
            args.stereo_conf = confs[0]
        elif not args.stereo_conf.is_file():
            parser.error(f"--stereo-conf file not found: {args.stereo_conf}")
    elif args.stereo_conf is not None:
        parser.error("--stereo-conf requires --stage-0-use-stereo.")

    if full_scene_mode:
        if not args.classes:
            parser.error(f"--mode {args.mode} requires --classes.")
        for flag, value in (
            ("--pick_place_target", args.pick_place_target),
            ("--place-target", args.place_target),
            ("--ai-scene-agent", args.ai_scene_agent),
        ):
            if value:
                parser.error(f"{flag} is only valid with --mode pick-and-place.")
        if args.mode == "full-scene" and args.robot_target:
            parser.error("--robot-target requires --mode full-scene-with-robot.")
        if args.mode == "full-scene-with-robot" and not args.run_usd:
            parser.error("--mode full-scene-with-robot cannot be combined with --skip_isaac.")
    elif args.robot_target:
        parser.error("--robot-target requires --mode full-scene-with-robot.")

    # pick-and-place picks a random pair unless the pick target is given or the scene agent chooses.
    args.random_pick_place = (
        args.mode == "pick-and-place" and not args.pick_place_target and not args.ai_scene_agent
    )

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else []

    # In pick-and-place mode, Stage 0b's Qwen3-VL pass always runs and always ends up populating
    # `classes` (conditioned on --classes if given, else freely discovered) by the time Stage 1
    # needs them, so per-class mesh reconstruction is never skipped there. It's only skipped in
    # full-scene mode without --classes -- but --classes is required there (checked above), so
    # `classes` is never empty at this point either.
    use_trellis = USE_TRELLIS

    if not MVSAM3D_VENDOR_DIR.is_dir():
        parser.error(f"Requires the vendored MV-SAM3D source at {MVSAM3D_VENDOR_DIR}.")

    if args.pick_place_target:
        if args.robot_target:
            parser.error("--pick_place_target and --robot-target are mutually exclusive.")
        if not classes:
            parser.error("--pick_place_target requires --classes to include the same label.")
        if args.pick_place_target not in classes:
            parser.error(f"--pick_place_target '{args.pick_place_target}' must exactly match one of the labels passed to --classes ('{args.classes}').")
        if not use_trellis:
            logger.info("Note: --pick_place_target requires TRELLIS.2's per-object mesh; overriding use_trellis to on for this run.")
            use_trellis = True

    if args.place_target:
        if not args.pick_place_target:
            parser.error("--place-target requires --pick_place_target to be set.")
        if args.place_target == args.pick_place_target:
            parser.error(f"--place-target must differ from --pick_place_target ('{args.pick_place_target}').")
        if args.place_target not in classes:
            parser.error(f"--place-target '{args.place_target}' must exactly match one of the labels passed to --classes ('{args.classes}').")

    if args.ai_scene_agent:
        if args.pick_place_target:
            parser.error("--ai-scene-agent and --pick_place_target are mutually exclusive.")
        if args.robot_target:
            parser.error("--ai-scene-agent and --robot-target are mutually exclusive.")
        if not use_trellis:
            logger.info("Note: --ai-scene-agent requires a per-object mesh for every --classes label; overriding use_trellis to on for this run.")
            use_trellis = True

    return classes, use_trellis


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m{s % 60:02d}s"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = GENRECON_DIR / "runs" / args.scene_name
    export_dir = run_dir / "stage_0_vggt"
    output_dir = run_dir / "stage_3_genrecon"

    pipeline_log = run_dir / "pipeline.log"
    debug_config_log = run_dir / "debug_config.log"
    # Must happen before genrecon.utils.logger (or anything that imports it, including
    # genrecon.pipeline.stages) is first imported anywhere in the process -- loguru's file sink
    # is added once, at that first import, based on this env var.
    os.environ["GENRECON_PIPELINE_LOG"] = str(pipeline_log)

    from genrecon.pipeline import stages
    from genrecon.pipeline.subprocess_utils import LogMirror, stage
    from genrecon.utils.logger import logger

    classes, use_trellis = validate_args(parser, args, logger)
    if args.mode == "full-scene-with-robot" and not args.robot_target:
        args.robot_target = random.choice(classes)
        logger.info(f"Random robot target: '{args.robot_target}' (chosen from classes {classes})")
    log_mirror = LogMirror(pipeline_log)

    def check_stop_after_stage(n: int) -> None:
        if args.stop_after_stage == n:
            logger.info(f"Stopping after stage {n} (--stop-after-stage {args.stop_after_stage}).")
            sys.exit(0)

    compose_scene_extra_args: list[str] = []
    if args.room:
        compose_scene_extra_args.append("--room")
        if args.room_asset_dir:
            compose_scene_extra_args.extend(["--room-asset-dir", str(args.room_asset_dir)])

    # compose_isaac_scene.py's --table defaults to on (it was built for the pick-and-place demo,
    # Stage P3 below, where a real table prop under the picked/placed objects is correct). The
    # full-reconstruction path (Stage 9) already has its own reconstructed background/floor
    # mesh spanning the whole scene, so the same small fixed-size table prop would get centered
    # underneath that mesh's bounding box at floor height -- making the entire reconstructed
    # scene appear perched on a tiny table. Suppress it there.
    full_scene_compose_extra_args = compose_scene_extra_args + ["--no-table"]

    pipeline_t0 = time.monotonic()
    logger.info(f"Pipeline started for scene '{args.scene_name}'")

    video_tmp_dir: Path | None = None
    try:
        # ── Video input: extract + downsample + resize into a temp image folder for Stage 0 ──
        if args.video and args.start_from_stage <= 0:
            video_tmp_dir = Path(tempfile.mkdtemp(prefix=f"{args.scene_name}_frames_"))
            with stage(f"Extracting frames from {args.image_folder} ({VIDEO_FPS} fps, {VIDEO_WIDTH}x{VIDEO_HEIGHT}) -> {video_tmp_dir}"):
                n_frames = extract_video_frames(args.image_folder, video_tmp_dir)
                if n_frames == 0:
                    raise RuntimeError(f"ffmpeg extracted no frames from {args.image_folder}")
                logger.info(f"Extracted {n_frames} frames")
            args.image_folder = video_tmp_dir

        # ── Stage 0: VGGT-Omega export (or FoundationStereo with --stage-0-use-stereo) ──
        if args.start_from_stage <= 0 and args.stage_0_use_stereo:
            with stage(f"Stage 0: FoundationStereo export -> {export_dir}"):
                export_dir.mkdir(parents=True, exist_ok=True)
                stages.stage0_stereo_export(
                    args.image_folder,
                    FOUNDATION_STEREO_CHECKPOINT,
                    args.stereo_conf,
                    export_dir,
                    align_to_gravity=True,
                    rotate_horizontal_deg=args.rotate_horizontal_deg,
                    conf_thres=args.vggt_conf_thres,
                )
        elif args.start_from_stage <= 0:
            with stage(f"Stage 0: VGGT-Omega export -> {export_dir}"):
                export_dir.mkdir(parents=True, exist_ok=True)
                stages.stage0_vggt_export(
                    args.image_folder,
                    VGGT_CHECKPOINT,
                    export_dir,
                    skip_frames=args.skip_frames,
                    max_frames=args.max_frames,
                    align_to_gravity=True,
                    rotate_horizontal_deg=args.rotate_horizontal_deg,
                    conf_thres=args.vggt_conf_thres,
                    spawn_viewer=args.vggt_viewer,
                )
        else:
            logger.info(f"Stage 0: skipped (--start-from-stage {args.start_from_stage}), assuming existing export at {export_dir}")
        check_stop_after_stage(0)

        if args.mode == "pick-and-place":
            # Cached to run_dir so a --start-from-stage rerun that skips Stage 1 (segmentation)
            # can't silently diverge from the class list Stage 1 actually segmented against --
            # discovery is model-sampled and can legitimately return different phrasing across
            # calls on the same images, which would otherwise leave a later stage looking for
            # masks under class names Stage 1 never produced.
            discovered_classes_cache = run_dir / "discovered_classes.json"
            discovered_boxes_cache = run_dir / "discovered_boxes.json"
            if args.start_from_stage > 0 and discovered_classes_cache.exists() and discovered_boxes_cache.exists():
                classes = json.loads(discovered_classes_cache.read_text())
                logger.info(
                    f"Stage 0b: skipped (--start-from-stage {args.start_from_stage}), reusing "
                    f"cached classes from {discovered_classes_cache}: {', '.join(classes)}"
                )
            else:
                with stage(f"Stage 0b: object discovery + detection (model={DISCOVERY_HF_MODEL})"):
                    from genrecon.utils.object_discovery import discover_objects

                    left_image, objects = discover_objects(
                        export_dir / "rgb", model_id=DISCOVERY_HF_MODEL, classes=classes or None,
                    )
                    classes = [o["label"] for o in objects]
                    logger.info(f"Discovery: found {len(classes)} objects: {', '.join(classes)}")
                discovered_classes_cache.write_text(json.dumps(classes, indent=2))
                discovered_boxes_cache.write_text(json.dumps({
                    "left_image": left_image,
                    "objects": {o["label"]: {"left_box": o["left_box"], "right_box": o["right_box"]}
                                for o in objects},
                }, indent=2))
            boxes_json = discovered_boxes_cache
        else:
            boxes_json = None

        if args.random_pick_place:
            if len(classes) < 2:
                logger.error(f"pick-and-place mode requires at least 2 classes for random selection, got {classes}.")
                sys.exit(1)
            args.pick_place_target, args.place_target = random.sample(classes, 2)
            logger.info(
                f"Random pick-and-place: pick='{args.pick_place_target}', "
                f"place='{args.place_target}' (chosen from classes {classes})"
            )

        cobgs_mask_dir = run_dir / "stage_1_segmentation"
        image_to_3d_output_dir = run_dir / "stage_2_mv_sam3d"
        seg_log = run_dir / "stage_1_segmentation" / "segmentation.log"
        mvsam3d_log = image_to_3d_output_dir / "mv_sam3d.log"

        # ── Stage 1-2: segmentation + per-class 3D reconstruction (only if --classes) ──
        if classes:
            if args.start_from_stage <= 1:
                with stage(f"Stage 1: segmentation (classes: {classes})"):
                    cobgs_mask_dir = stages.stage1_segmentation(
                        args.scene_name,
                        export_dir,
                        run_dir / "stage_1_segmentation",
                        classes,
                        depth_conf_thres=args.depth_conf_thres,
                        depth_edge_rtol=args.depth_edge_rtol,
                        seg_log=seg_log,
                        log_mirror=log_mirror,
                        detection_box_padding_frac=args.detection_box_padding_frac,
                        boxes_json=boxes_json,
                    )
            else:
                logger.info(f"Stage 1: skipped (--start-from-stage {args.start_from_stage}), assuming existing segmentation at {run_dir / 'stage_1_segmentation'}")
            check_stop_after_stage(1)

            if use_trellis and args.start_from_stage <= 2:
                with stage(f"Stage 2: MV-SAM3D reconstruction -> {image_to_3d_output_dir}"):
                    stages.stage2_mvsam3d(
                        run_dir,
                        args.scene_name,
                        classes,
                        image_to_3d_output_dir,
                        mvsam3d_vendor_dir=MVSAM3D_VENDOR_DIR,
                        log_file=mvsam3d_log,
                        log_mirror=log_mirror,
                        cobgs_mask_dir=cobgs_mask_dir,
                        stage1_steps=args.mvsam3d_stage1_steps,
                        stage2_steps=args.mvsam3d_stage2_steps,
                        top_k_views=args.mvsam3d_top_k_views,
                        align_shape_latents=args.mvsam3d_align_shape_latents,
                    )
            else:
                logger.info(f"Stage 2: skipped (use_trellis disabled, or --start-from-stage {args.start_from_stage})")
            check_stop_after_stage(2)

        # ── Stage P0 (only with --ai-scene-agent) ──
        pick_place_target = args.pick_place_target
        place_target = args.place_target
        place_offset = args.place_offset
        place_target_clearance = args.place_target_clearance
        gripper_open_width = args.gripper_open_width
        approach_side = args.approach_side
        scene_agent_extra_args: list[str] = []

        if args.ai_scene_agent:
            if classes:
                available_classes = stages.classes_with_meshes(classes, image_to_3d_output_dir)
                missing = [c for c in classes if c not in available_classes]
                if missing:
                    logger.warning(
                        f"Stage P0: {len(missing)} of {len(classes)} --classes label(s) have no "
                        f"generated mesh under {image_to_3d_output_dir} and won't be offered to "
                        f"the scene agent: {missing}"
                    )
            else:
                available_classes = stages.discover_classes_from_mesh_dir(image_to_3d_output_dir)
                logger.info(
                    f"Stage P0: --classes not set; discovered {len(available_classes)} class(es) "
                    f"from existing meshes under {image_to_3d_output_dir}: {available_classes}"
                )
                # No --classes was given, so this discovered list is the only one there is --
                # propagate it to `classes` so Stage P1 (which aligns every label in `classes`,
                # unfiltered) has something to align instead of silently processing nothing.
                classes = available_classes
            if len(available_classes) < 2:
                source = f"--classes {args.classes}" if classes else f"meshes under {image_to_3d_output_dir}"
                logger.error(
                    f"--ai-scene-agent requires at least 2 classes with a generated mesh, got "
                    f"{available_classes} (from {source})."
                )
                sys.exit(1)

            with stage(f"Stage P0: interactive scene agent (model={args.scene_agent_ollama_model})"):
                result = stages.stageP0_scene_agent(
                    available_classes,
                    image_to_3d_output_dir,
                    run_dir / "pick_place" / "scene_spec.json",
                    ollama_model=args.scene_agent_ollama_model,
                )
                pick_place_target = result["pick_target"]
                place_target = result["place_target"]
                if result["place_offset"] is not None:
                    place_offset = tuple(result["place_offset"])
                place_target_clearance = result["place_target_clearance"]
                gripper_open_width = result["gripper_open_width"]
                approach_side = result["approach_side"]

                def _opt_eq(flag: str, value) -> None:
                    if value is not None and value != "":
                        scene_agent_extra_args.extend([f"{flag}={value}"])

                _opt_eq("--start-distance", result.get("start_distance"))
                _opt_eq("--lighting-mode", result.get("lighting_mode"))
                lighting = result.get("lighting") or {}
                _opt_eq("--dome-light-intensity", lighting.get("dome_intensity"))
                dome_color = lighting.get("dome_color")
                if dome_color:
                    scene_agent_extra_args.append(f"--dome-light-color={','.join(str(v) for v in dome_color)}")
                _opt_eq("--distant-light-intensity", lighting.get("distant_intensity"))
                _opt_eq("--distant-light-angle", lighting.get("distant_angle"))
                distant_rotation = lighting.get("distant_rotation_deg")
                if distant_rotation:
                    scene_agent_extra_args.append(f"--distant-light-rotation-deg={','.join(str(v) for v in distant_rotation)}")
                camera = result.get("camera") or {}
                _opt_eq("--camera-mode", camera.get("mode"))
                _opt_eq("--camera-distance-multiplier", camera.get("distance_multiplier"))

        # ── Stage P1-P4 (fast path, mutually exclusive with Stage 3-12) ──
        if pick_place_target:
            pick_place_dir = run_dir / "pick_place"
            pick_place_glb_dir = pick_place_dir / "glb"
            pick_place_scene_usda = pick_place_glb_dir / "scene.usda"

            with stage(f"Stage P1: aligning meshes for all --classes labels to scene scale -> {pick_place_glb_dir}"):
                stages.stageP1_align_meshes(
                    run_dir, classes, cobgs_mask_dir, pick_place_glb_dir
                )

            with stage(f"Stage P2: convert_asset.py (collision_approximation={COLLISION_APPROXIMATION}) -> {pick_place_glb_dir}/<label>/asset.usd"):
                stages.stageP2_convert_asset(
                    pick_place_glb_dir,
                    isaacsim_dir=ISAACSIM_DIR,
                    collision_approximation=COLLISION_APPROXIMATION,
                    log_file=pick_place_dir / "convert_asset.log",
                    debug_config_log=debug_config_log,
                    log_mirror=log_mirror,
                )

            with stage(f"Stage P3: compose_isaac_scene.py -> {pick_place_scene_usda}"):
                stages.stageP3_compose_isaac_scene(
                    pick_place_glb_dir,
                    pick_place_scene_usda,
                    isaacsim_dir=ISAACSIM_DIR,
                    log_file=pick_place_dir / "compose_isaac_scene.log",
                    debug_config_log=debug_config_log,
                    log_mirror=log_mirror,
                    extra_args=compose_scene_extra_args,
                )

            # Sanitized here (space/slash -> underscore), not earlier: compose_isaac_scene.py always
            # names USD prims from the sanitized <label> dirname stageP1 wrote under pick_place_glb_dir
            # (see stages.sanitize_label), never the raw --classes spelling -- so a multi-word label
            # (e.g. an --ai-scene-agent pick like "plastic cup") must be sanitized before it's used as
            # demo_franka_pickplace.py's --pick-target/--place-target, or it won't match any prim.
            sanitized_pick_target = stages.sanitize_label(pick_place_target)
            sanitized_place_target = stages.sanitize_label(place_target) if place_target else None
            with stage(f"Stage P4: demo_franka_pickplace.py (pick_target={sanitized_pick_target}) -> {pick_place_dir}/pick_place.mp4"):
                stages.stageP4_franka_pickplace(
                    pick_place_scene_usda,
                    pick_place_dir,
                    isaacsim_dir=ISAACSIM_DIR,
                    pick_target=sanitized_pick_target,
                    place_target=sanitized_place_target,
                    place_offset=place_offset,
                    place_target_clearance=place_target_clearance,
                    gripper_open_width=gripper_open_width,
                    approach_side=approach_side,
                    scene_agent_extra_args=scene_agent_extra_args,
                    debug_config_log=debug_config_log,
                    log_mirror=log_mirror,
                )

            elapsed = time.monotonic() - pipeline_t0
            logger.info(f"Pipeline finished for scene '{args.scene_name}' (pick-and-place fast path, total elapsed {_format_duration(elapsed)})")
            logger.info(f"Done: {pick_place_dir}/pick_place.mp4")
            return

        # ── Stage 3: GenRecon reconstruction ──
        if args.start_from_stage <= 3:
            with stage("Stage 3: reconstruct_scene.py"):
                output_dir.mkdir(parents=True, exist_ok=True)
                stages.stage3_reconstruct_scene(
                    export_dir,
                    output_dir,
                    classes,
                    cobgs_mask_dir if classes else None,
                    num_imgs_per_scene=args.num_imgs_per_scene,
                    max_chunks_per_group=args.max_chunks_per_group,
                    max_inflated_voxels=args.max_inflated_voxels,
                    fix_num_chunks=args.fix_num_chunks,
                )
        else:
            logger.info(f"Stage 3: skipped (--start-from-stage {args.start_from_stage})")
        check_stop_after_stage(3)

        # ── Stage 4: reprojection validation ──
        if args.start_from_stage <= 4:
            with stage("Stage 4: render_reprojection_validation.py"):
                stages.stage4_reprojection_validation(output_dir, export_dir, classes, run_dir / "stage_4_reproject_synth")
        else:
            logger.info(f"Stage 4: skipped (--start-from-stage {args.start_from_stage})")
        check_stop_after_stage(4)

        shapes_dir = output_dir / "shapes"

        # ── Stage 5: per-object mesh extraction ──
        if classes and args.start_from_stage <= 5:
            with stage(f"Stage 5: cascading per-object mesh extraction -> {shapes_dir}"):
                stages.stage5_extract_object_meshes(
                    shapes_dir,
                    export_dir,
                    cobgs_mask_dir,
                    classes,
                    output_dir,
                )
        else:
            logger.info(f"Stage 5: skipped (no --classes, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(5)

        # ── Stage 6: floor segmentation + friction inference ──
        if classes and args.start_from_stage <= 6:
            with stage(f"Stage 6: extract_floor_mesh.py + infer_friction_assignments.py -> {shapes_dir}"):
                stages.stage6_floor_and_friction(
                    shapes_dir, cobgs_mask_dir,
                    friction_table_path=args.friction_table_path,
                    ollama_model=args.ollama_model,
                )
        else:
            logger.info(f"Stage 6: skipped (no --classes, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(6)

        # ── Stage 7/8/9: mesh -> GLB -> USD -> composed scene ──
        if args.run_usd:
            if args.start_from_stage <= 7:
                with stage(f"Stage 7: mesh_to_glb.py -> {shapes_dir}/glb"):
                    stages.stage7_mesh_to_glb(
                        shapes_dir, run_dir, classes, use_trellis=use_trellis
                    )
            else:
                logger.info(f"Stage 7: skipped (--start-from-stage {args.start_from_stage})")
            check_stop_after_stage(7)

            if args.start_from_stage <= 8:
                with stage(f"Stage 8: convert_asset.py (collision_approximation={COLLISION_APPROXIMATION}) -> {shapes_dir}/glb/<label>/asset.usd"):
                    stages.stage8_convert_asset(
                        shapes_dir, classes,
                        isaacsim_dir=ISAACSIM_DIR,
                        collision_approximation=COLLISION_APPROXIMATION,
                        output_dir=output_dir,
                        debug_config_log=debug_config_log,
                        log_mirror=log_mirror,
                    )
            else:
                logger.info(f"Stage 8: skipped (--start-from-stage {args.start_from_stage})")
            check_stop_after_stage(8)

            if args.start_from_stage <= 9:
                with stage(f"Stage 9: compose_isaac_scene.py -> {shapes_dir}/glb/scene.usda"):
                    stages.stage9_compose_isaac_scene(
                        shapes_dir, isaacsim_dir=ISAACSIM_DIR, output_dir=output_dir,
                        debug_config_log=debug_config_log, log_mirror=log_mirror,
                        extra_args=full_scene_compose_extra_args,
                    )
            else:
                logger.info(f"Stage 9: skipped (--start-from-stage {args.start_from_stage})")
            check_stop_after_stage(9)

        # ── Stage 10: chunked GLB bake ──
        if RUN_GLB and args.start_from_stage <= 10:
            with stage(f"Stage 10: chunked_to_glb.py (simplify_threshold={SIMPLIFY_THRESHOLD}, texture_size={TEXTURE_SIZE})"):
                stages.stage10_chunked_to_glb(
                    output_dir, simplify_threshold=SIMPLIFY_THRESHOLD, texture_size=TEXTURE_SIZE
                )
            logger.info(f"Done: {output_dir}/scene.glb")
        elif RUN_GLB:
            logger.info(f"Stage 10: skipped (--start-from-stage {args.start_from_stage})")
            logger.info(f"Done: {output_dir}/mesh.ply")
        else:
            logger.info("GLB bake disabled, skipping.")
            logger.info(f"Done: {output_dir}/mesh.ply")

        # ── Stage 11: organize final per-object deliverables ──
        final_objects_dir = run_dir / "final_objects"
        if classes and args.start_from_stage <= 11:
            with stage(f"Stage 11: organizing final objects -> {final_objects_dir}"):
                stages.stage11_organize_final_objects(run_dir, shapes_dir)
        else:
            logger.info(f"Stage 11: skipped (no --classes, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(11)

        # ── Stage 12: Isaac Sim robot-collision demo ──
        robot_collision_dir = run_dir / "robot_collision"
        if args.run_usd and args.robot_target and args.start_from_stage <= 12:
            sanitized_robot_target = stages.sanitize_label(args.robot_target)
            with stage(f"Stage 12: demo_robot_collide.py (robot_target={sanitized_robot_target}) -> {robot_collision_dir}/robot_collide.mp4"):
                stages.stage12_robot_collide(
                    shapes_dir, sanitized_robot_target, robot_collision_dir,
                    isaacsim_dir=ISAACSIM_DIR, debug_config_log=debug_config_log, log_mirror=log_mirror,
                )
        else:
            logger.info(f"Stage 12: skipped (no --robot-target, --skip_isaac, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(12)

        elapsed = time.monotonic() - pipeline_t0
        logger.info(f"Pipeline finished for scene '{args.scene_name}' (total elapsed {_format_duration(elapsed)})")
    except SystemExit:
        raise
    except Exception:
        logger.exception(f"Pipeline failed for scene '{args.scene_name}'")
        sys.exit(1)
    finally:
        if video_tmp_dir is not None:
            shutil.rmtree(video_tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
