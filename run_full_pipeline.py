#!/usr/bin/env python3
"""End-to-end pipeline: VGGT-Omega pose/depth prediction -> [optional GroundedSAM2 3D
segmentation] -> GenRecon reconstruction -> GLB bake.

Python port of run_full_pipeline.sh: same CLI surface, but every non-Isaac, non-TRELLIS.2 stage
runs in-process as a typed Python function call instead of a subprocess with a hand-built argv
string (see genrecon/pipeline/stages.py). Isaac Sim (../IsaacSim) and TRELLIS.2 (../trellis2,
only reachable via --mesh-backend trellis2) remain subprocesses, since both still run in their
own separate uv env.

Usage:
    uv run python run_full_pipeline.py <image_folder> <scene_name> [options...]

Example:
    uv run python run_full_pipeline.py /path/to/images food2_vggt --classes "banana,bowl" \\
        --pick_place_target banana --place-target bowl
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

GENRECON_DIR = Path(__file__).resolve().parent
TRELLIS2_DIR = GENRECON_DIR.parent / "trellis2"
ISAACSIM_DIR = GENRECON_DIR.parent / "IsaacSim"
MVSAM3D_VENDOR_DIR = GENRECON_DIR / "mv_sam3d"
VGGT_CHECKPOINT = GENRECON_DIR / "checkpoints" / "vggt_omega" / "ckpts" / "vggt_omega_1b_512.pt"


def _place_offset(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--place-offset must be DX,DY,DZ (got {s!r})")
    return (parts[0], parts[1], parts[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_folder", type=Path)
    parser.add_argument("scene_name", type=str)

    parser.add_argument("--simplify_threshold", type=int, default=250_000)
    parser.add_argument("--texture_size", type=int, default=2048)
    parser.add_argument("--num_imgs_per_scene", type=int, default=32)
    parser.add_argument("--skip-frames", dest="skip_frames", type=int, default=-1)
    parser.add_argument(
        "--no-align-to-gravity", dest="align_to_gravity", action="store_false", default=True,
        help="Gravity alignment is ON by default; pass this to disable it.",
    )
    parser.add_argument("--rotate-horizontal-deg", type=float, default=0.0)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--run_glb", action="store_true", default=False)
    parser.add_argument(
        "--use-trellis", dest="use_trellis", action=argparse.BooleanOptionalAction, default=True,
        help="Per-class mesh reconstruction (TRELLIS.2/MV-SAM3D). No-op without --classes.",
    )
    parser.add_argument("--mesh-backend", dest="mesh_backend", choices=["trellis2", "mvsam3d"], default="mvsam3d")
    parser.add_argument("--mvsam3d-stage1-steps", dest="mvsam3d_stage1_steps", type=int, default=25,
                        help="MV-SAM3D stage 1 (shape) inference steps. The upstream MV-SAM3D default is 50.")
    parser.add_argument("--mvsam3d-stage2-steps", dest="mvsam3d_stage2_steps", type=int, default=12,
                        help="MV-SAM3D stage 2 (texture) inference steps. The upstream MV-SAM3D default is 25.")
    parser.add_argument("--mvsam3d-top-k-views", dest="mvsam3d_top_k_views", type=int, default=5,
                        help="Prune to the k views that best cover each object angularly before "
                             "MV-SAM3D's main diffusion pass. Pass 0 or a value >= the scene's "
                             "view count to disable pruning (use every view with a mask).")
    parser.add_argument("--skip_isaac", dest="run_usd", action="store_false", default=True)
    parser.add_argument("--collision_approximation", default="convexDecomposition")
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
    parser.add_argument("--skip_floater_removal", dest="run_floater_removal", action="store_false", default=True)
    parser.add_argument("--floater_search_padding_factor", type=float, default=0.2)
    parser.add_argument("--floater_containment_frac", type=float, default=0.95)
    parser.add_argument("--floater_max_faces", type=int, default=5000)
    # Lowered from demo_rerun.py's own CLI default (50.0, a percentile filter that discards half
    # of every exported point cloud by construction) -- see vggt/demo_rerun.py's --conf-thres help.
    parser.add_argument("--vggt-conf-thres", dest="vggt_conf_thres", type=float, default=20.0)
    parser.add_argument(
        "--vggt-viewer", dest="vggt_viewer", action="store_true", default=False,
        help="Spawn the rerun point-cloud viewer after Stage 0's export (background thread). Off "
        "by default -- nothing downstream reads from it, and it adds nothing to a headless run.",
    )
    parser.add_argument("--skip_hull_consistency_check", action="store_true", default=False)
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


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace, logger) -> list[str]:
    """Ports run_full_pipeline.sh's cross-field validation (lines ~168-259). Mutates
    args.use_trellis in place where bash silently overrides it. Returns the parsed --classes
    list. Calls parser.error() (exit 2) on invalid combinations, matching bash's exit-1-with-
    message behavior closely enough for a CLI tool."""
    if not args.image_folder.is_dir():
        parser.error(f"Image folder not found: {args.image_folder}")

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else []

    if args.use_trellis and not classes:
        logger.info(
            "Note: --classes not set, so per-class mesh reconstruction (enabled by default) "
            "does not apply to this run."
        )
        args.use_trellis = False

    if args.mesh_backend == "mvsam3d" and not MVSAM3D_VENDOR_DIR.is_dir():
        parser.error(f"--mesh-backend mvsam3d requires the vendored MV-SAM3D source at {MVSAM3D_VENDOR_DIR}.")

    if args.pick_place_target:
        if args.robot_target:
            parser.error("--pick_place_target and --robot-target are mutually exclusive.")
        if not classes:
            parser.error("--pick_place_target requires --classes to include the same label.")
        if args.pick_place_target not in classes:
            parser.error(f"--pick_place_target '{args.pick_place_target}' must exactly match one of the labels passed to --classes ('{args.classes}').")
        if not args.use_trellis:
            logger.info("Note: --pick_place_target requires TRELLIS.2's per-object mesh; overriding --no-use-trellis to on for this run.")
            args.use_trellis = True

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
        if not classes:
            parser.error("--ai-scene-agent requires --classes (the objects it asks the user about).")
        if not args.use_trellis:
            logger.info("Note: --ai-scene-agent requires a per-object mesh for every --classes label; overriding --no-use-trellis to on for this run.")
            args.use_trellis = True

    return classes


def _format_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}m{s % 60:02d}s"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_dir = GENRECON_DIR / "runs" / args.scene_name
    export_dir = run_dir / "vggt_export"
    scene_dir = run_dir / "genrecon_input"
    output_dir = run_dir / "genrecon_output"
    export_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_log = run_dir / "pipeline.log"
    debug_config_log = run_dir / "debug_config.log"
    # Must happen before genrecon.utils.logger (or anything that imports it, including
    # genrecon.pipeline.stages) is first imported anywhere in the process -- loguru's file sink
    # is added once, at that first import, based on this env var.
    os.environ["GENRECON_PIPELINE_LOG"] = str(pipeline_log)

    from genrecon.pipeline import stages
    from genrecon.pipeline.subprocess_utils import LogMirror, stage
    from genrecon.utils.logger import logger

    classes = validate_args(parser, args, logger)
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

    pipeline_t0 = time.monotonic()
    logger.info(f"Pipeline started for scene '{args.scene_name}'")

    try:
        # ── Stage 0: VGGT-Omega export ──
        if args.start_from_stage <= 0:
            with stage(f"Stage 0: VGGT-Omega export -> {export_dir}"):
                stages.stage0_vggt_export(
                    args.image_folder,
                    VGGT_CHECKPOINT,
                    export_dir,
                    skip_frames=args.skip_frames,
                    align_to_gravity=args.align_to_gravity,
                    rotate_horizontal_deg=args.rotate_horizontal_deg,
                    conf_thres=args.vggt_conf_thres,
                    spawn_viewer=args.vggt_viewer,
                )
        else:
            logger.info(f"Stage 0: skipped (--start-from-stage {args.start_from_stage}), assuming existing export at {export_dir}")
        check_stop_after_stage(0)

        cobgs_mask_dir = output_dir / "segmentation_raw" / "masks" / "classes"
        seg_log = output_dir / "segmentation.log"
        image_to_3d_output_dir = run_dir / "image_to_3d_meshes"

        # ── Stage 1-3: segmentation + per-class 3D reconstruction (only if --classes) ──
        if classes:
            if args.start_from_stage <= 1:
                with stage(f"Stage 1: segmentation (classes: {args.classes})"):
                    cobgs_mask_dir = stages.stage1_segmentation(
                        args.scene_name,
                        export_dir,
                        output_dir / "segmentation_raw",
                        classes,
                        depth_conf_thres=args.depth_conf_thres,
                        depth_edge_rtol=args.depth_edge_rtol,
                        skip_hull_consistency_check=args.skip_hull_consistency_check,
                        seg_log=seg_log,
                        log_mirror=log_mirror,
                    )
            else:
                logger.info(f"Stage 1: skipped (--start-from-stage {args.start_from_stage}), assuming existing segmentation at {output_dir / 'segmentation_raw'}")
            check_stop_after_stage(1)

            if args.start_from_stage <= 2:
                with stage(f"Stage 2: RGBA mask export -> {cobgs_mask_dir}/<class>/mask_rgba"):
                    stages.stage2_export_rgba_masks(export_dir / "images", cobgs_mask_dir)
            else:
                logger.info(f"Stage 2: skipped (--start-from-stage {args.start_from_stage})")
            check_stop_after_stage(2)

            if args.use_trellis and args.start_from_stage <= 3:
                if args.mesh_backend == "trellis2":
                    with stage(f"Stage 3: TRELLIS.2 reconstruction -> {image_to_3d_output_dir}"):
                        stages.stage3_trellis2(
                            cobgs_mask_dir,
                            run_dir / "trellis2_input",
                            image_to_3d_output_dir,
                            trellis2_dir=TRELLIS2_DIR,
                            seg_log=seg_log,
                            log_mirror=log_mirror,
                        )
                else:
                    with stage(f"Stage 3: MV-SAM3D reconstruction -> {image_to_3d_output_dir}"):
                        stages.stage3_mvsam3d(
                            run_dir,
                            args.scene_name,
                            classes,
                            image_to_3d_output_dir,
                            mvsam3d_vendor_dir=MVSAM3D_VENDOR_DIR,
                            seg_log=seg_log,
                            log_mirror=log_mirror,
                            cobgs_mask_dir=cobgs_mask_dir,
                            stage1_steps=args.mvsam3d_stage1_steps,
                            stage2_steps=args.mvsam3d_stage2_steps,
                            top_k_views=args.mvsam3d_top_k_views,
                        )
            else:
                logger.info(f"Stage 3: skipped (no --use-trellis, or --start-from-stage {args.start_from_stage})")
            check_stop_after_stage(3)

        # ── Stage P0 (only with --ai-scene-agent) ──
        pick_place_target = args.pick_place_target
        place_target = args.place_target
        place_offset = args.place_offset
        place_target_clearance = args.place_target_clearance
        gripper_open_width = args.gripper_open_width
        approach_side = args.approach_side
        scene_agent_extra_args: list[str] = []

        if args.ai_scene_agent:
            with stage(f"Stage P0: interactive scene agent (model={args.scene_agent_ollama_model})"):
                result = stages.stageP0_scene_agent(
                    classes,
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

        # ── Stage P1-P4 (fast path, mutually exclusive with Stage 4-15) ──
        if pick_place_target:
            pick_place_dir = run_dir / "pick_place"
            pick_place_glb_dir = pick_place_dir / "glb"
            pick_place_scene_usda = pick_place_glb_dir / "scene.usda"

            with stage(f"Stage P1: aligning meshes for all --classes labels to scene scale -> {pick_place_glb_dir}"):
                stages.stageP1_align_meshes(
                    run_dir, classes, cobgs_mask_dir, pick_place_glb_dir, mesh_backend=args.mesh_backend
                )

            with stage(f"Stage P2: convert_asset.py (collision_approximation={args.collision_approximation}) -> {pick_place_glb_dir}/<label>/asset.usd"):
                stages.stageP2_convert_asset(
                    pick_place_glb_dir,
                    isaacsim_dir=ISAACSIM_DIR,
                    collision_approximation=args.collision_approximation,
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

        # ── Stage 4: stage GenRecon scene dir ──
        if args.start_from_stage <= 4:
            with stage(f"Stage 4: staging {scene_dir}"):
                stages.stage4_stage_scene_dir(export_dir, scene_dir)
        else:
            logger.info(f"Stage 4: skipped (--start-from-stage {args.start_from_stage}), assuming existing {scene_dir}")
        check_stop_after_stage(4)

        # ── Stage 5: GenRecon reconstruction ──
        if args.start_from_stage <= 5:
            with stage("Stage 5: reconstruct_scene.py"):
                stages.stage5_reconstruct_scene(
                    scene_dir,
                    output_dir,
                    classes,
                    cobgs_mask_dir if classes else None,
                    num_imgs_per_scene=args.num_imgs_per_scene,
                    max_chunks_per_group=args.max_chunks_per_group,
                    max_inflated_voxels=args.max_inflated_voxels,
                    fix_num_chunks=args.fix_num_chunks,
                )
        else:
            logger.info(f"Stage 5: skipped (--start-from-stage {args.start_from_stage})")
        check_stop_after_stage(5)

        # ── Stage 6: reprojection validation ──
        if args.start_from_stage <= 6:
            with stage("Stage 6: render_reprojection_validation.py"):
                stages.stage6_reprojection_validation(output_dir, scene_dir, classes)
        else:
            logger.info(f"Stage 6: skipped (--start-from-stage {args.start_from_stage})")
        check_stop_after_stage(6)

        # ── Stage 7: collect shapes ──
        if args.start_from_stage <= 7:
            with stage(f"Stage 7: collecting shapes -> {output_dir}/shapes"):
                shapes_dir = stages.stage7_collect_shapes(output_dir, classes, cobgs_mask_dir if classes else None)
        else:
            shapes_dir = output_dir / "shapes"
            logger.info(f"Stage 7: skipped (--start-from-stage {args.start_from_stage}), assuming existing {shapes_dir}")
        check_stop_after_stage(7)

        # ── Stage 8: per-object mesh extraction ──
        if classes and args.start_from_stage <= 8:
            with stage(f"Stage 8: cascading per-object mesh extraction -> {shapes_dir}"):
                stages.stage8_extract_object_meshes(
                    shapes_dir,
                    scene_dir,
                    cobgs_mask_dir,
                    classes,
                    output_dir,
                    run_floater_removal=args.run_floater_removal,
                    floater_search_padding_factor=args.floater_search_padding_factor,
                    floater_containment_frac=args.floater_containment_frac,
                    floater_max_faces=args.floater_max_faces,
                )
        else:
            logger.info(f"Stage 8: skipped (no --classes, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(8)

        # ── Stage 9: floor segmentation + friction inference ──
        if classes and args.start_from_stage <= 9:
            with stage(f"Stage 9: extract_floor_mesh.py + infer_friction_assignments.py -> {shapes_dir}"):
                stages.stage9_floor_and_friction(
                    shapes_dir, cobgs_mask_dir,
                    friction_table_path=args.friction_table_path,
                    ollama_model=args.ollama_model,
                )
        else:
            logger.info(f"Stage 9: skipped (no --classes, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(9)

        # ── Stage 10/11/12: mesh -> GLB -> USD -> composed scene ──
        if args.run_usd:
            if args.start_from_stage <= 10:
                with stage(f"Stage 10: mesh_to_glb.py -> {shapes_dir}/glb"):
                    stages.stage10_mesh_to_glb(
                        shapes_dir, run_dir, classes, use_trellis=args.use_trellis, mesh_backend=args.mesh_backend
                    )
            else:
                logger.info(f"Stage 10: skipped (--start-from-stage {args.start_from_stage})")
            check_stop_after_stage(10)

            if args.start_from_stage <= 11:
                with stage(f"Stage 11: convert_asset.py (collision_approximation={args.collision_approximation}) -> {shapes_dir}/glb/<label>/asset.usd"):
                    stages.stage11_convert_asset(
                        shapes_dir, classes,
                        isaacsim_dir=ISAACSIM_DIR,
                        collision_approximation=args.collision_approximation,
                        output_dir=output_dir,
                        debug_config_log=debug_config_log,
                        log_mirror=log_mirror,
                    )
            else:
                logger.info(f"Stage 11: skipped (--start-from-stage {args.start_from_stage})")
            check_stop_after_stage(11)

            if args.start_from_stage <= 12:
                with stage(f"Stage 12: compose_isaac_scene.py -> {shapes_dir}/glb/scene.usda"):
                    stages.stage12_compose_isaac_scene(
                        shapes_dir, isaacsim_dir=ISAACSIM_DIR, output_dir=output_dir,
                        debug_config_log=debug_config_log, log_mirror=log_mirror,
                        extra_args=compose_scene_extra_args,
                    )
            else:
                logger.info(f"Stage 12: skipped (--start-from-stage {args.start_from_stage})")
            check_stop_after_stage(12)

        # ── Stage 13: chunked GLB bake ──
        if args.run_glb and args.start_from_stage <= 13:
            with stage(f"Stage 13: chunked_to_glb.py (simplify_threshold={args.simplify_threshold}, texture_size={args.texture_size})"):
                stages.stage13_chunked_to_glb(
                    output_dir, simplify_threshold=args.simplify_threshold, texture_size=args.texture_size
                )
            logger.info(f"Done: {output_dir}/scene.glb")
        elif args.run_glb:
            logger.info(f"Stage 13: skipped (--start-from-stage {args.start_from_stage})")
            logger.info(f"Done: {shapes_dir}/mesh.ply")
        else:
            logger.info("--run_glb not set, skipping GLB bake.")
            logger.info(f"Done: {shapes_dir}/mesh.ply")

        # ── Stage 14: organize final per-object deliverables ──
        final_objects_dir = run_dir / "final_objects"
        if classes and args.start_from_stage <= 14:
            with stage(f"Stage 14: organizing final objects -> {final_objects_dir}"):
                stages.stage14_organize_final_objects(run_dir, output_dir, shapes_dir)
        else:
            logger.info(f"Stage 14: skipped (no --classes, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(14)

        # ── Stage 15: Isaac Sim robot-collision demo ──
        robot_collision_dir = run_dir / "robot_collision"
        if args.run_usd and args.robot_target and args.start_from_stage <= 15:
            sanitized_robot_target = stages.sanitize_label(args.robot_target)
            with stage(f"Stage 15: demo_robot_collide.py (robot_target={sanitized_robot_target}) -> {robot_collision_dir}/robot_collide.mp4"):
                stages.stage15_robot_collide(
                    shapes_dir, sanitized_robot_target, robot_collision_dir,
                    isaacsim_dir=ISAACSIM_DIR, debug_config_log=debug_config_log, log_mirror=log_mirror,
                )
        else:
            logger.info(f"Stage 15: skipped (no --robot-target, --skip_isaac, or --start-from-stage {args.start_from_stage})")
        check_stop_after_stage(15)

        elapsed = time.monotonic() - pipeline_t0
        logger.info(f"Pipeline finished for scene '{args.scene_name}' (total elapsed {_format_duration(elapsed)})")
    except SystemExit:
        raise
    except Exception:
        logger.exception(f"Pipeline failed for scene '{args.scene_name}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
