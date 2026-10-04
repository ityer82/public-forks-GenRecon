"""One function per run_full_pipeline.sh "Stage N" (plus Stage P0-P4, the pick-and-place fast
path). Every in-repo (non-Isaac, non-TRELLIS.2) stage is called in-process, as a typed Python
function -- see the plan at the top of run_full_pipeline.py for the full rationale.

Hard constraint (do not violate when editing): stages communicate only via files under
RUN_DIR, never via in-memory objects passed between stage functions -- this is what makes
--start-from-stage/--stop-after-stage meaningful. Every stage function re-reads its inputs from
disk and writes its outputs back to disk, even when called back-to-back with the stage before it.

Heavy/CUDA-bearing imports (torch, vggt_omega, mv_sam3d's inference stack, pytorch3d, ...)
happen inside each stage function body, not at module level -- both so GENRECON_PIPELINE_LOG can
be set before genrecon.utils.logger is first imported anywhere in the process, and so a run that
skips a stage (--start-from-stage) never pays the cost of importing its heavy dependencies.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from genrecon.pipeline.subprocess_utils import LogMirror, redirect_fd_to_file, run_external_step
from genrecon.utils.logger import logger

GENRECON_DIR = Path(__file__).resolve().parent.parent.parent


def _ensure_on_path(directory: Path) -> None:
    d = str(directory)
    if d not in sys.path:
        sys.path.insert(0, d)


def _split_classes(classes_csv: str | None) -> list[str]:
    if not classes_csv:
        return []
    return [c.strip() for c in classes_csv.split(",") if c.strip()]


def sanitize_label(label: str) -> str:
    """Mirrors labels.json's sanitize_label (spaces/slashes -> underscores), used wherever a
    scene-side artifact (shapes/, glb/) is keyed by the sanitized dirname instead of the raw
    --classes spelling (stage_2_mv_sam3d/, stage_2_mv_sam3d/input/) -- including the USD prim names
    compose_isaac_scene.py derives from those dirnames, which is why run_full_pipeline.py also
    applies this to --pick-target/--place-target/--robot-target before invoking Isaac Sim: those
    prims are always named from the sanitized dirname, never the raw (possibly multi-word)
    --classes spelling."""
    return label.replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Stage 0: VGGT-Omega pose/depth prediction + COLMAP-text export
# ---------------------------------------------------------------------------


def stage0_vggt_export(
    image_folder: Path,
    checkpoint: Path,
    export_dir: Path,
    *,
    skip_frames: int,
    max_frames: int = -1,
    align_to_gravity: bool,
    rotate_horizontal_deg: float,
    conf_thres: float,
    spawn_viewer: bool = False,
) -> None:
    _ensure_on_path(GENRECON_DIR / "vggt")
    import torch
    from demo_rerun import apply_gravity_alignment, export_colmap_dataset, load_model, log_to_rerun, run_model

    logger.info(f"Loading VGGT-Omega checkpoint from {checkpoint}")
    model = load_model(str(checkpoint))
    try:
        predictions = run_model(
            str(image_folder),
            model,
            image_resolution=512,
            skip_frames=skip_frames,
            max_frames=max_frames,
        )
        predictions = apply_gravity_alignment(predictions, align_to_gravity, rotate_horizontal_deg)
        export_colmap_dataset(
            predictions,
            str(export_dir),
            conf_thres=conf_thres,
            max_points=1_000_000,
            save_depth=True,
        )
        if spawn_viewer:
            import threading

            import rerun as rr

            def _spawn():
                rr.init("vggt-omega", spawn=True)
                log_to_rerun(
                    predictions,
                    conf_thres=conf_thres,
                    mask_black_bg=False,
                    mask_white_bg=False,
                    show_cam=True,
                    mask_sky=False,
                    image_folder=str(image_folder),
                    max_points=1_000_000,
                )

            threading.Thread(target=_spawn, daemon=True).start()
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _finalize_stage0_layout(export_dir)


def _finalize_stage0_layout(export_dir: Path) -> None:
    # Rename to the layout GenRecon's image loader expects (<dir>/rgb next to <dir>/colmap).
    (export_dir / "images").rename(export_dir / "rgb")
    (export_dir / "sparse" / "0").rename(export_dir / "colmap")
    (export_dir / "sparse").rmdir()

    cameras_txt = export_dir / "colmap" / "cameras.txt"
    if not cameras_txt.exists():
        raise FileNotFoundError(f"Expected COLMAP export not found at {cameras_txt}")


def stage0_stereo_export(
    image_folder: Path,
    checkpoint: Path,
    conf_path: Path,
    export_dir: Path,
    *,
    align_to_gravity: bool,
    rotate_horizontal_deg: float,
    conf_thres: float,
) -> None:
    """Stage 0 alternative for a rectified stereo pair: FoundationStereo depth for both views,
    written in exactly the layout stage0_vggt_export produces (rgb/, depth/, colmap/)."""
    _ensure_on_path(GENRECON_DIR / "vggt")  # demo_rerun's gravity alignment + COLMAP export
    _ensure_on_path(GENRECON_DIR / "foundation_stereo")
    import torch
    from demo_rerun import apply_gravity_alignment, export_colmap_dataset
    from stereo_export import find_stereo_pair, load_model, run_stereo

    left_path, right_path = find_stereo_pair(image_folder)
    logger.info(f"Loading FoundationStereo checkpoint from {checkpoint}")
    model = load_model(str(checkpoint))
    try:
        predictions = run_stereo(left_path, right_path, model, conf_path)
        predictions = apply_gravity_alignment(predictions, align_to_gravity, rotate_horizontal_deg)
        export_colmap_dataset(
            predictions,
            str(export_dir),
            conf_thres=conf_thres,
            max_points=1_000_000,
            save_depth=True,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _finalize_stage0_layout(export_dir)


# ---------------------------------------------------------------------------
# Stage 1: GroundedSAM2-based 3D segmentation (optional, only if --classes)
# ---------------------------------------------------------------------------


def stage1_segmentation(
    scene_name: str,
    export_dir: Path,
    output_root: Path,
    classes: list[str],
    *,
    depth_conf_thres: float,
    depth_edge_rtol: float,
    seg_log: Path,
    log_mirror: LogMirror,
    detection_box_padding_frac: float = 0.05,
    boxes_json: Path | None = None,
) -> Path:
    """Returns the COB-GS mask directory (output_root itself: labels.json + one dir per class)."""
    _ensure_on_path(GENRECON_DIR / "segmentation")
    from main_light import run_segmentation

    with redirect_fd_to_file(seg_log, append=False):
        run_segmentation(
            scene_name,
            str(export_dir),
            str(output_root),
            ",".join(classes),
            text="classes",
            resolution=-1,
            depth_dir=str(export_dir / "depth"),
            depth_conf_thres=depth_conf_thres,
            depth_edge_rtol=depth_edge_rtol,
            flat_output=True,
            detection_box_padding_frac=detection_box_padding_frac,
            boxes_json=str(boxes_json) if boxes_json else None,
        )
    log_mirror.mirror(seg_log)

    cobgs_mask_dir = output_root
    labels_json = cobgs_mask_dir / "labels.json"
    if not labels_json.exists():
        raise FileNotFoundError(f"Expected segmentation labels.json not found at {labels_json}")
    return cobgs_mask_dir


# ---------------------------------------------------------------------------
# Stage 2: per-class 3D reconstruction (MV-SAM3D backend)
# ---------------------------------------------------------------------------


def stage2_mvsam3d(
    run_dir: Path,
    scene_name: str,
    classes: list[str],
    image_to_3d_output_dir: Path,
    *,
    mvsam3d_vendor_dir: Path,
    log_file: Path,
    log_mirror: LogMirror,
    cobgs_mask_dir: Path | None = None,
    stage1_steps: int = 25,
    stage2_steps: int = 12,
    top_k_views: int | None = 5,
    align_shape_latents: bool = False,
) -> None:
    _ensure_on_path(mvsam3d_vendor_dir / "mvsam3d_scripts")
    from collect_mvsam3d_outputs import collect_mvsam3d_outputs
    from import_from_genrecon import import_from_genrecon

    mvsam3d_input_dir = image_to_3d_output_dir / "input"
    # Must match run_inference_weighted._dataset_name_for(), which keys visualization/ by scene.
    mvsam3d_dataset_name = f"{scene_name}_mvsam3d_input"

    with redirect_fd_to_file(log_file, append=False):
        import_from_genrecon(run_dir, mvsam3d_input_dir, classes)
    log_mirror.mirror(log_file)

    _ensure_on_path(mvsam3d_vendor_dir)
    from run_inference_weighted import run_multiobject_inference, run_weighted_inference

    da3_output = mvsam3d_input_dir / "da3_output.npz"
    if len(classes) > 1:
        run_multiobject_inference(
            input_path=mvsam3d_input_dir,
            mask_prompts=classes,
            da3_output_path=str(da3_output),
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            top_k_views=top_k_views,
            view_selection_pointcloud_dir=cobgs_mask_dir,
            align_shape_latents=align_shape_latents,
        )
    else:
        run_weighted_inference(
            input_path=mvsam3d_input_dir,
            mask_prompt=classes[0],
            da3_output_path=str(da3_output),
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            top_k_views=top_k_views,
            view_selection_pointcloud_dir=cobgs_mask_dir,
            align_shape_latents=align_shape_latents,
        )

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with redirect_fd_to_file(log_file, append=True):
        failures = collect_mvsam3d_outputs(
            mvsam3d_vendor_dir / "visualization",
            mvsam3d_dataset_name,
            classes,
            image_to_3d_output_dir,
            scene_pointcloud_dir=cobgs_mask_dir,
            mvsam3d_input_dir=mvsam3d_input_dir,
        )
    log_mirror.mirror(log_file)
    if failures:
        raise RuntimeError(f"MV-SAM3D collection failed for labels: {failures}")


# ---------------------------------------------------------------------------
# Stage P0: interactive scene agent (only with --ai-scene-agent)
# ---------------------------------------------------------------------------


def classes_with_meshes(classes: list[str], image_to_3d_output_dir: Path) -> list[str]:
    """Subset of `classes` that actually have a generated mesh.glb under
    image_to_3d_output_dir, keyed by the raw (unsanitized) label -- the same path
    stageP1_align_meshes and scene_agent.py._describe_objects already read from."""
    return [c for c in classes if (image_to_3d_output_dir / c / "mesh.glb").exists()]


def discover_classes_from_mesh_dir(image_to_3d_output_dir: Path) -> list[str]:
    """Classes list derived straight from existing meshes on disk -- used by --ai-scene-agent
    runs started mid-pipeline (--start-from-stage > 3) without --classes, where there's no
    --classes list to filter in the first place. Every subdirectory of image_to_3d_output_dir
    containing a mesh.glb is treated as a class (raw, unsanitized label), sorted for determinism."""
    if not image_to_3d_output_dir.is_dir():
        return []
    return sorted(p.name for p in image_to_3d_output_dir.iterdir() if p.is_dir() and (p / "mesh.glb").exists())


def stageP0_scene_agent(
    classes: list[str],
    mesh_dir: Path,
    scene_spec_json: Path,
    *,
    ollama_model: str,
    ollama_host: str = "http://localhost:11434",
    start_distance: float = 0.5,
    gripper_open_width: float = 0.06,
) -> dict:
    from genrecon.utils.scene_agent import run_scene_agent

    result = run_scene_agent(
        classes,
        mesh_dir,
        ollama_model=ollama_model,
        ollama_host=ollama_host,
        start_distance=start_distance,
        gripper_open_width=gripper_open_width,
    )

    spec = {
        "pick_target": result["pick_target"],
        "place_target": result["place_target"],
        "place_offset": result["place_offset"],
        "place_target_clearance": result["place_target_clearance"],
        "approach_side": result["approach_side"],
        "start_distance": result["start_distance"],
        "gripper_open_width": result["gripper_open_width"],
        "lighting": result["lighting"],
        "camera": result["camera"],
        "reasoning": result["reasoning"],
    }
    scene_spec_json.parent.mkdir(parents=True, exist_ok=True)
    scene_spec_json.write_text(json.dumps(spec, indent=2))
    place_target_desc = result["place_target"] or f"<offset {result['place_offset']}>"
    logger.info(
        f"Stage P0: agent chose pick_place_target={result['pick_target']}, "
        f"place_target={place_target_desc}, approach_side={result['approach_side']}"
    )
    return result


# ---------------------------------------------------------------------------
# Stage P1: align MV-SAM3D meshes to scene scale for every --classes label
# ---------------------------------------------------------------------------


def stageP1_align_meshes(
    run_dir: Path,
    classes: list[str],
    cobgs_mask_dir: Path,
    pick_place_glb_dir: Path,
) -> None:
    _ensure_on_path(GENRECON_DIR / "scripts")
    from align_meshes_to_scene import align_trellis_mesh_to_scene

    apply_zup_correction = False
    for label in classes:
        sanitized_label = sanitize_label(label)
        out_dir = pick_place_glb_dir / sanitized_label
        out_dir.mkdir(parents=True, exist_ok=True)

        trellis_glb = run_dir / "stage_2_mv_sam3d" / label / "mesh.glb"
        scale_ref_ply = cobgs_mask_dir / sanitized_label / "point_cloud" / f"{sanitized_label}.ply"
        if not trellis_glb.exists():
            raise FileNotFoundError(f"Expected TRELLIS.2 mesh not found at {trellis_glb}")
        if not scale_ref_ply.exists():
            raise FileNotFoundError(f"Expected segmented point cloud not found at {scale_ref_ply}")

        scene, _transform, diagnostics = align_trellis_mesh_to_scene(
            trellis_glb, scale_ref_ply, apply_zup_correction=apply_zup_correction
        )
        out_glb = out_dir / "mesh.glb"
        scene.export(out_glb)
        logger.info(f"stageP1: wrote {out_glb} (scale={diagnostics['scale']:.4f})")


# ---------------------------------------------------------------------------
# Stage P2-P4: pick-and-place fast path (Isaac subprocess stages)
# ---------------------------------------------------------------------------


def stageP2_convert_asset(
    pick_place_glb_dir: Path,
    *,
    isaacsim_dir: Path,
    collision_approximation: str,
    log_file: Path,
    debug_config_log: Path,
    log_mirror: LogMirror,
) -> None:
    run_external_step(
        "stageP2_convert_asset",
        isaacsim_dir / "convert_asset.py",
        isaacsim_dir,
        ["--input", str(pick_place_glb_dir), "--collision-approximation", collision_approximation],
        log_file=log_file,
        debug_config_log=debug_config_log,
        log_mirror=log_mirror,
    )


def stageP3_compose_isaac_scene(
    pick_place_glb_dir: Path,
    scene_usda: Path,
    *,
    isaacsim_dir: Path,
    log_file: Path,
    debug_config_log: Path,
    log_mirror: LogMirror,
    extra_args: list[str],
) -> None:
    run_external_step(
        "stageP3_compose_isaac_scene",
        isaacsim_dir / "compose_isaac_scene.py",
        isaacsim_dir,
        ["--input", str(pick_place_glb_dir), "--output", str(scene_usda), *extra_args],
        log_file=log_file,
        debug_config_log=debug_config_log,
        log_mirror=log_mirror,
    )


def stageP4_franka_pickplace(
    scene_usda: Path,
    pick_place_dir: Path,
    *,
    isaacsim_dir: Path,
    pick_target: str,
    place_target: str | None,
    place_offset: tuple[float, float, float],
    place_target_clearance: float,
    gripper_open_width: float,
    approach_side: str,
    scene_agent_extra_args: list[str],
    debug_config_log: Path,
    log_mirror: LogMirror,
) -> None:
    if place_target:
        place_args = ["--place-target", place_target, "--place-target-clearance", str(place_target_clearance)]
    else:
        place_args = ["--place-offset", *[str(v) for v in place_offset]]

    run_external_step(
        "stageP4_demo_franka_pickplace",
        isaacsim_dir / "demo_franka_pickplace.py",
        isaacsim_dir,
        [
            "--scene", str(scene_usda),
            "--pick-target", pick_target,
            *place_args,
            "--gripper-open-width", str(gripper_open_width),
            "--approach-side", approach_side,
            "--output", str(pick_place_dir / "pick_place.mp4"),
            "--stage-output", str(pick_place_dir / "pick_place_scene.usda"),
            *scene_agent_extra_args,
        ],
        log_file=pick_place_dir / "pick_place.log",
        debug_config_log=debug_config_log,
        log_mirror=log_mirror,
    )


# ---------------------------------------------------------------------------
# Stage 3: GenRecon reconstruction + GLB bake inputs (exp2 settings)
# ---------------------------------------------------------------------------


def stage3_reconstruct_scene(
    scene_dir: Path,
    output_dir: Path,
    classes: list[str],
    cobgs_mask_dir: Path | None,
    *,
    num_imgs_per_scene: int,
    max_chunks_per_group: int | None,
    max_inflated_voxels: int | None,
    fix_num_chunks: int | None,
) -> None:
    from reconstruct_scene import run_reconstruct_scene

    exclude_masks_root = cobgs_mask_dir if classes else None
    unmasked_path = scene_dir if classes else None

    run_reconstruct_scene(
        scene_dir,
        output_dir,
        ss_ckpt=GENRECON_DIR / "checkpoints/sparse_structure/ckpts/sparse_structure.pt",
        shape_ckpt=GENRECON_DIR / "checkpoints/shape_slat/ckpts/shape_slat.pt",
        tex_ckpt=GENRECON_DIR / "checkpoints/texture_slat/ckpts/texture_slat.pt",
        num_imgs_per_scene=num_imgs_per_scene,
        colmap_subdir="colmap",
        max_chunks_per_group=max_chunks_per_group,
        max_inflated_voxels=max_inflated_voxels,
        fix_num_chunks=fix_num_chunks,
        exclude_masks_root=exclude_masks_root,
        unmasked_path=unmasked_path,
    )


# ---------------------------------------------------------------------------
# Stage 4: reprojection validation
# ---------------------------------------------------------------------------


def stage4_reprojection_validation(output_dir: Path, scene_dir: Path, classes: list[str], validation_dir: Path) -> None:
    _ensure_on_path(GENRECON_DIR / "scripts")
    from render_reprojection_validation import run_reprojection_validation

    reprojection_mesh_ply = output_dir / ("object_source_mesh.ply" if classes else "mesh.ply")
    run_reprojection_validation(
        reprojection_mesh_ply,
        scene_dir / "colmap",
        scene_dir / "rgb",
        validation_dir / "synth_views",
        validation_dir / "compare_views",
    )


# ---------------------------------------------------------------------------
# Stage 5: per-object mesh extraction (cascading convex hull crop)
# ---------------------------------------------------------------------------


FLOATER_SEARCH_PADDING_FACTOR = 0.2
FLOATER_CONTAINMENT_FRAC = 0.95
FLOATER_MAX_FACES = 5000


def stage5_extract_object_meshes(
    shapes_dir: Path,
    scene_dir: Path,
    cobgs_mask_dir: Path,
    classes: list[str],
    output_dir: Path,
) -> None:
    _ensure_on_path(GENRECON_DIR / "scripts")
    from extract_object_mesh import extract_object_mesh
    from remove_floater_mesh import remove_floater_mesh

    shapes_dir.mkdir(parents=True, exist_ok=True)
    object_source_mesh = output_dir / "object_source_mesh.ply"
    background_mesh = shapes_dir / "background_mesh.ply"
    shutil.copyfile(output_dir / "mesh.ply", background_mesh)

    for obj_ply in sorted(cobgs_mask_dir.glob("*/point_cloud/*.ply")):
        if obj_ply.name == "background.ply":
            continue
        label = obj_ply.stem

        ok = extract_object_mesh(
            background_mesh,
            obj_ply,
            shapes_dir / f"{label}_mesh.ply",
            object_mesh_ply=object_source_mesh,
            remainder_out_ply=background_mesh,
            colmap_dir=scene_dir / "colmap",
            masks_dir=cobgs_mask_dir / label / "mask_bin",
        )
        if not ok:
            logger.warning(f"Stage 5: extract_object_mesh soft-failed for {label!r}, continuing.")

        remove_floater_mesh(
            background_mesh,
            obj_ply,
            background_mesh,
            floaters_out_ply=shapes_dir / f"{label}_floaters.ply",
            search_padding_factor=FLOATER_SEARCH_PADDING_FACTOR,
            containment_frac=FLOATER_CONTAINMENT_FRAC,
            max_floater_faces=FLOATER_MAX_FACES,
        )


# ---------------------------------------------------------------------------
# Stage 6: floor segmentation + friction inference
# ---------------------------------------------------------------------------


def stage6_floor_and_friction(
    shapes_dir: Path,
    cobgs_mask_dir: Path,
    *,
    friction_table_path: Path,
    ollama_model: str,
) -> None:
    _ensure_on_path(GENRECON_DIR / "scripts")
    from extract_floor_mesh import extract_floor_mesh
    from infer_friction_assignments import run_friction_assignments

    background_mesh = shapes_dir / "background_mesh.ply"
    extract_floor_mesh(background_mesh, shapes_dir / "floor_mesh.ply", remainder_out_ply=background_mesh)

    run_friction_assignments(
        cobgs_mask_dir / "labels.json",
        friction_table_path,
        shapes_dir / "friction_assignments.json",
        ollama_model=ollama_model,
    )


# ---------------------------------------------------------------------------
# Stage 7: per-object mesh -> GLB (+ MV-SAM3D substitution)
# ---------------------------------------------------------------------------


def stage7_mesh_to_glb(
    shapes_dir: Path,
    run_dir: Path,
    classes: list[str],
    *,
    use_trellis: bool,
) -> None:
    _ensure_on_path(GENRECON_DIR / "scripts")
    from align_meshes_to_scene import align_trellis_mesh_to_scene
    from mesh_to_glb import convert

    glb_dir = shapes_dir / "glb"
    convert(shapes_dir, glb_dir)

    if not use_trellis:
        return

    apply_zup_correction = False
    for label in classes:
        sanitized_label = sanitize_label(label)
        trellis_glb = run_dir / "stage_2_mv_sam3d" / label / "mesh.glb"
        scene_crop_ply = shapes_dir / f"{sanitized_label}_mesh.ply"
        if not trellis_glb.exists():
            logger.info(f"Stage 7: --use-trellis: no TRELLIS.2 mesh for '{label}' at {trellis_glb}, keeping scene-crop glb.")
            continue
        if not scene_crop_ply.exists():
            logger.info(f"Stage 7: --use-trellis: no scene crop for '{label}' at {scene_crop_ply}, keeping scene-crop glb.")
            continue

        scene, _transform, diagnostics = align_trellis_mesh_to_scene(
            trellis_glb, scene_crop_ply, apply_zup_correction=apply_zup_correction
        )
        out_glb = glb_dir / sanitized_label / "mesh.glb"
        out_glb.parent.mkdir(parents=True, exist_ok=True)
        scene.export(out_glb)
        logger.info(f"Stage 7: substituted {out_glb} with TRELLIS mesh (scale={diagnostics['scale']:.4f})")


# ---------------------------------------------------------------------------
# Stage 8/9: convert_asset.py -> compose_isaac_scene.py (Isaac subprocess stages)
# ---------------------------------------------------------------------------


def stage8_convert_asset(
    shapes_dir: Path,
    classes: list[str],
    *,
    isaacsim_dir: Path,
    collision_approximation: str,
    output_dir: Path,
    debug_config_log: Path,
    log_mirror: LogMirror,
) -> None:
    args = ["--input", str(shapes_dir / "glb"), "--collision-approximation", collision_approximation]
    friction_json = shapes_dir / "friction_assignments.json"
    if classes and friction_json.exists():
        args += ["--friction-table", str(friction_json), "--friction-combine-mode", "max"]

    run_external_step(
        "stage8_convert_asset",
        isaacsim_dir / "convert_asset.py",
        isaacsim_dir,
        args,
        log_file=output_dir / "convert_asset.log",
        debug_config_log=debug_config_log,
        log_mirror=log_mirror,
    )


def stage9_compose_isaac_scene(
    shapes_dir: Path,
    *,
    isaacsim_dir: Path,
    output_dir: Path,
    debug_config_log: Path,
    log_mirror: LogMirror,
    extra_args: list[str],
) -> None:
    run_external_step(
        "stage9_compose_isaac_scene",
        isaacsim_dir / "compose_isaac_scene.py",
        isaacsim_dir,
        ["--input", str(shapes_dir / "glb"), "--output", str(shapes_dir / "glb" / "scene.usda"), "--background-label", "background", *extra_args],
        log_file=output_dir / "compose_isaac_scene.log",
        debug_config_log=debug_config_log,
        log_mirror=log_mirror,
    )


# ---------------------------------------------------------------------------
# Stage 10: chunked GLB bake
# ---------------------------------------------------------------------------


def stage10_chunked_to_glb(output_dir: Path, *, simplify_threshold: int, texture_size: int) -> None:
    from chunked_to_glb import run_chunked_to_glb

    run_chunked_to_glb(
        output_dir / "to_glb_inputs.pt",
        output_dir / "chunk_inputs.pt",
        output_dir,
        simplify_threshold=simplify_threshold,
        texture_size=texture_size,
    )


# ---------------------------------------------------------------------------
# Stage 11: organize final per-object deliverables
# ---------------------------------------------------------------------------


def stage11_organize_final_objects(run_dir: Path, shapes_dir: Path) -> None:
    _ensure_on_path(GENRECON_DIR / "scripts")
    from organize_final_objects import organize_final_objects

    organize_final_objects(
        shapes_dir,
        run_dir / "stage_1_segmentation",
        run_dir / "stage_2_mv_sam3d" / "input",
        run_dir / "stage_2_mv_sam3d",
        run_dir / "final_objects",
    )


# ---------------------------------------------------------------------------
# Stage 12: Isaac Sim robot-collision demo (subprocess)
# ---------------------------------------------------------------------------


def stage12_robot_collide(
    shapes_dir: Path,
    robot_target: str,
    robot_collision_dir: Path,
    *,
    isaacsim_dir: Path,
    debug_config_log: Path,
    log_mirror: LogMirror,
) -> None:
    robot_collision_dir.mkdir(parents=True, exist_ok=True)
    run_external_step(
        "stage12_demo_robot_collide",
        isaacsim_dir / "demo_robot_collide.py",
        isaacsim_dir,
        [
            "--scene", str(shapes_dir / "glb" / "scene.usda"),
            "--robot-target", robot_target,
            "--output", str(robot_collision_dir / "robot_collide.mp4"),
            "--stage-output", str(robot_collision_dir / "robot_collide_scene.usda"),
        ],
        log_file=robot_collision_dir / "robot_collide.log",
        debug_config_log=debug_config_log,
        log_mirror=log_mirror,
    )
