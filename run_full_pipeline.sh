#!/usr/bin/env bash
# End-to-end pipeline: VGGT-Omega pose/depth prediction -> [optional COB-GS 3D segmentation] -> GenRecon reconstruction -> GLB bake.
#
# Usage:
#   ./run_full_pipeline.sh <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--no-align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb] [--use-trellis] [--skip_isaac] [--collision_approximation convexDecomposition|convexHull|boundingCube] [--start-from-stage N] [--stop-after-stage N] [--max_chunks_per_group N] [--max_inflated_voxels N] [--depth_conf_thres N] [--depth_edge_rtol N] [--fix_num_voxels N]
#
# Note: gravity alignment is ON by default; pass --no-align-to-gravity to disable it.
#
# Example:
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --run_glb
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --align-to-gravity --rotate-horizontal-deg 90
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "person,chair,bag"
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "person,chair,bag" --use-trellis

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENRECON_DIR="$SCRIPT_DIR"
VGGT_OMEGA_DIR="$(cd "$GENRECON_DIR/../vggt-omega" && pwd)"
COBGS_DIR="$(cd "$GENRECON_DIR/../COB-GS" && pwd)"
TRELLIS2_DIR="$(cd "$GENRECON_DIR/../trellis2" && pwd)"
ISAACSIM_DIR="$(cd "$GENRECON_DIR/../IsaacSim" && pwd)"
VGGT_CHECKPOINT="${VGGT_OMEGA_DIR}/vggt_omega_1b_512.pt"

# ── CUDA toolkit selection for git-dependency builds (e.g. nvdiffrec-render) ──
# uv's isolated build env compiles that package's native extension with nvcc,
# which must match the CUDA version the pinned torch wheel was built for
# (pyproject.toml's cuXXX index) or the build fails with a version-mismatch
# error. If CUDA_HOME isn't already set and a matching /usr/local/cuda-X.Y
# exists, point the build at it; otherwise leave PATH/CUDA_HOME untouched
# (machine-specific — set CUDA_HOME yourself if this doesn't apply to you).
if [[ -z "${CUDA_HOME:-}" ]]; then
    TORCH_CUDA_TAG="$(sed -n 's#.*/cu\([0-9]\+\)".*#\1#p' "${GENRECON_DIR}/pyproject.toml" | head -1)"
    if [[ -n "$TORCH_CUDA_TAG" ]]; then
        TORCH_CUDA_VER="${TORCH_CUDA_TAG:0:2}.${TORCH_CUDA_TAG:2}"
        CANDIDATE="/usr/local/cuda-${TORCH_CUDA_VER}"
        if [[ -d "$CANDIDATE" ]]; then
            export CUDA_HOME="$CANDIDATE"
            export PATH="${CUDA_HOME}/bin:${PATH}"
        fi
    fi
fi

SIMPLIFY_THRESHOLD=250000
TEXTURE_SIZE=2048
NUM_IMGS_PER_SCENE=32
VGGT_EXPORT_TIMEOUT=1800
RUN_GLB=0
USE_TRELLIS=0
RUN_USD=1
COLLISION_APPROXIMATION="convexDecomposition"
SKIP_FRAMES=-1
ALIGN_TO_GRAVITY=1
ROTATE_HORIZONTAL_DEG=0.0
CLASSES=""
START_FROM_STAGE=0
STOP_AFTER_STAGE=999
MAX_CHUNKS_PER_GROUP=""
MAX_INFLATED_VOXELS=""
DEPTH_CONF_THRES=50.0
DEPTH_EDGE_RTOL=0.03
FIX_NUM_VOXELS=16

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--no-align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb] [--use-trellis] [--skip_isaac] [--collision_approximation convexDecomposition|convexHull|boundingCube] [--start-from-stage N] [--stop-after-stage N] [--max_chunks_per_group N] [--max_inflated_voxels N] [--depth_conf_thres N] [--depth_edge_rtol N] [--fix_num_voxels N]" >&2
    exit 1
fi

IMAGE_FOLDER="$1"
SCENE_NAME="$2"
shift 2

while [[ $# -gt 0 ]]; do
    case "$1" in
        --simplify_threshold) SIMPLIFY_THRESHOLD="$2"; shift 2 ;;
        --texture_size) TEXTURE_SIZE="$2"; shift 2 ;;
        --num_imgs_per_scene) NUM_IMGS_PER_SCENE="$2"; shift 2 ;;
        --skip-frames) SKIP_FRAMES="$2"; shift 2 ;;
        --align-to-gravity) ALIGN_TO_GRAVITY=1; shift 1 ;;
        --no-align-to-gravity) ALIGN_TO_GRAVITY=0; shift 1 ;;
        --rotate-horizontal-deg) ROTATE_HORIZONTAL_DEG="$2"; shift 2 ;;
        --classes) CLASSES="$2"; shift 2 ;;
        --run_glb) RUN_GLB=1; shift 1 ;;
        --use-trellis) USE_TRELLIS=1; shift 1 ;;
        --skip_isaac) RUN_USD=0; shift 1 ;;
        --collision_approximation) COLLISION_APPROXIMATION="$2"; shift 2 ;;
        --start-from-stage) START_FROM_STAGE="$2"; shift 2 ;;
        --stop-after-stage) STOP_AFTER_STAGE="$2"; shift 2 ;;
        --max_chunks_per_group) MAX_CHUNKS_PER_GROUP="$2"; shift 2 ;;
        --max_inflated_voxels) MAX_INFLATED_VOXELS="$2"; shift 2 ;;
        --depth_conf_thres) DEPTH_CONF_THRES="$2"; shift 2 ;;
        --depth_edge_rtol) DEPTH_EDGE_RTOL="$2"; shift 2 ;;
        --fix_num_voxels) FIX_NUM_VOXELS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "Image folder not found: $IMAGE_FOLDER" >&2
    exit 1
fi

if [[ "$USE_TRELLIS" -eq 1 && -z "$CLASSES" ]]; then
    echo "--use-trellis requires --classes (TRELLIS.2 reconstruction is per-class)." >&2
    exit 1
fi

RUN_DIR="${GENRECON_DIR}/runs/${SCENE_NAME}"
EXPORT_DIR="${RUN_DIR}/vggt_export"
SEG_DIR="${RUN_DIR}/vggt_export_after_segmentation"
SCENE_DIR="${RUN_DIR}/genrecon_input"
FULL_SCENE_DIR="${RUN_DIR}/genrecon_input_full"
OUTPUT_DIR="${RUN_DIR}/genrecon_output"

mkdir -p "$EXPORT_DIR" "$OUTPUT_DIR"

PIPELINE_LOG="${RUN_DIR}/pipeline.log"
PIPELINE_SUMMARY_LOG="${RUN_DIR}/pipeline_summary.log"
DEBUG_CONFIG_LOG="${RUN_DIR}/debug_config.log"
export GENRECON_PIPELINE_LOG="$PIPELINE_LOG"

# ── timestamp / elapsed-time helpers ──
log() {
    echo "[run_full_pipeline] [$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG" "$PIPELINE_SUMMARY_LOG"
}

# Appends a VS Code debugpy launch.json config entry (program/cwd/args as
# actually invoked for this run) to debug_config.log, so a stage can be
# re-run under the debugger on the exact same data by pasting the entry
# into the target repo's .vscode/launch.json "configurations" array.
log_debug_config() {
    local name="$1" program="$2" cwd="$3"
    shift 3
    python3 "${GENRECON_DIR}/scripts/append_debug_launch_config.py" \
        --log "$DEBUG_CONFIG_LOG" \
        --name "$name" \
        --program "$program" \
        --cwd "$cwd" \
        --python "\${workspaceFolder}/.venv/bin/python" \
        -- "$@"
}

stage_start() {
    STAGE_NAME="$1"
    STAGE_T0=$(date +%s)
    log "${STAGE_NAME}"
}

stage_end() {
    local elapsed=$(( $(date +%s) - STAGE_T0 ))
    log "${STAGE_NAME} done (elapsed $(format_duration "$elapsed"))"
}

format_duration() {
    local s=$1
    printf '%dm%02ds' $((s / 60)) $((s % 60))
}

# Exits after the given stage number finishes (whether it ran or was
# skipped) if that's what --stop-after-stage asked for -- lets a single
# stage be re-run in isolation via --start-from-stage N --stop-after-stage N.
check_stop_after_stage() {
    local n="$1"
    if [[ "$STOP_AFTER_STAGE" -eq "$n" ]]; then
        log "Stopping after stage ${n} (--stop-after-stage ${STOP_AFTER_STAGE})."
        exit 0
    fi
}

declare -A LOG_LINE_OFFSET

# Appends whatever's been newly written to $1 since the last call for that
# file into pipeline.log. Needed because external subprocesses (COB-GS,
# VGGT-Omega, TRELLIS2, IsaacSim) don't log through the shared loguru logger,
# so their per-stage log files (segmentation.log, reconstruct.log, ...) would
# otherwise never make it into the unified pipeline.log.
mirror_log() {
    local log_file="$1"
    local prev="${LOG_LINE_OFFSET[$log_file]:-0}"
    local total
    total=$(wc -l < "$log_file" 2>/dev/null || echo 0)
    if (( total > prev )); then
        tail -n "+$((prev + 1))" "$log_file" >> "$PIPELINE_LOG"
    fi
    LOG_LINE_OFFSET[$log_file]=$total
}

PIPELINE_T0=$(date +%s)
log "Pipeline started for scene '${SCENE_NAME}'"

# ── Stage 0: VGGT-Omega pose/depth prediction + COLMAP-text export ──
if [[ "$START_FROM_STAGE" -le 0 ]]; then
    stage_start "Stage 0: VGGT-Omega export -> ${EXPORT_DIR}"
    VGGT_LOG="${OUTPUT_DIR}/vggt_export.log"

    VGGT_SKIP_FRAMES_ARGS=()
    if [[ "$SKIP_FRAMES" -ne -1 ]]; then
        VGGT_SKIP_FRAMES_ARGS=(--skip-frames "$SKIP_FRAMES")
    fi

    VGGT_GRAVITY_ARGS=()
    if [[ "$ALIGN_TO_GRAVITY" -eq 1 ]]; then
        VGGT_GRAVITY_ARGS+=(--align-to-gravity)
    fi
    if [[ "$ROTATE_HORIZONTAL_DEG" != "0.0" && "$ROTATE_HORIZONTAL_DEG" != "0" ]]; then
        VGGT_GRAVITY_ARGS+=(--rotate-horizontal-deg "$ROTATE_HORIZONTAL_DEG")
    fi

    log_debug_config "stage0_vggt_export" "${VGGT_OMEGA_DIR}/demo_rerun.py" "$VGGT_OMEGA_DIR" \
        "$IMAGE_FOLDER" --checkpoint "$VGGT_CHECKPOINT" --export-for-3dgs "$EXPORT_DIR" \
        "${VGGT_SKIP_FRAMES_ARGS[@]}" "${VGGT_GRAVITY_ARGS[@]}"

    (
        cd "$VGGT_OMEGA_DIR"
        uv run python -u demo_rerun.py "$IMAGE_FOLDER" \
            --checkpoint "$VGGT_CHECKPOINT" \
            --export-for-3dgs "$EXPORT_DIR" \
            "${VGGT_SKIP_FRAMES_ARGS[@]}" \
            "${VGGT_GRAVITY_ARGS[@]}"
    ) > "$VGGT_LOG" 2>&1 &
    VGGT_PID=$!

    # The script writes the COLMAP-text dataset, prints "Exported COLMAP dataset to ...",
    # and only afterwards spawns a Rerun GUI viewer (rr.init(spawn=True)) + logs points/
    # cameras to it. The viewer itself is a separate detached subprocess (the `rr` binary),
    # so once export is confirmed we detach (disown) the demo_rerun.py job and move on
    # immediately instead of waiting for it — the viewer keeps running independently for
    # later inspection, but no longer blocks stages 2/3. vggt_export.log still captures its
    # output if you want to check on it separately.
    wait_elapsed=0
    while true; do
        if grep -q "Exported COLMAP dataset to" "$VGGT_LOG" 2>/dev/null; then
            log "VGGT-Omega export confirmed, detaching viewer process (PID ${VGGT_PID}) and continuing."
            mirror_log "$VGGT_LOG"
            disown "$VGGT_PID" 2>/dev/null || true
            break
        fi
        if ! kill -0 "$VGGT_PID" 2>/dev/null; then
            log "VGGT-Omega process exited before export completed. See $VGGT_LOG" >&2
            mirror_log "$VGGT_LOG"
            exit 1
        fi
        if (( wait_elapsed >= VGGT_EXPORT_TIMEOUT )); then
            log "Timed out waiting for VGGT-Omega export after ${VGGT_EXPORT_TIMEOUT}s." >&2
            kill "$VGGT_PID" 2>/dev/null || true
            mirror_log "$VGGT_LOG"
            exit 1
        fi
        sleep 2
        wait_elapsed=$((wait_elapsed + 2))
    done

    if [[ ! -f "${EXPORT_DIR}/sparse/0/cameras.txt" ]]; then
        log "Expected COLMAP export not found at ${EXPORT_DIR}/sparse/0/cameras.txt" >&2
        exit 1
    fi
    stage_end
else
    log "Stage 0: skipped (--start-from-stage ${START_FROM_STAGE}), assuming existing export at ${EXPORT_DIR}"
fi
check_stop_after_stage 0

# ── Stage 1 (optional): COB-GS 3D segmentation ──
# When --classes is set, only the background reaches GenRecon: foreground
# points are dropped from the sparse point cloud (chunk layout may shift as
# a result, since GenRecon derives chunk placement from points3D.txt) and
# foreground pixels are masked out of every RGB frame.
if [[ -n "$CLASSES" ]]; then
    SEG_LOG="${OUTPUT_DIR}/segmentation.log"
    COBGS_MASK_DIR="${OUTPUT_DIR}/segmentation_raw/masks/classes"

    if [[ "$START_FROM_STAGE" -le 1 ]]; then
        stage_start "Stage 1: COB-GS segmentation (classes: ${CLASSES})"

        # grounded_sam2_stable_tracking.py hardcodes its input image path per
        # --dataset_type (ignoring --dataset_root), so the export must also live
        # at this fixed location relative to COBGS_DIR.
        COBGS_SCENE_DATASET_DIR="${COBGS_DIR}/dataset/${SCENE_NAME}"
        mkdir -p "$COBGS_SCENE_DATASET_DIR"
        ln -sfn "${EXPORT_DIR}/images" "${COBGS_SCENE_DATASET_DIR}/images"
        mkdir -p "${COBGS_SCENE_DATASET_DIR}/sparse"
        ln -sfn "${EXPORT_DIR}/sparse/0" "${COBGS_SCENE_DATASET_DIR}/sparse/0"

        # --output_root is already scene-specific (it's under runs/<scene>/), and --text
        # is only ever a directory label here (--classes is always set, so it never
        # feeds the detection caption) -- so both --flat_output and a fixed --text
        # avoid redundantly repeating the scene name in the output path.
        log_debug_config "stage1_cobgs_segmentation" "${COBGS_DIR}/main_light.py" "$COBGS_DIR" \
            --scene "$SCENE_NAME" --text "classes" \
            --classes "$CLASSES" --dataset_root dataset --dataset_type tum_rgbd \
            --output_root "${OUTPUT_DIR}/segmentation_raw" --resolution -1 --skip_visualize \
            --flat_output --depth_dir "${EXPORT_DIR}/depth" \
            --depth_conf_thres "$DEPTH_CONF_THRES" --depth_edge_rtol "$DEPTH_EDGE_RTOL"

        (
            cd "$COBGS_DIR"
            uv run python -u main_light.py --scene "$SCENE_NAME" --text "classes" \
                --classes "$CLASSES" --dataset_root dataset --dataset_type tum_rgbd \
                --output_root "${OUTPUT_DIR}/segmentation_raw" --resolution -1 --skip_visualize \
                --flat_output --depth_dir "${EXPORT_DIR}/depth" \
                --depth_conf_thres "$DEPTH_CONF_THRES" --depth_edge_rtol "$DEPTH_EDGE_RTOL"
        ) > "$SEG_LOG" 2>&1
        mirror_log "$SEG_LOG"

        if [[ ! -f "${COBGS_MASK_DIR}/labels.json" ]]; then
            log "Expected COB-GS labels.json not found at ${COBGS_MASK_DIR}/labels.json" >&2
            exit 1
        fi

        log "Stage 1: applying segmentation masks -> ${SEG_DIR}"
        mkdir -p "${SEG_DIR}/masked_rgb" "${SEG_DIR}/filtered_colmap"
        ln -sfn "${EXPORT_DIR}/sparse/0/cameras.txt" "${SEG_DIR}/filtered_colmap/cameras.txt"
        ln -sfn "${EXPORT_DIR}/sparse/0/images.txt" "${SEG_DIR}/filtered_colmap/images.txt"

        cd "$GENRECON_DIR"
        log_debug_config "stage1_apply_segmentation_mask" "${GENRECON_DIR}/scripts/apply_segmentation_mask.py" "$GENRECON_DIR" \
            --images_dir "${EXPORT_DIR}/images" \
            --masks_root "$COBGS_MASK_DIR" \
            --background_ply "${COBGS_MASK_DIR}/background/point_cloud/background.ply" \
            --out_rgb_dir "${SEG_DIR}/masked_rgb" \
            --out_points3d "${SEG_DIR}/filtered_colmap/points3D.txt"
        uv run python -u scripts/apply_segmentation_mask.py \
            --images_dir "${EXPORT_DIR}/images" \
            --masks_root "$COBGS_MASK_DIR" \
            --background_ply "${COBGS_MASK_DIR}/background/point_cloud/background.ply" \
            --out_rgb_dir "${SEG_DIR}/masked_rgb" \
            --out_points3d "${SEG_DIR}/filtered_colmap/points3D.txt" \
            >> "$SEG_LOG" 2>&1
        mirror_log "$SEG_LOG"
        stage_end
    else
        log "Stage 1: skipped (--start-from-stage ${START_FROM_STAGE}), assuming existing segmentation at ${SEG_DIR}"
    fi
    check_stop_after_stage 1

    # ── Stage 2: RGBA best-view mask export ──
    # For each class, picks the frame where the object covers the largest
    # fraction of the image (most detail) and composites an RGBA image from
    # it, for use by image-conditioned generators (e.g. TRELLIS.2 generate.py).
    if [[ "$START_FROM_STAGE" -le 2 ]]; then
        stage_start "Stage 2: RGBA mask export -> ${COBGS_MASK_DIR}/<class>/mask_rgba"
        log_debug_config "stage2_export_rgba_masks" "${GENRECON_DIR}/scripts/export_rgba_masks.py" "$GENRECON_DIR" \
            --images_dir "${EXPORT_DIR}/images" \
            --masks_root "$COBGS_MASK_DIR"
        uv run python -u scripts/export_rgba_masks.py \
            --images_dir "${EXPORT_DIR}/images" \
            --masks_root "$COBGS_MASK_DIR" \
            >> "$SEG_LOG" 2>&1
        mirror_log "$SEG_LOG"
        stage_end
    else
        log "Stage 2: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 2

    # ── Stage 3 (optional): TRELLIS.2 3D reconstruction per class ──
    if [[ "$USE_TRELLIS" -eq 1 && "$START_FROM_STAGE" -le 3 ]]; then
        stage_start "Stage 3: TRELLIS.2 reconstruction -> ${RUN_DIR}/trellis2_meshes"
        TRELLIS2_INPUT_DIR="${RUN_DIR}/trellis2_input"
        TRELLIS2_OUTPUT_DIR="${RUN_DIR}/trellis2_meshes"
        log_debug_config "stage3_stage_trellis2_inputs" "${GENRECON_DIR}/scripts/stage_trellis2_inputs.py" "$GENRECON_DIR" \
            --masks_root "$COBGS_MASK_DIR" \
            --out_dir "$TRELLIS2_INPUT_DIR"
        uv run python -u scripts/stage_trellis2_inputs.py \
            --masks_root "$COBGS_MASK_DIR" \
            --out_dir "$TRELLIS2_INPUT_DIR" \
            >> "$SEG_LOG" 2>&1

        log_debug_config "stage3_trellis2_generate" "${TRELLIS2_DIR}/generate.py" "$TRELLIS2_DIR" \
            --input "$TRELLIS2_INPUT_DIR" \
            --output-dir "$TRELLIS2_OUTPUT_DIR" \
            --resolution 512 --no-preview
        (
            cd "$TRELLIS2_DIR"
            uv run --no-sync generate.py \
                --input "$TRELLIS2_INPUT_DIR" \
                --output-dir "$TRELLIS2_OUTPUT_DIR" \
                --resolution 512 --no-preview
        ) >> "$SEG_LOG" 2>&1
        mirror_log "$SEG_LOG"
        stage_end
    fi
    check_stop_after_stage 3
fi

# ── Stage 4: stage GenRecon scene dir(s) ──
# When --classes is set, also stages FULL_SCENE_DIR pointing at the *unmasked*
# export (images + full, object-inclusive point cloud) -- used by Stage 5's
# second, unexcluded reconstruct_scene.py run as the source for real per-object
# meshes, since the primary (masked) run's mesh.ply no longer contains them.
if [[ "$START_FROM_STAGE" -le 4 ]]; then
    stage_start "Stage 4: staging ${SCENE_DIR}"
    mkdir -p "$SCENE_DIR"
    if [[ -n "$CLASSES" ]]; then
        ln -sfn "${SEG_DIR}/masked_rgb" "${SCENE_DIR}/rgb"
        ln -sfn "${SEG_DIR}/filtered_colmap" "${SCENE_DIR}/colmap"
        mkdir -p "$FULL_SCENE_DIR"
        ln -sfn "${EXPORT_DIR}/images" "${FULL_SCENE_DIR}/rgb"
        ln -sfn "${EXPORT_DIR}/sparse/0" "${FULL_SCENE_DIR}/colmap"
    else
        ln -sfn "${EXPORT_DIR}/images" "${SCENE_DIR}/rgb"
        ln -sfn "${EXPORT_DIR}/sparse/0" "${SCENE_DIR}/colmap"
    fi
    stage_end
else
    log "Stage 4: skipped (--start-from-stage ${START_FROM_STAGE}), assuming existing ${SCENE_DIR}"
fi
check_stop_after_stage 4

# ── Stage 5: GenRecon reconstruction + GLB bake (exp2 settings) ──
cd "$GENRECON_DIR"
export MPLBACKEND=Agg

if [[ "$START_FROM_STAGE" -le 5 ]]; then
    stage_start "Stage 5: reconstruct_scene.py"

    RECON_VRAM_ARGS=()
    if [[ -n "$MAX_CHUNKS_PER_GROUP" ]]; then
        RECON_VRAM_ARGS+=(--max_chunks_per_group "$MAX_CHUNKS_PER_GROUP")
    fi
    if [[ -n "$MAX_INFLATED_VOXELS" ]]; then
        RECON_VRAM_ARGS+=(--max_inflated_voxels "$MAX_INFLATED_VOXELS")
    fi
    if [[ -n "$FIX_NUM_VOXELS" ]]; then
        RECON_VRAM_ARGS+=(--fix_num_chunks "$FIX_NUM_VOXELS")
    fi

    # Excludes each segmented class's object from generation itself (voxel-level
    # carving before shape/texture SLat sampling), instead of relying on the
    # black-masked/point-dropped input alone -- GenRecon is generative and will
    # otherwise "resurrect" plausible geometry into the masked region. Also runs
    # a second, unexcluded reconstruction over FULL_SCENE_DIR's unmasked images
    # (--unmasked_path), sharing the primary run's chunk geometry/world frame,
    # so Stage 8 has a real (non-hallucinated) source mesh to crop each object
    # from -- the primary run's mesh.ply no longer contains them.
    RECON_EXCLUDE_ARGS=()
    if [[ -n "$CLASSES" ]]; then
        RECON_EXCLUDE_ARGS=(--exclude_masks_root "$COBGS_MASK_DIR" --unmasked_path "$FULL_SCENE_DIR")
    fi

    log_debug_config "stage5_reconstruct_scene" "${GENRECON_DIR}/reconstruct_scene.py" "$GENRECON_DIR" \
        --mode Iphone --path "$SCENE_DIR" --output_path "$OUTPUT_DIR" \
        --ss_ckpt checkpoints/sparse_structure/ckpts/sparse_structure.pt \
        --shape_ckpt checkpoints/shape_slat/ckpts/shape_slat.pt \
        --tex_ckpt checkpoints/texture_slat/ckpts/texture_slat.pt \
        --num_imgs_per_scene "$NUM_IMGS_PER_SCENE" --colmap_subdir colmap \
        "${RECON_VRAM_ARGS[@]}" "${RECON_EXCLUDE_ARGS[@]}"
    uv run python -u reconstruct_scene.py --mode Iphone --path "$SCENE_DIR" --output_path "$OUTPUT_DIR" \
        --ss_ckpt checkpoints/sparse_structure/ckpts/sparse_structure.pt \
        --shape_ckpt checkpoints/shape_slat/ckpts/shape_slat.pt \
        --tex_ckpt checkpoints/texture_slat/ckpts/texture_slat.pt \
        --num_imgs_per_scene "$NUM_IMGS_PER_SCENE" --colmap_subdir colmap \
        "${RECON_VRAM_ARGS[@]}" "${RECON_EXCLUDE_ARGS[@]}" \
        > "${OUTPUT_DIR}/reconstruct.log" 2>&1
    mirror_log "${OUTPUT_DIR}/reconstruct.log"
    stage_end
else
    log "Stage 5: skipped (--start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 5

# ── Stage 6: reprojection validation ──
if [[ "$START_FROM_STAGE" -le 6 ]]; then
    stage_start "Stage 6: render_reprojection_validation.py"
    log_debug_config "stage6_render_reprojection_validation" "${GENRECON_DIR}/scripts/render_reprojection_validation.py" "$GENRECON_DIR" \
        --mesh_ply "${OUTPUT_DIR}/mesh.ply" \
        --colmap_dir "${SCENE_DIR}/colmap" \
        --images_dir "${SCENE_DIR}/rgb" \
        --out_synth_dir "${OUTPUT_DIR}/synth_views" \
        --out_compare_dir "${OUTPUT_DIR}/compare_views"
    uv run python -u scripts/render_reprojection_validation.py \
        --mesh_ply "${OUTPUT_DIR}/mesh.ply" \
        --colmap_dir "${SCENE_DIR}/colmap" \
        --images_dir "${SCENE_DIR}/rgb" \
        --out_synth_dir "${OUTPUT_DIR}/synth_views" \
        --out_compare_dir "${OUTPUT_DIR}/compare_views" \
        > "${OUTPUT_DIR}/reprojection_validation.log" 2>&1
    mirror_log "${OUTPUT_DIR}/reprojection_validation.log"
    stage_end
else
    log "Stage 6: skipped (--start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 6

# ── Stage 7: collect shapes (reconstructed mesh + per-class point clouds) ──
SHAPES_DIR="${OUTPUT_DIR}/shapes"
if [[ "$START_FROM_STAGE" -le 7 ]]; then
    stage_start "Stage 7: collecting shapes -> ${OUTPUT_DIR}/shapes"
    mkdir -p "$SHAPES_DIR"
    mv "${OUTPUT_DIR}/mesh.ply" "${SHAPES_DIR}/mesh.ply"
    if [[ -n "$CLASSES" ]]; then
        find "$COBGS_MASK_DIR" -mindepth 3 -maxdepth 3 -name "*.ply" -exec cp {} "$SHAPES_DIR/" \;
    fi
    stage_end
else
    log "Stage 7: skipped (--start-from-stage ${START_FROM_STAGE}), assuming existing ${SHAPES_DIR}"
fi
check_stop_after_stage 7

# ── Stage 8 (optional): per-object mesh extraction (cascading convex hull crop) ──
# Crops each class's object out of the scene mesh in turn, using its point
# cloud (shapes/<label>.ply) as a spatial reference -- both are already in
# the same world frame as mesh.ply, so no alignment step is needed. mesh.ply
# itself is left untouched: background_mesh.ply is seeded as a copy of it
# up front, and each class is cropped out of background_mesh.ply in place
# (extract_object_mesh.py fully reads --mesh_ply before writing
# --remainder_out_ply, so reading and overwriting the same path is safe), so
# the final background_mesh.ply is exactly mesh.ply with every class's
# object removed. background.ply (COB-GS's separately-sampled leftover-point
# cloud) is not used as a crop input here; it's still copied into shapes/ by
# stage 7 for reference. --colmap_dir/--masks_dir feed extract_object_mesh.py's
# stage-2 padding search, which tightens the crop by reprojecting the padded
# hull into each camera view and comparing it against that class's real
# per-frame segmentation mask (COBGS_MASK_DIR/<label>/mask_bin), instead of
# using a single fixed --hull_padding.
#
# --object_mesh_ply points the object side of the crop at Stage 5's second
# (unexcluded) reconstruction instead of background_mesh.ply -- the primary
# run had every class excluded from generation, so background_mesh.ply no
# longer contains real object geometry to crop; --remainder_out_ply still
# always chains off background_mesh.ply.
OBJECT_SOURCE_MESH="${OUTPUT_DIR}/object_source_mesh.ply"
if [[ -n "$CLASSES" && "$START_FROM_STAGE" -le 8 ]]; then
    stage_start "Stage 8: cascading per-object mesh extraction -> ${SHAPES_DIR}"
    BACKGROUND_MESH="${SHAPES_DIR}/background_mesh.ply"
    cp "${SHAPES_DIR}/mesh.ply" "$BACKGROUND_MESH"
    for obj_ply in "$SHAPES_DIR"/*.ply; do
        obj_name="$(basename "$obj_ply")"
        # Skip mesh.ply/background.ply plus anything matching this stage's own
        # <label>_mesh.ply output naming -- a stale one from a previous run
        # (Stage 7 doesn't wipe SHAPES_DIR before re-copying raw point clouds)
        # would otherwise get globbed as a brand-new "object" here, producing a
        # spurious <label>_mesh_mesh.ply crop and a duplicate USD asset later.
        [[ "$obj_name" == "mesh.ply" || "$obj_name" == "background.ply" || "$obj_name" == *_mesh.ply ]] && continue
        label="${obj_name%.ply}"
        log_debug_config "stage8_extract_object_mesh_${label}" "${GENRECON_DIR}/scripts/extract_object_mesh.py" "$GENRECON_DIR" \
            --mesh_ply "$BACKGROUND_MESH" \
            --object_mesh_ply "$OBJECT_SOURCE_MESH" \
            --object_ply "$obj_ply" \
            --out_ply "${SHAPES_DIR}/${label}_mesh.ply" \
            --remainder_out_ply "$BACKGROUND_MESH" \
            --colmap_dir "${SCENE_DIR}/colmap" \
            --masks_dir "${COBGS_MASK_DIR}/${label}/mask_bin"
        uv run python -u scripts/extract_object_mesh.py \
            --mesh_ply "$BACKGROUND_MESH" \
            --object_mesh_ply "$OBJECT_SOURCE_MESH" \
            --object_ply "$obj_ply" \
            --out_ply "${SHAPES_DIR}/${label}_mesh.ply" \
            --remainder_out_ply "$BACKGROUND_MESH" \
            --colmap_dir "${SCENE_DIR}/colmap" \
            --masks_dir "${COBGS_MASK_DIR}/${label}/mask_bin" \
            >> "${OUTPUT_DIR}/reconstruct.log" 2>&1
    done
    mirror_log "${OUTPUT_DIR}/reconstruct.log"
    stage_end
fi
check_stop_after_stage 8

# ── Stage 9/10/11 (optional): per-object mesh -> GLB -> physics-ready USD -> composed scene ──
# Only the *_mesh.ply crops from Stage 8 are converted (never the whole-scene
# mesh.ply, and never the plain per-class point clouds like chair.ply/background.ply).
if [[ "$RUN_USD" -eq 1 ]]; then
    if [[ "$START_FROM_STAGE" -le 9 ]]; then
        if [[ "$USE_TRELLIS" -eq 1 ]]; then
            stage_start "Stage 9: mesh_to_glb.py -> ${SHAPES_DIR}/glb (+ TRELLIS.2 substitution)"
        else
            stage_start "Stage 9: mesh_to_glb.py -> ${SHAPES_DIR}/glb"
        fi
        log_debug_config "stage9_mesh_to_glb" "${GENRECON_DIR}/scripts/mesh_to_glb.py" "$GENRECON_DIR" \
            --shapes_dir "$SHAPES_DIR" \
            --out_dir "${SHAPES_DIR}/glb"
        uv run python -u scripts/mesh_to_glb.py \
            --shapes_dir "$SHAPES_DIR" \
            --out_dir "${SHAPES_DIR}/glb" \
            > "${OUTPUT_DIR}/mesh_to_glb.log" 2>&1
        mirror_log "${OUTPUT_DIR}/mesh_to_glb.log"

        # --use-trellis: swap the crop-based glb produced above for the
        # TRELLIS.2 reconstruction, rescaled/translated (no rotation search)
        # into the scene's world frame by align_trellis2_mesh_to_scene.py,
        # for every class that has both a trellis mesh and a scene crop.
        # Stage 10/11 only ever move an asset by a pure translation derived
        # from its own glb bbox (no external pose file), so overwriting
        # <label>/mesh.glb in place here is a drop-in substitution -- no
        # changes needed downstream. background is never substituted (no
        # TRELLIS.2 reconstruction exists for it).
        if [[ "$USE_TRELLIS" -eq 1 ]]; then
            IFS=',' read -ra USE_TRELLIS_CLASSES <<< "$CLASSES"
            for label in "${USE_TRELLIS_CLASSES[@]}"; do
                TRELLIS_LABEL_GLB="${RUN_DIR}/trellis2_meshes/${label}/mesh.glb"
                SCENE_CROP_PLY="${SHAPES_DIR}/${label}_mesh.ply"
                if [[ ! -f "$TRELLIS_LABEL_GLB" ]]; then
                    log "Stage 9: --use-trellis: no TRELLIS.2 mesh for '${label}' at ${TRELLIS_LABEL_GLB}, keeping scene-crop glb."
                    continue
                fi
                if [[ ! -f "$SCENE_CROP_PLY" ]]; then
                    log "Stage 9: --use-trellis: no scene crop for '${label}' at ${SCENE_CROP_PLY}, keeping scene-crop glb."
                    continue
                fi
                log_debug_config "stage9_align_trellis2_mesh_${label}" "${GENRECON_DIR}/scripts/align_trellis2_mesh_to_scene.py" "$GENRECON_DIR" \
                    --trellis_glb "$TRELLIS_LABEL_GLB" \
                    --scene_mesh_ply "$SCENE_CROP_PLY" \
                    --out_glb "${SHAPES_DIR}/glb/${label}/mesh.glb"
                uv run python -u scripts/align_trellis2_mesh_to_scene.py \
                    --trellis_glb "$TRELLIS_LABEL_GLB" \
                    --scene_mesh_ply "$SCENE_CROP_PLY" \
                    --out_glb "${SHAPES_DIR}/glb/${label}/mesh.glb" \
                    >> "${OUTPUT_DIR}/mesh_to_glb.log" 2>&1
            done
            mirror_log "${OUTPUT_DIR}/mesh_to_glb.log"
        fi
        stage_end
    else
        log "Stage 9: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 9

    if [[ "$START_FROM_STAGE" -le 10 ]]; then
        stage_start "Stage 10: convert_asset.py (collision_approximation=${COLLISION_APPROXIMATION}) -> ${SHAPES_DIR}/glb/<label>/asset.usd"
        log_debug_config "stage10_convert_asset" "${ISAACSIM_DIR}/convert_asset.py" "$ISAACSIM_DIR" \
            --input "${SHAPES_DIR}/glb" \
            --collision-approximation "$COLLISION_APPROXIMATION"
        (
            cd "$ISAACSIM_DIR"
            uv run convert_asset.py \
                --input "${SHAPES_DIR}/glb" \
                --collision-approximation "$COLLISION_APPROXIMATION"
        ) > "${OUTPUT_DIR}/convert_asset.log" 2>&1
        mirror_log "${OUTPUT_DIR}/convert_asset.log"
        stage_end
    else
        log "Stage 10: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 10

    if [[ "$START_FROM_STAGE" -le 11 ]]; then
        stage_start "Stage 11: compose_isaac_scene.py -> ${SHAPES_DIR}/glb/scene.usda"
        log_debug_config "stage11_compose_isaac_scene" "${ISAACSIM_DIR}/compose_isaac_scene.py" "$ISAACSIM_DIR" \
            --input "${SHAPES_DIR}/glb" \
            --output "${SHAPES_DIR}/glb/scene.usda" \
            --background-label background
        (
            cd "$ISAACSIM_DIR"
            uv run compose_isaac_scene.py \
                --input "${SHAPES_DIR}/glb" \
                --output "${SHAPES_DIR}/glb/scene.usda" \
                --background-label background
        ) > "${OUTPUT_DIR}/compose_isaac_scene.log" 2>&1
        mirror_log "${OUTPUT_DIR}/compose_isaac_scene.log"
        stage_end
    else
        log "Stage 11: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 11
fi

if [[ "$RUN_GLB" -eq 1 && "$START_FROM_STAGE" -le 12 ]]; then
    stage_start "Stage 12: chunked_to_glb.py (simplify_threshold=${SIMPLIFY_THRESHOLD}, texture_size=${TEXTURE_SIZE})"
    log_debug_config "stage12_chunked_to_glb" "${GENRECON_DIR}/chunked_to_glb.py" "$GENRECON_DIR" \
        --inputs "${OUTPUT_DIR}/to_glb_inputs.pt" \
        --chunk_inputs "${OUTPUT_DIR}/chunk_inputs.pt" \
        --output_dir "$OUTPUT_DIR" \
        --simplify_threshold "$SIMPLIFY_THRESHOLD" \
        --texture_size "$TEXTURE_SIZE"
    uv run python -u chunked_to_glb.py \
        --inputs "${OUTPUT_DIR}/to_glb_inputs.pt" \
        --chunk_inputs "${OUTPUT_DIR}/chunk_inputs.pt" \
        --output_dir "$OUTPUT_DIR" \
        --simplify_threshold "$SIMPLIFY_THRESHOLD" \
        --texture_size "$TEXTURE_SIZE" \
        > "${OUTPUT_DIR}/glb.log" 2>&1
    mirror_log "${OUTPUT_DIR}/glb.log"
    stage_end

    log "Done: ${OUTPUT_DIR}/scene.glb"
elif [[ "$RUN_GLB" -eq 1 ]]; then
    log "Stage 12: skipped (--start-from-stage ${START_FROM_STAGE})"
    log "Done: ${SHAPES_DIR}/mesh.ply"
else
    log "--run_glb not set, skipping GLB bake."
    log "Done: ${SHAPES_DIR}/mesh.ply"
fi

# ── Stage 13 (optional): organize final per-object deliverables ──
# Copies the background point-cloud/mesh plus, per class, the detection
# preview, point cloud, cropped mesh, TRELLIS.2 input image and TRELLIS.2
# mesh (if --use-trellis was used) into one self-contained final_objects/
# folder -- only meaningful when --classes was set (otherwise there's no
# per-object split, just a single whole-scene mesh.ply).
FINAL_OBJECTS_DIR="${RUN_DIR}/final_objects"
if [[ -n "$CLASSES" && "$START_FROM_STAGE" -le 13 ]]; then
    stage_start "Stage 13: organizing final objects -> ${FINAL_OBJECTS_DIR}"
    log_debug_config "stage13_organize_final_objects" "${GENRECON_DIR}/scripts/organize_final_objects.py" "$GENRECON_DIR" \
        --shapes_dir "$SHAPES_DIR" \
        --segmentation_raw_dir "${OUTPUT_DIR}/segmentation_raw" \
        --trellis2_input_dir "${RUN_DIR}/trellis2_input" \
        --trellis2_meshes_dir "${RUN_DIR}/trellis2_meshes" \
        --out_dir "$FINAL_OBJECTS_DIR"
    uv run python -u scripts/organize_final_objects.py \
        --shapes_dir "$SHAPES_DIR" \
        --segmentation_raw_dir "${OUTPUT_DIR}/segmentation_raw" \
        --trellis2_input_dir "${RUN_DIR}/trellis2_input" \
        --trellis2_meshes_dir "${RUN_DIR}/trellis2_meshes" \
        --out_dir "$FINAL_OBJECTS_DIR" \
        >> "${OUTPUT_DIR}/reconstruct.log" 2>&1
    mirror_log "${OUTPUT_DIR}/reconstruct.log"
    stage_end
else
    log "Stage 13: skipped (no --classes, or --start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 13

PIPELINE_ELAPSED=$(( $(date +%s) - PIPELINE_T0 ))
log "Pipeline finished for scene '${SCENE_NAME}' (total elapsed $(format_duration "$PIPELINE_ELAPSED"))"
