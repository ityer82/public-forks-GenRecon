#!/usr/bin/env bash
# End-to-end pipeline: VGGT-Omega pose/depth prediction -> [optional COB-GS 3D segmentation] -> GenRecon reconstruction -> GLB bake.
#
# Usage:
#   ./run_full_pipeline.sh <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--no-align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb] [--run_trellis2] [--run_usd] [--collision_approximation convexDecomposition|convexHull|boundingCube] [--start-from-stage N]
#
# Note: gravity alignment is ON by default; pass --no-align-to-gravity to disable it.
#
# Example:
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --run_glb
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --align-to-gravity --rotate-horizontal-deg 90
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "person,chair,bag"
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "person,chair,bag" --run_trellis2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENRECON_DIR="$SCRIPT_DIR"
VGGT_OMEGA_DIR="$(cd "$GENRECON_DIR/../vggt-omega" && pwd)"
COBGS_DIR="$(cd "$GENRECON_DIR/../COB-GS" && pwd)"
TRELLIS2_DIR="$(cd "$GENRECON_DIR/../trellis2" && pwd)"
ISAACSIM_DIR="$(cd "$GENRECON_DIR/../IsaacSim" && pwd)"
VGGT_CHECKPOINT="${VGGT_OMEGA_DIR}/vggt_omega_1b_512.pt"

SIMPLIFY_THRESHOLD=250000
TEXTURE_SIZE=2048
NUM_IMGS_PER_SCENE=32
VGGT_EXPORT_TIMEOUT=1800
RUN_GLB=0
RUN_TRELLIS2=0
RUN_USD=0
COLLISION_APPROXIMATION="convexDecomposition"
SKIP_FRAMES=-1
ALIGN_TO_GRAVITY=1
ROTATE_HORIZONTAL_DEG=0.0
CLASSES=""
START_FROM_STAGE=0

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--no-align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb] [--run_trellis2] [--run_usd] [--collision_approximation convexDecomposition|convexHull|boundingCube] [--start-from-stage N]" >&2
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
        --run_trellis2) RUN_TRELLIS2=1; shift 1 ;;
        --run_usd) RUN_USD=1; shift 1 ;;
        --collision_approximation) COLLISION_APPROXIMATION="$2"; shift 2 ;;
        --start-from-stage) START_FROM_STAGE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "Image folder not found: $IMAGE_FOLDER" >&2
    exit 1
fi

RUN_DIR="${GENRECON_DIR}/runs/${SCENE_NAME}"
EXPORT_DIR="${RUN_DIR}/vggt_export"
SEG_DIR="${RUN_DIR}/vggt_export_after_segmentation"
SCENE_DIR="${RUN_DIR}/genrecon_input"
OUTPUT_DIR="${RUN_DIR}/genrecon_output"

mkdir -p "$EXPORT_DIR" "$OUTPUT_DIR"

PIPELINE_LOG="${RUN_DIR}/pipeline.log"
export GENRECON_PIPELINE_LOG="$PIPELINE_LOG"

# ── timestamp / elapsed-time helpers ──
log() {
    echo "[run_full_pipeline] [$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
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
            disown "$VGGT_PID" 2>/dev/null || true
            break
        fi
        if ! kill -0 "$VGGT_PID" 2>/dev/null; then
            log "VGGT-Omega process exited before export completed. See $VGGT_LOG" >&2
            exit 1
        fi
        if (( wait_elapsed >= VGGT_EXPORT_TIMEOUT )); then
            log "Timed out waiting for VGGT-Omega export after ${VGGT_EXPORT_TIMEOUT}s." >&2
            kill "$VGGT_PID" 2>/dev/null || true
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
        (
            cd "$COBGS_DIR"
            uv run python -u main_light.py --scene "$SCENE_NAME" --text "classes" \
                --classes "$CLASSES" --dataset_root dataset --dataset_type tum_rgbd \
                --output_root "${OUTPUT_DIR}/segmentation_raw" --resolution -1 --skip_visualize \
                --flat_output
        ) > "$SEG_LOG" 2>&1

        if [[ ! -f "${COBGS_MASK_DIR}/labels.json" ]]; then
            log "Expected COB-GS labels.json not found at ${COBGS_MASK_DIR}/labels.json" >&2
            exit 1
        fi

        log "Stage 1: applying segmentation masks -> ${SEG_DIR}"
        mkdir -p "${SEG_DIR}/masked_rgb" "${SEG_DIR}/filtered_colmap"
        ln -sfn "${EXPORT_DIR}/sparse/0/cameras.txt" "${SEG_DIR}/filtered_colmap/cameras.txt"
        ln -sfn "${EXPORT_DIR}/sparse/0/images.txt" "${SEG_DIR}/filtered_colmap/images.txt"

        cd "$GENRECON_DIR"
        uv run python -u scripts/apply_segmentation_mask.py \
            --images_dir "${EXPORT_DIR}/images" \
            --masks_root "$COBGS_MASK_DIR" \
            --background_ply "${COBGS_MASK_DIR}/background/point_cloud/background.ply" \
            --out_rgb_dir "${SEG_DIR}/masked_rgb" \
            --out_points3d "${SEG_DIR}/filtered_colmap/points3D.txt" \
            >> "$SEG_LOG" 2>&1
        stage_end
    else
        log "Stage 1: skipped (--start-from-stage ${START_FROM_STAGE}), assuming existing segmentation at ${SEG_DIR}"
    fi

    # ── Stage 2: RGBA best-view mask export ──
    # For each class, picks the frame where the object covers the largest
    # fraction of the image (most detail) and composites an RGBA image from
    # it, for use by image-conditioned generators (e.g. TRELLIS.2 generate.py).
    if [[ "$START_FROM_STAGE" -le 2 ]]; then
        stage_start "Stage 2: RGBA mask export -> ${COBGS_MASK_DIR}/<class>/mask_rgba"
        uv run python -u scripts/export_rgba_masks.py \
            --images_dir "${EXPORT_DIR}/images" \
            --masks_root "$COBGS_MASK_DIR" \
            >> "$SEG_LOG" 2>&1
        stage_end
    else
        log "Stage 2: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi

    # ── Stage 3 (optional): TRELLIS.2 3D reconstruction per class ──
    if [[ "$RUN_TRELLIS2" -eq 1 && "$START_FROM_STAGE" -le 3 ]]; then
        stage_start "Stage 3: TRELLIS.2 reconstruction -> ${RUN_DIR}/trellis2_meshes"
        TRELLIS2_INPUT_DIR="${RUN_DIR}/trellis2_input"
        TRELLIS2_OUTPUT_DIR="${RUN_DIR}/trellis2_meshes"
        uv run python -u scripts/stage_trellis2_inputs.py \
            --masks_root "$COBGS_MASK_DIR" \
            --out_dir "$TRELLIS2_INPUT_DIR" \
            >> "$SEG_LOG" 2>&1

        (
            cd "$TRELLIS2_DIR"
            uv run --no-sync generate.py \
                --input "$TRELLIS2_INPUT_DIR" \
                --output-dir "$TRELLIS2_OUTPUT_DIR" \
                --resolution 512 --no-preview
        ) >> "$SEG_LOG" 2>&1
        stage_end
    fi
fi

# ── Stage 4: stage GenRecon scene dir ──
if [[ "$START_FROM_STAGE" -le 4 ]]; then
    stage_start "Stage 4: staging ${SCENE_DIR}"
    mkdir -p "$SCENE_DIR"
    if [[ -n "$CLASSES" ]]; then
        ln -sfn "${SEG_DIR}/masked_rgb" "${SCENE_DIR}/rgb"
        ln -sfn "${SEG_DIR}/filtered_colmap" "${SCENE_DIR}/colmap"
    else
        ln -sfn "${EXPORT_DIR}/images" "${SCENE_DIR}/rgb"
        ln -sfn "${EXPORT_DIR}/sparse/0" "${SCENE_DIR}/colmap"
    fi
    stage_end
else
    log "Stage 4: skipped (--start-from-stage ${START_FROM_STAGE}), assuming existing ${SCENE_DIR}"
fi

# ── Stage 5: GenRecon reconstruction + GLB bake (exp2 settings) ──
cd "$GENRECON_DIR"
export MPLBACKEND=Agg

if [[ "$START_FROM_STAGE" -le 5 ]]; then
    stage_start "Stage 5: reconstruct_scene.py"
    uv run python -u reconstruct_scene.py --mode Iphone --path "$SCENE_DIR" --output_path "$OUTPUT_DIR" \
        --ss_ckpt checkpoints/sparse_structure/ckpts/sparse_structure.pt \
        --shape_ckpt checkpoints/shape_slat/ckpts/shape_slat.pt \
        --tex_ckpt checkpoints/texture_slat/ckpts/texture_slat.pt \
        --num_imgs_per_scene "$NUM_IMGS_PER_SCENE" --colmap_subdir colmap \
        > "${OUTPUT_DIR}/reconstruct.log" 2>&1
    stage_end
else
    log "Stage 5: skipped (--start-from-stage ${START_FROM_STAGE})"
fi

# ── Stage 6: reprojection validation ──
if [[ "$START_FROM_STAGE" -le 6 ]]; then
    stage_start "Stage 6: render_reprojection_validation.py"
    uv run python -u scripts/render_reprojection_validation.py \
        --mesh_ply "${OUTPUT_DIR}/mesh.ply" \
        --colmap_dir "${SCENE_DIR}/colmap" \
        --images_dir "${SCENE_DIR}/rgb" \
        --out_synth_dir "${OUTPUT_DIR}/synth_views" \
        --out_compare_dir "${OUTPUT_DIR}/compare_views" \
        > "${OUTPUT_DIR}/reprojection_validation.log" 2>&1
    stage_end
else
    log "Stage 6: skipped (--start-from-stage ${START_FROM_STAGE})"
fi

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

# ── Stage 8 (optional): per-object mesh extraction (cascading convex hull crop) ──
# Crops each class's object out of the scene mesh in turn, using its point
# cloud (shapes/<label>.ply) as a spatial reference -- both are already in
# the same world frame as mesh.ply, so no alignment step is needed. Each
# class is cropped out of what's left of the mesh after the previous class
# (rather than independently from the original), so the final remainder --
# saved as background_mesh.ply -- is exactly mesh.ply with every class's
# object removed. background.ply (COB-GS's separately-sampled leftover-point
# cloud) is not used as a crop input here; it's still copied into shapes/ by
# stage 7 for reference.
if [[ -n "$CLASSES" && "$START_FROM_STAGE" -le 8 ]]; then
    stage_start "Stage 8: cascading per-object mesh extraction -> ${SHAPES_DIR}"
    CURRENT_MESH="${SHAPES_DIR}/mesh.ply"
    REMAINDER_PLY="${SHAPES_DIR}/_remainder.ply"
    CROPPED_ANY=0
    for obj_ply in "$SHAPES_DIR"/*.ply; do
        obj_name="$(basename "$obj_ply")"
        [[ "$obj_name" == "mesh.ply" || "$obj_name" == "background.ply" ]] && continue
        label="${obj_name%.ply}"
        uv run python -u scripts/extract_object_mesh.py \
            --mesh_ply "$CURRENT_MESH" \
            --object_ply "$obj_ply" \
            --out_ply "${SHAPES_DIR}/${label}_mesh.ply" \
            --remainder_out_ply "$REMAINDER_PLY" \
            >> "${OUTPUT_DIR}/reconstruct.log" 2>&1
        CURRENT_MESH="$REMAINDER_PLY"
        CROPPED_ANY=1
    done
    if [[ "$CROPPED_ANY" -eq 1 ]]; then
        mv "$REMAINDER_PLY" "${SHAPES_DIR}/background_mesh.ply"
    fi
    stage_end
fi

# ── Stage 9/10 (optional): per-object mesh -> GLB -> physics-ready USD ──
# Only the *_mesh.ply crops from Stage 8 are converted (never the whole-scene
# mesh.ply, and never the plain per-class point clouds like chair.ply/background.ply).
if [[ "$RUN_USD" -eq 1 ]]; then
    if [[ "$START_FROM_STAGE" -le 9 ]]; then
        stage_start "Stage 9: mesh_to_glb.py -> ${SHAPES_DIR}/glb"
        uv run python -u scripts/mesh_to_glb.py \
            --shapes_dir "$SHAPES_DIR" \
            --out_dir "${SHAPES_DIR}/glb" \
            > "${OUTPUT_DIR}/mesh_to_glb.log" 2>&1
        stage_end
    else
        log "Stage 9: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi

    if [[ "$START_FROM_STAGE" -le 10 ]]; then
        stage_start "Stage 10: convert_asset.py (collision_approximation=${COLLISION_APPROXIMATION}) -> ${SHAPES_DIR}/glb/<label>/asset.usd"
        (
            cd "$ISAACSIM_DIR"
            uv run convert_asset.py \
                --input "${SHAPES_DIR}/glb" \
                --collision-approximation "$COLLISION_APPROXIMATION"
        ) > "${OUTPUT_DIR}/convert_asset.log" 2>&1
        stage_end
    else
        log "Stage 10: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
fi

if [[ "$RUN_GLB" -eq 1 && "$START_FROM_STAGE" -le 11 ]]; then
    stage_start "Stage 11: chunked_to_glb.py (simplify_threshold=${SIMPLIFY_THRESHOLD}, texture_size=${TEXTURE_SIZE})"
    uv run python -u chunked_to_glb.py \
        --inputs "${OUTPUT_DIR}/to_glb_inputs.pt" \
        --chunk_inputs "${OUTPUT_DIR}/chunk_inputs.pt" \
        --output_dir "$OUTPUT_DIR" \
        --simplify_threshold "$SIMPLIFY_THRESHOLD" \
        --texture_size "$TEXTURE_SIZE" \
        > "${OUTPUT_DIR}/glb.log" 2>&1
    stage_end

    log "Done: ${OUTPUT_DIR}/scene.glb"
elif [[ "$RUN_GLB" -eq 1 ]]; then
    log "Stage 11: skipped (--start-from-stage ${START_FROM_STAGE})"
    log "Done: ${SHAPES_DIR}/mesh.ply"
else
    log "--run_glb not set, skipping GLB bake."
    log "Done: ${SHAPES_DIR}/mesh.ply"
fi

PIPELINE_ELAPSED=$(( $(date +%s) - PIPELINE_T0 ))
log "Pipeline finished for scene '${SCENE_NAME}' (total elapsed $(format_duration "$PIPELINE_ELAPSED"))"
