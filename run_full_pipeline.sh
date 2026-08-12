#!/usr/bin/env bash
# End-to-end pipeline: VGGT-Omega pose/depth prediction -> [optional COB-GS 3D segmentation] -> GenRecon reconstruction -> GLB bake.
#
# Usage:
#   ./run_full_pipeline.sh <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb]
#
# Example:
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --run_glb
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --align-to-gravity --rotate-horizontal-deg 90
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "person,chair,bag"

set -euo pipefail

VGGT_OMEGA_DIR="/home/gabis/Work/GitHub/vggt-omega"
GENRECON_DIR="/home/gabis/Work/GitHub/public-forks-GenRecon"
COBGS_DIR="/home/gabis/Work/GitHub/COB-GS"
VGGT_CHECKPOINT="${VGGT_OMEGA_DIR}/vggt_omega_1b_512.pt"

SIMPLIFY_THRESHOLD=250000
TEXTURE_SIZE=2048
NUM_IMGS_PER_SCENE=32
VGGT_EXPORT_TIMEOUT=1800
RUN_GLB=0
SKIP_FRAMES=-1
ALIGN_TO_GRAVITY=0
ROTATE_HORIZONTAL_DEG=0.0
CLASSES=""

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb]" >&2
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
        --rotate-horizontal-deg) ROTATE_HORIZONTAL_DEG="$2"; shift 2 ;;
        --classes) CLASSES="$2"; shift 2 ;;
        --run_glb) RUN_GLB=1; shift 1 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "Image folder not found: $IMAGE_FOLDER" >&2
    exit 1
fi

RUN_DIR="${GENRECON_DIR}/runs/${SCENE_NAME}"
EXPORT_DIR="${RUN_DIR}/vggt_export"
SEG_DIR="${RUN_DIR}/segmentation"
SCENE_DIR="${RUN_DIR}/scene"
OUTPUT_DIR="${RUN_DIR}/output"

mkdir -p "$EXPORT_DIR" "$OUTPUT_DIR"

# ── Stage 1: VGGT-Omega pose/depth prediction + COLMAP-text export ──
echo "[run_full_pipeline] Stage 1: VGGT-Omega export -> ${EXPORT_DIR}"
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
elapsed=0
while true; do
    if grep -q "Exported COLMAP dataset to" "$VGGT_LOG" 2>/dev/null; then
        echo "[run_full_pipeline] VGGT-Omega export confirmed, detaching viewer process (PID ${VGGT_PID}) and continuing."
        disown "$VGGT_PID" 2>/dev/null || true
        break
    fi
    if ! kill -0 "$VGGT_PID" 2>/dev/null; then
        echo "[run_full_pipeline] VGGT-Omega process exited before export completed. See $VGGT_LOG" >&2
        exit 1
    fi
    if (( elapsed >= VGGT_EXPORT_TIMEOUT )); then
        echo "[run_full_pipeline] Timed out waiting for VGGT-Omega export after ${VGGT_EXPORT_TIMEOUT}s." >&2
        kill "$VGGT_PID" 2>/dev/null || true
        exit 1
    fi
    sleep 2
    elapsed=$((elapsed + 2))
done

if [[ ! -f "${EXPORT_DIR}/sparse/0/cameras.txt" ]]; then
    echo "[run_full_pipeline] Expected COLMAP export not found at ${EXPORT_DIR}/sparse/0/cameras.txt" >&2
    exit 1
fi

# ── Stage 1.5 (optional): COB-GS 3D segmentation ──
# When --classes is set, only the background reaches GenRecon: foreground
# points are dropped from the sparse point cloud (chunk layout may shift as
# a result, since GenRecon derives chunk placement from points3D.txt) and
# foreground pixels are masked out of every RGB frame.
if [[ -n "$CLASSES" ]]; then
    echo "[run_full_pipeline] Stage 1.5: COB-GS segmentation (classes: ${CLASSES})"
    SEG_LOG="${OUTPUT_DIR}/segmentation.log"

    # grounded_sam2_stable_tracking.py hardcodes its input image path per
    # --dataset_type (ignoring --dataset_root), so the export must also live
    # at this fixed location relative to COBGS_DIR.
    COBGS_SCENE_DATASET_DIR="${COBGS_DIR}/dataset/${SCENE_NAME}"
    mkdir -p "$COBGS_SCENE_DATASET_DIR"
    ln -sfn "${EXPORT_DIR}/images" "${COBGS_SCENE_DATASET_DIR}/images"
    mkdir -p "${COBGS_SCENE_DATASET_DIR}/sparse"
    ln -sfn "${EXPORT_DIR}/sparse/0" "${COBGS_SCENE_DATASET_DIR}/sparse/0"

    (
        cd "$COBGS_DIR"
        uv run python -u main_light.py --scene "$SCENE_NAME" --text "$SCENE_NAME" \
            --classes "$CLASSES" --dataset_root dataset --dataset_type tum_rgbd \
            --output_root "${OUTPUT_DIR}/segmentation_raw" --resolution -1 --skip_visualize
    ) > "$SEG_LOG" 2>&1

    COBGS_MASK_DIR="${OUTPUT_DIR}/segmentation_raw/${SCENE_NAME}/masks/${SCENE_NAME}"
    if [[ ! -f "${COBGS_MASK_DIR}/labels.json" ]]; then
        echo "[run_full_pipeline] Expected COB-GS labels.json not found at ${COBGS_MASK_DIR}/labels.json" >&2
        exit 1
    fi

    echo "[run_full_pipeline] Stage 1.5: applying segmentation masks -> ${SEG_DIR}"
    mkdir -p "${SEG_DIR}/masked_rgb" "${SEG_DIR}/filtered_colmap"
    ln -sfn "${EXPORT_DIR}/sparse/0/cameras.txt" "${SEG_DIR}/filtered_colmap/cameras.txt"
    ln -sfn "${EXPORT_DIR}/sparse/0/images.txt" "${SEG_DIR}/filtered_colmap/images.txt"

    cd "$GENRECON_DIR"
    uv run python -u scripts/apply_segmentation_mask.py \
        --images_dir "${EXPORT_DIR}/images" \
        --masks_root "$COBGS_MASK_DIR" \
        --background_ply "${COBGS_MASK_DIR}/ply_pointcloud/background.ply" \
        --out_rgb_dir "${SEG_DIR}/masked_rgb" \
        --out_points3d "${SEG_DIR}/filtered_colmap/points3D.txt" \
        >> "$SEG_LOG" 2>&1
fi

# ── Stage 2: stage GenRecon scene dir ──
echo "[run_full_pipeline] Stage 2: staging ${SCENE_DIR}"
mkdir -p "$SCENE_DIR"
if [[ -n "$CLASSES" ]]; then
    ln -sfn "${SEG_DIR}/masked_rgb" "${SCENE_DIR}/rgb"
    ln -sfn "${SEG_DIR}/filtered_colmap" "${SCENE_DIR}/colmap"
else
    ln -sfn "${EXPORT_DIR}/images" "${SCENE_DIR}/rgb"
    ln -sfn "${EXPORT_DIR}/sparse/0" "${SCENE_DIR}/colmap"
fi

# ── Stage 3: GenRecon reconstruction + GLB bake (exp2 settings) ──
cd "$GENRECON_DIR"
export MPLBACKEND=Agg

echo "[run_full_pipeline] Stage 3: reconstruct_scene.py"
uv run python -u reconstruct_scene.py --mode Iphone --path "$SCENE_DIR" --output_path "$OUTPUT_DIR" \
    --ss_ckpt checkpoints/sparse_structure/ckpts/sparse_structure.pt \
    --shape_ckpt checkpoints/shape_slat/ckpts/shape_slat.pt \
    --tex_ckpt checkpoints/texture_slat/ckpts/texture_slat.pt \
    --num_imgs_per_scene "$NUM_IMGS_PER_SCENE" --colmap_subdir colmap \
    > "${OUTPUT_DIR}/reconstruct.log" 2>&1

if [[ "$RUN_GLB" -eq 1 ]]; then
    echo "[run_full_pipeline] Stage 4: chunked_to_glb.py (simplify_threshold=${SIMPLIFY_THRESHOLD}, texture_size=${TEXTURE_SIZE})"
    uv run python -u chunked_to_glb.py \
        --inputs "${OUTPUT_DIR}/to_glb_inputs.pt" \
        --chunk_inputs "${OUTPUT_DIR}/chunk_inputs.pt" \
        --output_dir "$OUTPUT_DIR" \
        --simplify_threshold "$SIMPLIFY_THRESHOLD" \
        --texture_size "$TEXTURE_SIZE" \
        > "${OUTPUT_DIR}/glb.log" 2>&1

    echo "[run_full_pipeline] Done: ${OUTPUT_DIR}/scene.glb"
else
    echo "[run_full_pipeline] --run_glb not set, skipping GLB bake."
    echo "[run_full_pipeline] Done: ${OUTPUT_DIR}/mesh.ply"
fi
