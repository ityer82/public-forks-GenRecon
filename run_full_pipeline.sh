#!/usr/bin/env bash
# End-to-end pipeline: VGGT-Omega pose/depth prediction -> [optional GroundedSAM2 3D segmentation] -> GenRecon reconstruction -> GLB bake.
# VGGT-Omega and the segmentation stage both now run from in-repo code (vggt/, segmentation/).
#
# Usage:
#   ./run_full_pipeline.sh <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--no-align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb] [--use-trellis (default: on)] [--no-use-trellis] [--mesh-backend trellis2|mvsam3d] [--skip_isaac] [--collision_approximation convexDecomposition|convexHull|boundingCube] [--friction-table-path PATH] [--ollama-model NAME] [--start-from-stage N] [--stop-after-stage N] [--max_chunks_per_group N] [--max_inflated_voxels N] [--depth_conf_thres N] [--depth_edge_rtol N] [--fix_num_chunks N] [--robot-target LABEL] [--skip_floater_removal] [--floater_search_padding_factor N] [--floater_containment_frac N] [--floater_max_faces N] [--vggt-conf-thres N] [--skip_hull_consistency_check] [--pick_place_target LABEL] [--place-offset DX,DY,DZ] [--place-target LABEL] [--place-target-clearance N] [--gripper-open-width N] [--approach-side neg-x|pos-x|neg-y|pos-y]
#
# Note: gravity alignment is ON by default; pass --no-align-to-gravity to disable it.
#
# Note: --pick_place_target LABEL is a fast-path alternative to the full scene reconstruction --
# once segmentation + TRELLIS.2 (Stages 1-3) produce a per-object mesh for every label in --classes,
# it skips the GenRecon scene reconstruction/GLB bake entirely (Stages 4-14) and instead converts
# every --classes object (not just LABEL) + a synthetic ground plane to a minimal physics-ready USD
# scene -- each object independently rescaled/positioned into the shared real-world scene frame, so
# their relative poses match the actual scanned scene -- then runs a Franka pick-and-place demo that
# manipulates only LABEL. Mutually exclusive with --robot-target, which needs the full reconstructed
# scene.
#
# Example:
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --run_glb
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --rotate-horizontal-deg 90
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "person,chair,bag"
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "person,chair,bag" --use-trellis
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "banana" --pick_place_target banana
#   ./run_full_pipeline.sh /home/gabis/Work/GitHub/COB-GS/dataset/food2/images food2_vggt --classes "banana,bowl" --pick_place_target banana --place-target bowl --gripper-open-width 0.08 --approach-side neg-y

set -euo pipefail

GENRECON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRELLIS2_DIR="$(cd "$GENRECON_DIR/../trellis2" && pwd)"
ISAACSIM_DIR="$(cd "$GENRECON_DIR/../IsaacSim" && pwd)"
# MV-SAM3D's source (sam3d_objects/, notebook/, mvsam3d_scripts/, run_inference_weighted.py)
# is vendored in-repo under mv_sam3d/ -- no sibling checkout needed for --mesh-backend mvsam3d.
MVSAM3D_VENDOR_DIR="${GENRECON_DIR}/mv_sam3d"
VGGT_CHECKPOINT="${GENRECON_DIR}/checkpoints/vggt_omega/ckpts/vggt_omega_1b_512.pt"

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
USE_TRELLIS=1
MESH_BACKEND="mvsam3d"
RUN_USD=1
COLLISION_APPROXIMATION="convexDecomposition"
FRICTION_TABLE_PATH="${GENRECON_DIR}/configs/materials/friction_table.example.yaml"
OLLAMA_MODEL="llama3.1:8b"
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
FIX_NUM_CHUNKS=16
ROBOT_TARGET=""
PICK_PLACE_TARGET=""
PLACE_OFFSET="0.3,0.0,0.0"
PLACE_TARGET=""
PLACE_TARGET_CLEARANCE="0.05"
GRIPPER_OPEN_WIDTH="0.06"
APPROACH_SIDE="neg-y"
RUN_FLOATER_REMOVAL=1
SKIP_HULL_CONSISTENCY_CHECK=0
FLOATER_SEARCH_PADDING_FACTOR=0.2
FLOATER_CONTAINMENT_FRAC=0.95
FLOATER_MAX_FACES=5000
VGGT_CONF_THRES="20"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <image_folder> <scene_name> [--simplify_threshold N] [--texture_size N] [--num_imgs_per_scene N] [--skip-frames N] [--no-align-to-gravity] [--rotate-horizontal-deg N] [--classes a,b,c] [--run_glb] [--use-trellis (default: on)] [--no-use-trellis] [--mesh-backend trellis2|mvsam3d] [--skip_isaac] [--collision_approximation convexDecomposition|convexHull|boundingCube] [--friction-table-path PATH] [--ollama-model NAME] [--start-from-stage N] [--stop-after-stage N] [--max_chunks_per_group N] [--max_inflated_voxels N] [--depth_conf_thres N] [--depth_edge_rtol N] [--fix_num_chunks N] [--robot-target LABEL] [--skip_floater_removal] [--floater_search_padding_factor N] [--floater_containment_frac N] [--floater_max_faces N] [--vggt-conf-thres N] [--skip_hull_consistency_check] [--pick_place_target LABEL] [--place-offset DX,DY,DZ] [--place-target LABEL] [--place-target-clearance N] [--gripper-open-width N] [--approach-side neg-x|pos-x|neg-y|pos-y]" >&2
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
        --no-align-to-gravity) ALIGN_TO_GRAVITY=0; shift 1 ;;
        --rotate-horizontal-deg) ROTATE_HORIZONTAL_DEG="$2"; shift 2 ;;
        --classes) CLASSES="$2"; shift 2 ;;
        --run_glb) RUN_GLB=1; shift 1 ;;
        --use-trellis) USE_TRELLIS=1; shift 1 ;;
        --no-use-trellis) USE_TRELLIS=0; shift 1 ;;
        --mesh-backend) MESH_BACKEND="$2"; shift 2 ;;
        --skip_isaac) RUN_USD=0; shift 1 ;;
        --collision_approximation) COLLISION_APPROXIMATION="$2"; shift 2 ;;
        --friction-table-path) FRICTION_TABLE_PATH="$2"; shift 2 ;;
        --ollama-model) OLLAMA_MODEL="$2"; shift 2 ;;
        --start-from-stage) START_FROM_STAGE="$2"; shift 2 ;;
        --stop-after-stage) STOP_AFTER_STAGE="$2"; shift 2 ;;
        --max_chunks_per_group) MAX_CHUNKS_PER_GROUP="$2"; shift 2 ;;
        --max_inflated_voxels) MAX_INFLATED_VOXELS="$2"; shift 2 ;;
        --depth_conf_thres) DEPTH_CONF_THRES="$2"; shift 2 ;;
        --depth_edge_rtol) DEPTH_EDGE_RTOL="$2"; shift 2 ;;
        --fix_num_chunks) FIX_NUM_CHUNKS="$2"; shift 2 ;;
        --robot-target) ROBOT_TARGET="$2"; shift 2 ;;
        --pick_place_target) PICK_PLACE_TARGET="$2"; shift 2 ;;
        --place-offset) PLACE_OFFSET="$2"; shift 2 ;;
        --place-target) PLACE_TARGET="$2"; shift 2 ;;
        --place-target-clearance) PLACE_TARGET_CLEARANCE="$2"; shift 2 ;;
        --gripper-open-width) GRIPPER_OPEN_WIDTH="$2"; shift 2 ;;
        --approach-side) APPROACH_SIDE="$2"; shift 2 ;;
        --skip_floater_removal) RUN_FLOATER_REMOVAL=0; shift 1 ;;
        --skip_hull_consistency_check) SKIP_HULL_CONSISTENCY_CHECK=1; shift 1 ;;
        --floater_search_padding_factor) FLOATER_SEARCH_PADDING_FACTOR="$2"; shift 2 ;;
        --floater_containment_frac) FLOATER_CONTAINMENT_FRAC="$2"; shift 2 ;;
        --floater_max_faces) FLOATER_MAX_FACES="$2"; shift 2 ;;
        --vggt-conf-thres) VGGT_CONF_THRES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$IMAGE_FOLDER" ]]; then
    echo "Image folder not found: $IMAGE_FOLDER" >&2
    exit 1
fi

# TRELLIS.2 is per-class, so it's a no-op without --classes -- USE_TRELLIS
# defaults to on, which would otherwise break every whole-scene (no
# --classes) invocation, so silently drop it here instead of erroring.
if [[ "$USE_TRELLIS" -eq 1 && -z "$CLASSES" ]]; then
    echo "Note: --classes not set, so per-class mesh reconstruction (enabled by default) does not apply to this run." >&2
    USE_TRELLIS=0
fi

if [[ "$MESH_BACKEND" != "trellis2" && "$MESH_BACKEND" != "mvsam3d" ]]; then
    echo "--mesh-backend must be 'trellis2' or 'mvsam3d', got '${MESH_BACKEND}'." >&2
    exit 1
fi
if [[ "$MESH_BACKEND" == "mvsam3d" && ! -d "$MVSAM3D_VENDOR_DIR" ]]; then
    echo "--mesh-backend mvsam3d requires the vendored MV-SAM3D source at ${MVSAM3D_VENDOR_DIR}." >&2
    exit 1
fi

# --pick_place_target is a fast-path alternative to the full scene reconstruction (see Stage P1-P4
# below): it needs a TRELLIS.2 per-object mesh for the target class, and has no meaning alongside
# --robot-target, which drives a mobile robot into an object inside the *full* reconstructed scene.
if [[ -n "$PICK_PLACE_TARGET" ]]; then
    if [[ -n "$ROBOT_TARGET" ]]; then
        echo "--pick_place_target and --robot-target are mutually exclusive (one skips the full scene reconstruction, the other requires it)." >&2
        exit 1
    fi
    if [[ -z "$CLASSES" ]]; then
        echo "--pick_place_target requires --classes to include the same label." >&2
        exit 1
    fi
    IFS=',' read -ra PICK_PLACE_CLASSES_CHECK <<< "$CLASSES"
    PICK_PLACE_TARGET_FOUND=0
    for c in "${PICK_PLACE_CLASSES_CHECK[@]}"; do
        [[ "$c" == "$PICK_PLACE_TARGET" ]] && PICK_PLACE_TARGET_FOUND=1
    done
    if [[ "$PICK_PLACE_TARGET_FOUND" -eq 0 ]]; then
        echo "--pick_place_target '${PICK_PLACE_TARGET}' must exactly match one of the labels passed to --classes ('${CLASSES}')." >&2
        exit 1
    fi
    if [[ "$USE_TRELLIS" -eq 0 ]]; then
        echo "Note: --pick_place_target requires TRELLIS.2's per-object mesh; overriding --no-use-trellis to on for this run." >&2
        USE_TRELLIS=1
    fi
fi

# --place-target places the picked object above another --classes object (e.g. a bowl) instead of
# at a fixed --place-offset -- it only makes sense alongside --pick_place_target, and the place
# target must itself be one of --classes (so Stage P1-P3 produce a physics-ready mesh for it too).
if [[ -n "$PLACE_TARGET" ]]; then
    if [[ -z "$PICK_PLACE_TARGET" ]]; then
        echo "--place-target requires --pick_place_target to be set." >&2
        exit 1
    fi
    if [[ "$PLACE_TARGET" == "$PICK_PLACE_TARGET" ]]; then
        echo "--place-target must differ from --pick_place_target ('${PICK_PLACE_TARGET}')." >&2
        exit 1
    fi
    IFS=',' read -ra PLACE_TARGET_CLASSES_CHECK <<< "$CLASSES"
    PLACE_TARGET_FOUND=0
    for c in "${PLACE_TARGET_CLASSES_CHECK[@]}"; do
        [[ "$c" == "$PLACE_TARGET" ]] && PLACE_TARGET_FOUND=1
    done
    if [[ "$PLACE_TARGET_FOUND" -eq 0 ]]; then
        echo "--place-target '${PLACE_TARGET}' must exactly match one of the labels passed to --classes ('${CLASSES}')." >&2
        exit 1
    fi
fi

if [[ "$APPROACH_SIDE" != "neg-x" && "$APPROACH_SIDE" != "pos-x" && "$APPROACH_SIDE" != "neg-y" && "$APPROACH_SIDE" != "pos-y" ]]; then
    echo "--approach-side must be one of neg-x, pos-x, neg-y, pos-y, got '${APPROACH_SIDE}'." >&2
    exit 1
fi

RUN_DIR="${GENRECON_DIR}/runs/${SCENE_NAME}"
EXPORT_DIR="${RUN_DIR}/vggt_export"
SCENE_DIR="${RUN_DIR}/genrecon_input"
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

# Appends `flag value` to the named array, but only when value is non-empty --
# collapses the repeated `if [[ -n "$X" ]]; then ARR+=(--flag "$X"); fi` idiom
# used throughout the per-stage argument building below.
opt_arg() {
    local -n _arr="$1"
    local flag="$2" value="$3"
    [[ -n "$value" ]] && _arr+=("$flag" "$value")
    return 0
}

declare -A LOG_LINE_OFFSET

# Appends whatever's been newly written to $1 since the last call for that
# file into pipeline.log. Needed because external subprocesses (the
# segmentation stage, VGGT-Omega, TRELLIS2, IsaacSim) don't log through the shared loguru logger,
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

# Runs an in-repo python stage (uv run python -u <script> args...), logging a
# replayable debugpy launch config first and mirroring its output into the
# unified pipeline log afterwards. `redir` is "new" to truncate log_file or
# "append" to keep appending to a log file shared with other stages.
run_py_step() {
    local name="$1" script="$2" log_file="$3" redir="$4"; shift 4
    log_debug_config "$name" "${GENRECON_DIR}/${script}" "$GENRECON_DIR" "$@"
    if [[ "$redir" == append ]]; then
        uv run python -u "$script" "$@" >> "$log_file" 2>&1
    else
        uv run python -u "$script" "$@" > "$log_file" 2>&1
    fi
    mirror_log "$log_file"
}

# Same as run_py_step, but for stages that live in a sibling repo (TRELLIS.2,
# IsaacSim) and run from that repo's own directory with its own `uv run`
# invocation (e.g. "uv run --no-sync generate.py", "uv run convert_asset.py").
run_external_step() {
    local name="$1" script="$2" cwd="$3" log_file="$4" redir="$5" runner="$6"; shift 6
    log_debug_config "$name" "$script" "$cwd" "$@"
    if [[ "$redir" == append ]]; then
        ( cd "$cwd" && $runner "$@" ) >> "$log_file" 2>&1
    else
        ( cd "$cwd" && $runner "$@" ) > "$log_file" 2>&1
    fi
    mirror_log "$log_file"
}

# Like run_py_step, but cd's into a vendored subdir first -- needed only for
# mv_sam3d/run_inference_weighted.py, which resolves checkpoints/ and
# visualization/ relative to CWD rather than __file__. Still uses genrecon's
# own shared root .venv (uv run from a subdirectory resolves to the same
# project), unlike run_external_step's separate --no-sync sibling-repo venv.
run_py_step_in_dir() {
    local name="$1" script="$2" cwd="$3" log_file="$4" redir="$5"; shift 5
    log_debug_config "$name" "${cwd}/${script}" "$cwd" "$@"
    if [[ "$redir" == append ]]; then
        ( cd "$cwd" && uv run python -u "$script" "$@" ) >> "$log_file" 2>&1
    else
        ( cd "$cwd" && uv run python -u "$script" "$@" ) > "$log_file" 2>&1
    fi
    mirror_log "$log_file"
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

    # --vggt-conf-thres passes through to demo_rerun.py's --conf-thres, which
    # governs the confidence-percentile filter applied to BOTH the exported
    # points3D.txt/.ply and the rerun-viewer preview. demo_rerun.py's own
    # CLI default (50.0) discards half of every exported point cloud by
    # construction (it's a percentile, not an absolute threshold), and since
    # low confidence skews toward scene periphery/boundaries, that
    # disproportionately shrinks spatial coverage rather than just density.
    # Default here is lowered to 20.0 (matching visual_util.filter_points's
    # own default) after an A/B export on the same scene showed it roughly
    # doubles the exported point cloud's bounding-box volume; raise it back
    # toward 50 if the looser threshold lets in too much noise.
    VGGT_CONF_THRES_ARGS=()
    opt_arg VGGT_CONF_THRES_ARGS --conf-thres "$VGGT_CONF_THRES"

    log_debug_config "stage0_vggt_export" "${GENRECON_DIR}/vggt/demo_rerun.py" "$GENRECON_DIR" \
        "$IMAGE_FOLDER" --checkpoint "$VGGT_CHECKPOINT" --export-for-3dgs "$EXPORT_DIR" \
        "${VGGT_SKIP_FRAMES_ARGS[@]}" "${VGGT_GRAVITY_ARGS[@]}" "${VGGT_CONF_THRES_ARGS[@]}"

    (
        uv run python -u vggt/demo_rerun.py "$IMAGE_FOLDER" \
            --checkpoint "$VGGT_CHECKPOINT" \
            --export-for-3dgs "$EXPORT_DIR" \
            "${VGGT_SKIP_FRAMES_ARGS[@]}" \
            "${VGGT_GRAVITY_ARGS[@]}" \
            "${VGGT_CONF_THRES_ARGS[@]}"
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

# ── Stage 1 (optional): GroundedSAM2-based 3D segmentation ──
# GenRecon always conditions on VGGT-Omega's original images and point cloud
# (staged unmodified in Stage 4) -- class exclusion happens later, at
# generation time, via Stage 5's --exclude_masks_root voxel carving, which
# reads this stage's per-class point clouds/masks directly rather than
# requiring pre-masked/filtered input.
if [[ -n "$CLASSES" ]]; then
    SEG_LOG="${OUTPUT_DIR}/segmentation.log"
    COBGS_MASK_DIR="${OUTPUT_DIR}/segmentation_raw/masks/classes"

    if [[ "$START_FROM_STAGE" -le 1 ]]; then
        stage_start "Stage 1: segmentation (classes: ${CLASSES})"

        # --output_root is already scene-specific (it's under runs/<scene>/), and --text
        # is only ever a directory label here (--classes is always set, so it never
        # feeds the detection caption) -- so both --flat_output and a fixed --text
        # avoid redundantly repeating the scene name in the output path.
        HULL_CONSISTENCY_ARGS=()
        if [[ "$SKIP_HULL_CONSISTENCY_CHECK" -eq 1 ]]; then
            HULL_CONSISTENCY_ARGS+=(--skip_hull_consistency_check)
        fi

        SEG_ARGS=(--scene "$SCENE_NAME" --text "classes" \
            --classes "$CLASSES" --dataset_root "$EXPORT_DIR" \
            --output_root "${OUTPUT_DIR}/segmentation_raw" --resolution -1 \
            --flat_output --depth_dir "${EXPORT_DIR}/depth" \
            --depth_conf_thres "$DEPTH_CONF_THRES" --depth_edge_rtol "$DEPTH_EDGE_RTOL" \
            "${HULL_CONSISTENCY_ARGS[@]}")
        run_py_step "stage1_segmentation" "segmentation/main_light.py" "$SEG_LOG" new "${SEG_ARGS[@]}"

        if [[ ! -f "${COBGS_MASK_DIR}/labels.json" ]]; then
            log "Expected segmentation labels.json not found at ${COBGS_MASK_DIR}/labels.json" >&2
            exit 1
        fi
        stage_end
    else
        log "Stage 1: skipped (--start-from-stage ${START_FROM_STAGE}), assuming existing segmentation at ${OUTPUT_DIR}/segmentation_raw"
    fi
    check_stop_after_stage 1

    # ── Stage 2: RGBA best-view mask export ──
    # For each class, picks the frame where the object covers the largest
    # fraction of the image (most detail) and composites an RGBA image from
    # it, for use by image-conditioned generators (e.g. TRELLIS.2 generate.py).
    if [[ "$START_FROM_STAGE" -le 2 ]]; then
        stage_start "Stage 2: RGBA mask export -> ${COBGS_MASK_DIR}/<class>/mask_rgba"
        run_py_step "stage2_export_rgba_masks" "scripts/export_rgba_masks.py" "$SEG_LOG" append \
            --images_dir "${EXPORT_DIR}/images" \
            --masks_root "$COBGS_MASK_DIR"
        stage_end
    else
        log "Stage 2: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 2

    # ── Stage 3 (optional): per-class 3D reconstruction (TRELLIS.2 or MV-SAM3D) ──
    # Output convention is shared regardless of backend: ${RUN_DIR}/image_to_3d_meshes/<label>/mesh.glb
    # -- so every downstream consumer (Stage 10 substitution, Stage 14 organize_final_objects,
    # the pick-place fast path P1-P4) needs no backend-specific handling.
    IMAGE_TO_3D_OUTPUT_DIR="${RUN_DIR}/image_to_3d_meshes"
    if [[ "$USE_TRELLIS" -eq 1 && "$START_FROM_STAGE" -le 3 ]]; then
        if [[ "$MESH_BACKEND" == "trellis2" ]]; then
            stage_start "Stage 3: TRELLIS.2 reconstruction -> ${IMAGE_TO_3D_OUTPUT_DIR}"
            TRELLIS2_INPUT_DIR="${RUN_DIR}/trellis2_input"
            run_py_step "stage3_stage_trellis2_inputs" "scripts/stage_trellis2_inputs.py" "$SEG_LOG" append \
                --masks_root "$COBGS_MASK_DIR" \
                --out_dir "$TRELLIS2_INPUT_DIR"

            run_external_step "stage3_trellis2_generate" "${TRELLIS2_DIR}/generate.py" "$TRELLIS2_DIR" \
                "$SEG_LOG" append "uv run --no-sync generate.py" \
                --input "$TRELLIS2_INPUT_DIR" \
                --output-dir "$IMAGE_TO_3D_OUTPUT_DIR" \
                --resolution 512 --no-preview
        else
            # MV-SAM3D backend: bridge genrecon's own VGGT depth/poses + Stage 1's
            # per-class masks directly (no DA3, no Stage 2 RGBA export needed), run
            # MV-SAM3D's multi-object inference (vendored in-repo under mv_sam3d/,
            # sharing genrecon's own uv venv), then transform each object's
            # canonical-space mesh into genrecon's world frame and drop it at the
            # same path TRELLIS.2 would have used.
            stage_start "Stage 3: MV-SAM3D reconstruction -> ${IMAGE_TO_3D_OUTPUT_DIR}"
            # basename must be scene-specific: run_inference_weighted.py derives its
            # visualization/<dataset_name>/... output dir from --input_path's basename
            # alone, so a generic name here would collide across different scenes.
            MVSAM3D_INPUT_DIR="${RUN_DIR}/${SCENE_NAME}_mvsam3d_input"
            MVSAM3D_DATASET_NAME="$(basename "$MVSAM3D_INPUT_DIR")"

            run_py_step "stage3_import_from_genrecon" "mv_sam3d/mvsam3d_scripts/import_from_genrecon.py" \
                "$SEG_LOG" append \
                --genrecon_run "$RUN_DIR" \
                --output_dir "$MVSAM3D_INPUT_DIR" \
                --objects "$CLASSES"

            run_py_step_in_dir "stage3_mvsam3d_inference" "run_inference_weighted.py" "$MVSAM3D_VENDOR_DIR" \
                "$SEG_LOG" append \
                --input_path "$MVSAM3D_INPUT_DIR" \
                --mask_prompt "$CLASSES" \
                --da3_output "${MVSAM3D_INPUT_DIR}/da3_output.npz"

            run_py_step "stage3_collect_mvsam3d_outputs" "mv_sam3d/mvsam3d_scripts/collect_mvsam3d_outputs.py" \
                "$SEG_LOG" append \
                --dataset_name "$MVSAM3D_DATASET_NAME" \
                --labels "$CLASSES" \
                --out_dir "$IMAGE_TO_3D_OUTPUT_DIR" \
                --visualization_dir "${MVSAM3D_VENDOR_DIR}/visualization"
        fi
        stage_end
    else
        log "Stage 3: skipped (no --use-trellis, or --start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 3
fi

# ── Stage P1-P4 (fast path, mutually exclusive with the rest of the pipeline): Franka
# pick-and-place demo from the segmented object alone, skipping GenRecon scene reconstruction
# entirely. Only TRELLIS.2's per-object mesh (Stage 3, already produced above) is needed -- no
# background reconstruction, GLB bake, or per-object mesh cropping (Stages 4-14) applies here,
# since there is no reconstructed scene to crop from.
if [[ -n "$PICK_PLACE_TARGET" ]]; then
    PICK_PLACE_DIR="${RUN_DIR}/pick_place"
    PICK_PLACE_GLB_DIR="${PICK_PLACE_DIR}/glb"
    PICK_PLACE_SCENE_USDA="${PICK_PLACE_GLB_DIR}/scene.usda"

    # TRELLIS.2's mesh.glb is in its own object-centric canonical frame (longest bbox axis
    # normalized to ~1.0, see align_trellis2_mesh_to_scene.py's docstring) -- it carries no
    # real-world scale at all. The full pipeline's Stage 10 recovers scale/position by aligning
    # against the object's real crop out of the reconstructed scene mesh (shapes/<label>_mesh.ply),
    # which doesn't exist in this fast path (no scene reconstruction runs). Stage 1's segmented
    # point cloud (COBGS_MASK_DIR/<label>/point_cloud/<label>.ply) is already in that same metric
    # scene frame -- align_trellis2_mesh_to_scene.py's target-bbox loader only reads vertex
    # positions, so it works identically as a scale/translation reference whether the PLY is a
    # mesh or a raw point cloud.
    #
    # Every --classes label is aligned here, not just PICK_PLACE_TARGET: each label's TRELLIS.2
    # mesh gets independently rescaled/translated into the same shared metric scene frame (via its
    # own segmented point cloud as reference), so once composed (Stage P3) their relative poses
    # match the real scanned scene -- compose_isaac_scene.py reads no separate pose file, it derives
    # each asset's world position purely from the pivot baked in by convert_asset.py, so getting
    # every object into this shared frame here is the only thing that makes that work. "background"
    # is never one of --classes, so it's naturally excluded; there's no reconstructed background
    # asset in this fast path to include even if it were.
    stage_start "Stage P1: aligning TRELLIS.2 meshes for all --classes labels to scene scale -> ${PICK_PLACE_GLB_DIR}"
    IFS=',' read -ra PICK_PLACE_CLASSES <<< "$CLASSES"
    pick_place_first_label=1
    for label in "${PICK_PLACE_CLASSES[@]}"; do
        # Mirrors Stage 10's identical sanitization (spaces/slashes -> underscores): image_to_3d_meshes/
        # (staged by stage_trellis2_inputs.py) keeps the raw --classes spelling, but COBGS_MASK_DIR
        # (segmentation_raw/masks/classes, via labels.json's sanitize_label) and the scene-side glb
        # directory (and everything Stage P2/P3 derive from it, including the composed prim name)
        # both use the sanitized spelling -- a multi-word label (e.g. "white chair") must use each
        # spelling against the artifact that actually uses it, or this lookup silently target the
        # wrong (nonexistent) path.
        sanitized_label="${label// /_}"
        sanitized_label="${sanitized_label//\//_}"
        mkdir -p "${PICK_PLACE_GLB_DIR}/${sanitized_label}"
        TRELLIS_LABEL_GLB="${RUN_DIR}/image_to_3d_meshes/${label}/mesh.glb"
        LABEL_SCALE_REF_PLY="${COBGS_MASK_DIR}/${sanitized_label}/point_cloud/${sanitized_label}.ply"
        if [[ ! -f "$TRELLIS_LABEL_GLB" ]]; then
            log "Expected TRELLIS.2 mesh not found at ${TRELLIS_LABEL_GLB}" >&2
            exit 1
        fi
        if [[ ! -f "$LABEL_SCALE_REF_PLY" ]]; then
            log "Expected segmented point cloud not found at ${LABEL_SCALE_REF_PLY}" >&2
            exit 1
        fi
        pick_place_redir="append"
        [[ "$pick_place_first_label" -eq 1 ]] && pick_place_redir="new"
        ALIGN_ZUP_ARGS=()
        [[ "$MESH_BACKEND" == "mvsam3d" ]] && ALIGN_ZUP_ARGS=(--no-zup-correction)
        run_py_step "stageP1_align_trellis2_mesh_${sanitized_label}" "scripts/align_trellis2_mesh_to_scene.py" \
            "${PICK_PLACE_DIR}/align_trellis2_mesh.log" "$pick_place_redir" \
            --trellis_glb "$TRELLIS_LABEL_GLB" \
            --scene_mesh_ply "$LABEL_SCALE_REF_PLY" \
            --out_glb "${PICK_PLACE_GLB_DIR}/${sanitized_label}/mesh.glb" \
            "${ALIGN_ZUP_ARGS[@]}"
        pick_place_first_label=0
    done
    stage_end

    stage_start "Stage P2: convert_asset.py (collision_approximation=${COLLISION_APPROXIMATION}) -> ${PICK_PLACE_GLB_DIR}/<label>/asset.usd"
    run_external_step "stageP2_convert_asset" "${ISAACSIM_DIR}/convert_asset.py" "$ISAACSIM_DIR" \
        "${PICK_PLACE_DIR}/convert_asset.log" new "uv run convert_asset.py" \
        --input "$PICK_PLACE_GLB_DIR" \
        --collision-approximation "$COLLISION_APPROXIMATION"
    stage_end

    stage_start "Stage P3: compose_isaac_scene.py -> ${PICK_PLACE_SCENE_USDA}"
    # No --background-label/--floor-label passed: neither asset exists in this fast path (there's no
    # reconstructed background/floor mesh, by design -- see Stage P1's comment), and both flags just
    # fail their (harmless) label lookups when absent -- compose_isaac_scene.py still composes every
    # object plus its default synthetic ground plane, sized to sit under the lowest of all of them.
    run_external_step "stageP3_compose_isaac_scene" "${ISAACSIM_DIR}/compose_isaac_scene.py" "$ISAACSIM_DIR" \
        "${PICK_PLACE_DIR}/compose_isaac_scene.log" new "uv run compose_isaac_scene.py" \
        --input "$PICK_PLACE_GLB_DIR" \
        --output "$PICK_PLACE_SCENE_USDA"
    stage_end

    if [[ -n "$PLACE_TARGET" ]]; then
        stage_start "Stage P4: demo_franka_pickplace.py (pick_target=${PICK_PLACE_TARGET}, place_target=${PLACE_TARGET}) -> ${PICK_PLACE_DIR}/pick_place.mp4"
        PLACE_ARGS=(--place-target "$PLACE_TARGET" --place-target-clearance "$PLACE_TARGET_CLEARANCE")
    else
        stage_start "Stage P4: demo_franka_pickplace.py (pick_target=${PICK_PLACE_TARGET}) -> ${PICK_PLACE_DIR}/pick_place.mp4"
        IFS=',' read -ra PLACE_OFFSET_ARGS <<< "$PLACE_OFFSET"
        PLACE_ARGS=(--place-offset "${PLACE_OFFSET_ARGS[@]}")
    fi
    run_external_step "stageP4_demo_franka_pickplace" "${ISAACSIM_DIR}/demo_franka_pickplace.py" "$ISAACSIM_DIR" \
        "${PICK_PLACE_DIR}/pick_place.log" new "uv run demo_franka_pickplace.py" \
        --scene "$PICK_PLACE_SCENE_USDA" \
        --pick-target "$PICK_PLACE_TARGET" \
        "${PLACE_ARGS[@]}" \
        --gripper-open-width "$GRIPPER_OPEN_WIDTH" \
        --approach-side "$APPROACH_SIDE" \
        --output "${PICK_PLACE_DIR}/pick_place.mp4" \
        --stage-output "${PICK_PLACE_DIR}/pick_place_scene.usda"
    stage_end

    PIPELINE_ELAPSED=$(( $(date +%s) - PIPELINE_T0 ))
    log "Pipeline finished for scene '${SCENE_NAME}' (pick-and-place fast path, total elapsed $(format_duration "$PIPELINE_ELAPSED"))"
    log "Done: ${PICK_PLACE_DIR}/pick_place.mp4"
    exit 0
fi

# ── Stage 4: stage GenRecon scene dir ──
# Always points at the original VGGT-Omega export (full images + full point
# cloud) -- both Stage 5 reconstruct_scene.py passes (primary excluded run
# and secondary unexcluded run for real per-object meshes) read this same
# dir, differing only in whether --exclude_masks_root is applied.
if [[ "$START_FROM_STAGE" -le 4 ]]; then
    stage_start "Stage 4: staging ${SCENE_DIR}"
    mkdir -p "$SCENE_DIR"
    ln -sfn "${EXPORT_DIR}/images" "${SCENE_DIR}/rgb"
    ln -sfn "${EXPORT_DIR}/sparse/0" "${SCENE_DIR}/colmap"
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
    opt_arg RECON_VRAM_ARGS --max_chunks_per_group "$MAX_CHUNKS_PER_GROUP"
    opt_arg RECON_VRAM_ARGS --max_inflated_voxels "$MAX_INFLATED_VOXELS"
    opt_arg RECON_VRAM_ARGS --fix_num_chunks "$FIX_NUM_CHUNKS"

    # Excludes each segmented class's object from generation itself (voxel-level
    # carving before shape/texture SLat sampling) -- GenRecon conditions on the
    # original, unmasked images/point cloud (SCENE_DIR), and this carving is
    # what keeps it from "resurrecting" plausible geometry into the excluded
    # region, rather than relying on pre-masked input. Also runs a second,
    # unexcluded reconstruction over the same SCENE_DIR (--unmasked_path),
    # sharing the primary run's chunk geometry/world frame, so Stage 8 has a
    # real (non-hallucinated) source mesh to crop each object from -- the
    # primary run's mesh.ply no longer contains them.
    RECON_EXCLUDE_ARGS=()
    if [[ -n "$CLASSES" ]]; then
        RECON_EXCLUDE_ARGS=(--exclude_masks_root "$COBGS_MASK_DIR" --unmasked_path "$SCENE_DIR")
    fi

    RECON_ARGS=(--mode Iphone --path "$SCENE_DIR" --output_path "$OUTPUT_DIR" \
        --ss_ckpt checkpoints/sparse_structure/ckpts/sparse_structure.pt \
        --shape_ckpt checkpoints/shape_slat/ckpts/shape_slat.pt \
        --tex_ckpt checkpoints/texture_slat/ckpts/texture_slat.pt \
        --num_imgs_per_scene "$NUM_IMGS_PER_SCENE" --colmap_subdir colmap \
        "${RECON_VRAM_ARGS[@]}" "${RECON_EXCLUDE_ARGS[@]}")
    run_py_step "stage5_reconstruct_scene" "reconstruct_scene.py" "${OUTPUT_DIR}/reconstruct.log" new "${RECON_ARGS[@]}"
    stage_end
else
    log "Stage 5: skipped (--start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 5

# ── Stage 6: reprojection validation ──
# Renders novel views against the scene BEFORE foreground-object removal:
# when --classes is set, Stage 5's secondary unexcluded run wrote
# object_source_mesh.ply (same chunk geometry/world frame as mesh.ply, but
# without the voxel-level carving of segmented objects), so that's used
# here instead of the excluded mesh.ply. Without --classes there's no
# exclusion to begin with, so mesh.ply already is the pre-removal scene.
if [[ -n "$CLASSES" ]]; then
    REPROJECTION_MESH_PLY="${OUTPUT_DIR}/object_source_mesh.ply"
else
    REPROJECTION_MESH_PLY="${OUTPUT_DIR}/mesh.ply"
fi
if [[ "$START_FROM_STAGE" -le 6 ]]; then
    stage_start "Stage 6: render_reprojection_validation.py"
    run_py_step "stage6_render_reprojection_validation" "scripts/render_reprojection_validation.py" \
        "${OUTPUT_DIR}/reprojection_validation.log" new \
        --mesh_ply "$REPROJECTION_MESH_PLY" \
        --colmap_dir "${SCENE_DIR}/colmap" \
        --images_dir "${SCENE_DIR}/rgb" \
        --out_synth_dir "${OUTPUT_DIR}/synth_views" \
        --out_compare_dir "${OUTPUT_DIR}/compare_views"
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
#
# Immediately after each class's crop, remove_floater_mesh.py does a second
# cascading pass over background_mesh.ply: it builds a deliberately enlarged
# convex hull around that class's point cloud (padding scaled to the
# object's own bbox size via --search_padding_factor) and strips out any
# small disconnected mesh component (capped at --max_floater_faces) whose
# vertices are almost entirely inside that hull -- these are floaters left
# behind by the tight object crop (e.g. hallucinated double-walled skin).
# The single largest connected component is always kept untouched. Disabled
# via --skip_floater_removal.
OBJECT_SOURCE_MESH="${OUTPUT_DIR}/object_source_mesh.ply"
if [[ -n "$CLASSES" && "$START_FROM_STAGE" -le 8 ]]; then
    stage_start "Stage 8: cascading per-object mesh extraction -> ${SHAPES_DIR}"
    BACKGROUND_MESH="${SHAPES_DIR}/background_mesh.ply"
    cp "${SHAPES_DIR}/mesh.ply" "$BACKGROUND_MESH"
    for obj_ply in "$SHAPES_DIR"/*.ply; do
        obj_name="$(basename "$obj_ply")"
        # Skip mesh.ply/background.ply plus anything matching this stage's own
        # <label>_mesh.ply / <label>_floaters.ply output naming -- a stale one
        # from a previous run (Stage 7 doesn't wipe SHAPES_DIR before re-copying
        # raw point clouds) would otherwise get globbed as a brand-new "object"
        # here, producing a spurious <label>_mesh_mesh.ply crop and a duplicate
        # USD asset later.
        [[ "$obj_name" == "mesh.ply" || "$obj_name" == "background.ply" || "$obj_name" == *_mesh.ply || "$obj_name" == *_floaters.ply ]] && continue
        label="${obj_name%.ply}"
        run_py_step "stage8_extract_object_mesh_${label}" "scripts/extract_object_mesh.py" \
            "${OUTPUT_DIR}/reconstruct.log" append \
            --mesh_ply "$BACKGROUND_MESH" \
            --object_mesh_ply "$OBJECT_SOURCE_MESH" \
            --object_ply "$obj_ply" \
            --out_ply "${SHAPES_DIR}/${label}_mesh.ply" \
            --remainder_out_ply "$BACKGROUND_MESH" \
            --colmap_dir "${SCENE_DIR}/colmap" \
            --masks_dir "${COBGS_MASK_DIR}/${label}/mask_bin"

        if [[ "$RUN_FLOATER_REMOVAL" -eq 1 ]]; then
            run_py_step "stage8_remove_floater_mesh_${label}" "scripts/remove_floater_mesh.py" \
                "${OUTPUT_DIR}/reconstruct.log" append \
                --mesh_ply "$BACKGROUND_MESH" \
                --object_ply "$obj_ply" \
                --out_ply "$BACKGROUND_MESH" \
                --floaters_out_ply "${SHAPES_DIR}/${label}_floaters.ply" \
                --search_padding_factor "$FLOATER_SEARCH_PADDING_FACTOR" \
                --containment_frac "$FLOATER_CONTAINMENT_FRAC" \
                --max_floater_faces "$FLOATER_MAX_FACES"
        fi
    done
    stage_end
else
    log "Stage 8: skipped (no --classes, or --start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 8

# ── Stage 9 (optional): floor segmentation ──
# Splits the flat floor region out of background_mesh.ply into its own asset
# (floor_mesh.ply) so Stage 11 (convert_asset.py) can give it an analytic-plane
# collider -- the same mechanism that makes the synthetic ground-plane safety
# net reliable -- instead of the rest of the background's decimated
# meshSimplification triangle-mesh proxy, which is what lets objects fall
# through the visible floor. Only meaningful once a background_mesh.ply exists
# (i.e. --classes was set); background_mesh.ply is overwritten in place with
# the floor removed, chaining the same way Stage 8's per-class crops do.
if [[ -n "$CLASSES" && "$START_FROM_STAGE" -le 9 ]]; then
    stage_start "Stage 9: extract_floor_mesh.py -> ${SHAPES_DIR}/floor_mesh.ply"
    run_py_step "stage9_extract_floor_mesh" "scripts/extract_floor_mesh.py" \
        "${OUTPUT_DIR}/reconstruct.log" append \
        --mesh_ply "${SHAPES_DIR}/background_mesh.ply" \
        --out_ply "${SHAPES_DIR}/floor_mesh.ply" \
        --remainder_out_ply "${SHAPES_DIR}/background_mesh.ply"

    # Friction inference: a LangGraph agent (local Ollama LLM) maps each detected class
    # label to a mu(material, floor) friction coefficient looked up from
    # --friction-table-path's YAML reference table, so Stage 11 (convert_asset.py) can
    # author true per-object friction instead of one global value for every asset.
    stage_start "Stage 9: infer_friction_assignments.py -> ${SHAPES_DIR}/friction_assignments.json"
    run_py_step "stage9_infer_friction_assignments" "scripts/infer_friction_assignments.py" \
        "${OUTPUT_DIR}/reconstruct.log" append \
        --labels_json "${COBGS_MASK_DIR}/labels.json" \
        --friction_table "$FRICTION_TABLE_PATH" \
        --out_json "${SHAPES_DIR}/friction_assignments.json" \
        --ollama_model "$OLLAMA_MODEL"
    stage_end
else
    log "Stage 9: skipped (no --classes, or --start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 9

# ── Stage 10/11/12 (optional): per-object mesh -> GLB -> physics-ready USD -> composed scene ──
# Only the *_mesh.ply crops from Stage 8/9 are converted (never the whole-scene
# mesh.ply, and never the plain per-class point clouds like chair.ply/background.ply).
if [[ "$RUN_USD" -eq 1 ]]; then
    if [[ "$START_FROM_STAGE" -le 10 ]]; then
        if [[ "$USE_TRELLIS" -eq 1 ]]; then
            stage_start "Stage 10: mesh_to_glb.py -> ${SHAPES_DIR}/glb (+ TRELLIS.2 substitution)"
        else
            stage_start "Stage 10: mesh_to_glb.py -> ${SHAPES_DIR}/glb"
        fi
        run_py_step "stage10_mesh_to_glb" "scripts/mesh_to_glb.py" "${OUTPUT_DIR}/mesh_to_glb.log" new \
            --shapes_dir "$SHAPES_DIR" \
            --out_dir "${SHAPES_DIR}/glb"

        # --use-trellis: swap the crop-based glb produced above for the
        # TRELLIS.2 reconstruction, rescaled/translated (no rotation search)
        # into the scene's world frame by align_trellis2_mesh_to_scene.py,
        # for every class that has both a trellis mesh and a scene crop.
        # Stage 11/12 only ever move an asset by a pure translation derived
        # from its own glb bbox (no external pose file), so overwriting
        # <label>/mesh.glb in place here is a drop-in substitution -- no
        # changes needed downstream. background is never substituted (no
        # TRELLIS.2 reconstruction exists for it).
        if [[ "$USE_TRELLIS" -eq 1 ]]; then
            IFS=',' read -ra USE_TRELLIS_CLASSES <<< "$CLASSES"
            for label in "${USE_TRELLIS_CLASSES[@]}"; do
                # image_to_3d_meshes/<label> keeps the raw --classes spelling (spaces
                # and all -- staged straight from labels.json's class names by
                # stage_trellis2_inputs.py), but every scene-side artifact
                # (shapes/<label>_mesh.ply from Stage 8, glb/<label>/ from
                # mesh_to_glb.py above) uses labels.json's *sanitized* directory
                # name (spaces/slashes -> underscores, see detect_and_segment.py's
                # sanitize_label). A multi-word class name must use each spelling
                # against the artifact that actually uses it, or the scene-crop/
                # out_glb lookups below silently miss and TRELLIS substitution
                # never applies to that class.
                sanitized_label="${label// /_}"
                sanitized_label="${sanitized_label//\//_}"
                TRELLIS_LABEL_GLB="${RUN_DIR}/image_to_3d_meshes/${label}/mesh.glb"
                SCENE_CROP_PLY="${SHAPES_DIR}/${sanitized_label}_mesh.ply"
                if [[ ! -f "$TRELLIS_LABEL_GLB" ]]; then
                    log "Stage 10: --use-trellis: no TRELLIS.2 mesh for '${label}' at ${TRELLIS_LABEL_GLB}, keeping scene-crop glb."
                    continue
                fi
                if [[ ! -f "$SCENE_CROP_PLY" ]]; then
                    log "Stage 10: --use-trellis: no scene crop for '${label}' at ${SCENE_CROP_PLY}, keeping scene-crop glb."
                    continue
                fi
                ALIGN_ZUP_ARGS=()
                [[ "$MESH_BACKEND" == "mvsam3d" ]] && ALIGN_ZUP_ARGS=(--no-zup-correction)
                run_py_step "stage10_align_trellis2_mesh_${sanitized_label}" "scripts/align_trellis2_mesh_to_scene.py" \
                    "${OUTPUT_DIR}/mesh_to_glb.log" append \
                    --trellis_glb "$TRELLIS_LABEL_GLB" \
                    --scene_mesh_ply "$SCENE_CROP_PLY" \
                    --out_glb "${SHAPES_DIR}/glb/${sanitized_label}/mesh.glb" \
                    "${ALIGN_ZUP_ARGS[@]}"
            done
        fi
        stage_end
    else
        log "Stage 10: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 10

    if [[ "$START_FROM_STAGE" -le 11 ]]; then
        stage_start "Stage 11: convert_asset.py (collision_approximation=${COLLISION_APPROXIMATION}) -> ${SHAPES_DIR}/glb/<label>/asset.usd"
        # Only pass the friction table (and switch PhysX to frictionCombineMode=max) when
        # Stage 9 actually produced one -- keeps the --skip_isaac/no-classes invocation
        # byte-identical to before this feature existed.
        CONVERT_ASSET_FRICTION_ARGS=()
        if [[ -n "$CLASSES" && -f "${SHAPES_DIR}/friction_assignments.json" ]]; then
            CONVERT_ASSET_FRICTION_ARGS=(--friction-table "${SHAPES_DIR}/friction_assignments.json" --friction-combine-mode max)
        fi
        run_external_step "stage11_convert_asset" "${ISAACSIM_DIR}/convert_asset.py" "$ISAACSIM_DIR" \
            "${OUTPUT_DIR}/convert_asset.log" new "uv run convert_asset.py" \
            --input "${SHAPES_DIR}/glb" \
            --collision-approximation "$COLLISION_APPROXIMATION" \
            "${CONVERT_ASSET_FRICTION_ARGS[@]}"
        stage_end
    else
        log "Stage 11: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 11

    if [[ "$START_FROM_STAGE" -le 12 ]]; then
        stage_start "Stage 12: compose_isaac_scene.py -> ${SHAPES_DIR}/glb/scene.usda"
        run_external_step "stage12_compose_isaac_scene" "${ISAACSIM_DIR}/compose_isaac_scene.py" "$ISAACSIM_DIR" \
            "${OUTPUT_DIR}/compose_isaac_scene.log" new "uv run compose_isaac_scene.py" \
            --input "${SHAPES_DIR}/glb" \
            --output "${SHAPES_DIR}/glb/scene.usda" \
            --background-label background
        stage_end
    else
        log "Stage 12: skipped (--start-from-stage ${START_FROM_STAGE})"
    fi
    check_stop_after_stage 12
fi

if [[ "$RUN_GLB" -eq 1 && "$START_FROM_STAGE" -le 13 ]]; then
    stage_start "Stage 13: chunked_to_glb.py (simplify_threshold=${SIMPLIFY_THRESHOLD}, texture_size=${TEXTURE_SIZE})"
    run_py_step "stage13_chunked_to_glb" "chunked_to_glb.py" "${OUTPUT_DIR}/glb.log" new \
        --inputs "${OUTPUT_DIR}/to_glb_inputs.pt" \
        --chunk_inputs "${OUTPUT_DIR}/chunk_inputs.pt" \
        --output_dir "$OUTPUT_DIR" \
        --simplify_threshold "$SIMPLIFY_THRESHOLD" \
        --texture_size "$TEXTURE_SIZE"
    stage_end

    log "Done: ${OUTPUT_DIR}/scene.glb"
elif [[ "$RUN_GLB" -eq 1 ]]; then
    log "Stage 13: skipped (--start-from-stage ${START_FROM_STAGE})"
    log "Done: ${SHAPES_DIR}/mesh.ply"
else
    log "--run_glb not set, skipping GLB bake."
    log "Done: ${SHAPES_DIR}/mesh.ply"
fi

# ── Stage 14 (optional): organize final per-object deliverables ──
# Copies the background point-cloud/mesh plus, per class, the detection
# preview, point cloud, cropped mesh, TRELLIS.2 input image and TRELLIS.2
# mesh (if --use-trellis was used) into one self-contained final_objects/
# folder -- only meaningful when --classes was set (otherwise there's no
# per-object split, just a single whole-scene mesh.ply).
FINAL_OBJECTS_DIR="${RUN_DIR}/final_objects"
if [[ -n "$CLASSES" && "$START_FROM_STAGE" -le 14 ]]; then
    stage_start "Stage 14: organizing final objects -> ${FINAL_OBJECTS_DIR}"
    run_py_step "stage14_organize_final_objects" "scripts/organize_final_objects.py" \
        "${OUTPUT_DIR}/reconstruct.log" append \
        --shapes_dir "$SHAPES_DIR" \
        --segmentation_raw_dir "${OUTPUT_DIR}/segmentation_raw" \
        --trellis2_input_dir "${RUN_DIR}/trellis2_input" \
        --trellis2_meshes_dir "${RUN_DIR}/image_to_3d_meshes" \
        --out_dir "$FINAL_OBJECTS_DIR"
    stage_end
else
    log "Stage 14: skipped (no --classes, or --start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 14

# ── Stage 15 (optional): Isaac Sim robot-collision demo ──
# Drives a Jetbot into --robot-target inside the composed scene.usda (Stage 12's
# output) and records a collision proof video + a reusable pre-drive USD stage.
# Runs by default once the USD scene exists and --robot-target is supplied
# (RUN_USD=1 is itself the default; --skip_isaac disables it, same as Stage 11/12).
ROBOT_COLLISION_DIR="${RUN_DIR}/robot_collision"
if [[ "$RUN_USD" -eq 1 && -n "$ROBOT_TARGET" && "$START_FROM_STAGE" -le 15 ]]; then
    stage_start "Stage 15: demo_robot_collide.py (robot_target=${ROBOT_TARGET}) -> ${ROBOT_COLLISION_DIR}/robot_collide.mp4"
    mkdir -p "$ROBOT_COLLISION_DIR"
    run_external_step "stage15_demo_robot_collide" "${ISAACSIM_DIR}/demo_robot_collide.py" "$ISAACSIM_DIR" \
        "${ROBOT_COLLISION_DIR}/robot_collide.log" new "uv run demo_robot_collide.py" \
        --scene "${SHAPES_DIR}/glb/scene.usda" \
        --robot-target "$ROBOT_TARGET" \
        --output "${ROBOT_COLLISION_DIR}/robot_collide.mp4" \
        --stage-output "${ROBOT_COLLISION_DIR}/robot_collide_scene.usda"
    stage_end
else
    log "Stage 15: skipped (no --robot-target, --skip_isaac, or --start-from-stage ${START_FROM_STAGE})"
fi
check_stop_after_stage 15

PIPELINE_ELAPSED=$(( $(date +%s) - PIPELINE_T0 ))
log "Pipeline finished for scene '${SCENE_NAME}' (total elapsed $(format_duration "$PIPELINE_ELAPSED"))"
