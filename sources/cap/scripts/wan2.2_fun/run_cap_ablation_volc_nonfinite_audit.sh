#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_cap_ablation_volc.sh {arm|camera} {uniform|e_only|s_only|s_max1|current}

Submit each command as an independent single-node Volcano job with 8 GPUs.
Optional environment:
  CAP_ABLATION_OUTPUT_ROOT  Shared output root.
  CAP_ABLATION_RUN_ID       Optional run timestamp override (default: current UTC time).
  CAP_ABLATION_RESUME_CHECKPOINT
                            Absolute checkpoint-N path to continue in its original run directory.
  CAP_ABLATION_TRAIN_SAMPLES
                            Positive prefix length (rounded up to 32), or all (default).
  CAP_VOLC_CACHE_ROOT       Cache root (default: this run's node-local cache).
  CAP_NODE_CACHE_ROOT       Node-local runtime/cache root.
  CAP_STAGE_LOCAL_SITE      Stage Python packages to /dev/shm (formal default: 1).
  VIDEOX_LOCAL_SITE_PACKAGES  Node-local Python package mirror.
  CAP_PYTHONPYCACHEPREFIX   Optional node-local Python bytecode cache path.
  CAP_LOG_FLUSH_SECONDS     Max delay before new console.log bytes are closed/flushed.
  TRITON_CACHE_DIR          Node-local DeepSpeed/Triton cache override.
  PYTHON_BIN                VideoX-Fun Python runtime.
  DRY_RUN=1                 Preflight and print the expanded command only.
  SMOKE=1                   One-GPU, one-sample, one-step validation.
EOF
}

if [[ $# -lt 2 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    [[ $# -ge 1 ]] && exit 0
    exit 2
fi

MODALITY="$1"
LOSS_VARIANT="$2"
shift 2

case "$MODALITY" in
    arm|camera) ;;
    *)
        echo "ERROR: modality must be arm or camera; got $MODALITY" >&2
        exit 2
        ;;
esac
case "$LOSS_VARIANT" in
    uniform|e_only|s_only|s_max1|current) ;;
    *)
        echo "ERROR: invalid loss variant: $LOSS_VARIANT" >&2
        exit 2
        ;;
esac

DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"

WORKER_NUM="${MLP_WORKER_NUM:-1}"
WORKER_RANK="${MLP_ROLE_INDEX:-0}"
if [[ "$WORKER_NUM" != "1" || "$WORKER_RANK" != "0" ]]; then
    echo "ERROR: each ablation is a separate single-node job; got MLP_WORKER_NUM=$WORKER_NUM MLP_ROLE_INDEX=$WORKER_RANK" >&2
    exit 1
fi
CONTROLLED_ABLATION_VARIABLES=(
    TRANSFORMER_PATH
    RESUME_FROM_CHECKPOINT
    PRETRAINED_MODEL
    CONTROL_MODEL
    TRAIN_DATA_META
    TRAIN_DATA_DIR
    MAX_TRAIN_STEPS
    MAX_TRAIN_SAMPLES
    VIDEOX_RESUME_RESET_OPTIMIZER
    BENCHMARK_TIMING_PATH
    METHOD1_SAMPLE_LOSS_DIR
)
# Formal ablations always use the fixed model, dataset, and training length
# below. Ignore stale variables inherited from an existing Volcano template.
# Controlled continuation is enabled only through
# CAP_ABLATION_RESUME_CHECKPOINT, never through raw RESUME_FROM_CHECKPOINT.
for variable_name in "${CONTROLLED_ABLATION_VARIABLES[@]}"; do
    unset "$variable_name"
done

case "$MODALITY" in
    arm) CAP_METADATA_SAMPLES=365831 ;;
    camera) CAP_METADATA_SAMPLES=505955 ;;
esac
CAP_ABLATION_TRAIN_SAMPLES="${CAP_ABLATION_TRAIN_SAMPLES:-all}"
CAP_ALIGNED_TRAIN_SAMPLES="all"
CAP_SELECTED_METADATA_SAMPLES="$CAP_METADATA_SAMPLES"
if [[ "$CAP_ABLATION_TRAIN_SAMPLES" != "all" ]]; then
    if [[ ! "$CAP_ABLATION_TRAIN_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: CAP_ABLATION_TRAIN_SAMPLES must be a positive integer or all; got $CAP_ABLATION_TRAIN_SAMPLES" >&2
        exit 2
    fi
    CAP_REQUESTED_TRAIN_SAMPLES=$((10#$CAP_ABLATION_TRAIN_SAMPLES))
    if (( CAP_REQUESTED_TRAIN_SAMPLES > CAP_METADATA_SAMPLES )); then
        echo "ERROR: CAP_ABLATION_TRAIN_SAMPLES cannot exceed $CAP_METADATA_SAMPLES for $MODALITY; got $CAP_ABLATION_TRAIN_SAMPLES" >&2
        exit 2
    fi
    CAP_ALIGNED_TRAIN_SAMPLES=$(( (CAP_REQUESTED_TRAIN_SAMPLES + 31) / 32 * 32 ))
    export MAX_TRAIN_SAMPLES="$CAP_ALIGNED_TRAIN_SAMPLES"
    if (( CAP_ALIGNED_TRAIN_SAMPLES < CAP_SELECTED_METADATA_SAMPLES )); then
        CAP_SELECTED_METADATA_SAMPLES="$CAP_ALIGNED_TRAIN_SAMPLES"
    fi
fi
CAP_SCHEDULED_TRAIN_SAMPLES=$(( (CAP_SELECTED_METADATA_SAMPLES + 31) / 32 * 32 ))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_CAP="$SCRIPT_DIR/run_cap_train_nonfinite_audit.sh"
REALTIME_TEE="$SCRIPT_DIR/realtime_tee.py"
PREPARE_LOCAL_SITE="$SCRIPT_DIR/prepare_method1_local_site.sh"
METHOD1_COMPAT_DIR="$SCRIPT_DIR/method1_compat"
if [[ ! -f "$RUN_CAP" ]]; then
    echo "ERROR: missing CAP launcher: $RUN_CAP" >&2
    exit 1
fi
if [[ ! -f "$REALTIME_TEE" ]]; then
    echo "ERROR: missing realtime log writer: $REALTIME_TEE" >&2
    exit 1
fi
if [[ ! -f "$PREPARE_LOCAL_SITE" || ! -f "$METHOD1_COMPAT_DIR/sitecustomize.py" ]]; then
    echo "ERROR: missing CAP local-runtime helpers under $SCRIPT_DIR" >&2
    exit 1
fi

CAP_ABLATION_OUTPUT_ROOT="${CAP_ABLATION_OUTPUT_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)/outputs/cap-ablation}"
CAP_ABLATION_RESUME_CHECKPOINT="${CAP_ABLATION_RESUME_CHECKPOINT:-}"
CAP_ABLATION_RUN_ID_WAS_SET="${CAP_ABLATION_RUN_ID+x}"
OUTPUT_DIR_WAS_SET="${OUTPUT_DIR+x}"
RESUMING=0
CAP_RESUME_DATASET_ARGS=()

if [[ -n "$CAP_ABLATION_RESUME_CHECKPOINT" ]]; then
    RESUMING=1
    if [[ "$SMOKE" == "1" ]]; then
        echo "ERROR: checkpoint continuation is not compatible with SMOKE=1" >&2
        exit 1
    fi
    if [[ "$CAP_ABLATION_RESUME_CHECKPOINT" != /* ]]; then
        echo "ERROR: CAP_ABLATION_RESUME_CHECKPOINT must be an absolute path" >&2
        exit 1
    fi
    if [[ ! -d "$CAP_ABLATION_RESUME_CHECKPOINT" ]]; then
        echo "ERROR: resume checkpoint directory does not exist: $CAP_ABLATION_RESUME_CHECKPOINT" >&2
        exit 1
    fi

    RESUME_FROM_CHECKPOINT="$(readlink -f -- "$CAP_ABLATION_RESUME_CHECKPOINT")"
    RESUME_CHECKPOINT_NAME="$(basename -- "$RESUME_FROM_CHECKPOINT")"
    if [[ ! "$RESUME_CHECKPOINT_NAME" =~ ^checkpoint-([1-9][0-9]*)$ ]]; then
        echo "ERROR: resume path must end with checkpoint-<positive step>: $RESUME_FROM_CHECKPOINT" >&2
        exit 1
    fi
    RESUME_STEP="${BASH_REMATCH[1]}"
    RESUME_OUTPUT_DIR="$(dirname -- "$RESUME_FROM_CHECKPOINT")"
    RESUME_RUN_ID="$(basename -- "$RESUME_OUTPUT_DIR")"
    RESUME_VARIANT_DIR="$(dirname -- "$RESUME_OUTPUT_DIR")"
    RESUME_MODALITY_DIR="$(dirname -- "$RESUME_VARIANT_DIR")"
    RESUME_VARIANT="$(basename -- "$RESUME_VARIANT_DIR")"
    RESUME_MODALITY="$(basename -- "$RESUME_MODALITY_DIR")"
    if [[ "$RESUME_MODALITY" != "$MODALITY" || "$RESUME_VARIANT" != "$LOSS_VARIANT" ]]; then
        echo "ERROR: checkpoint belongs to $RESUME_MODALITY/$RESUME_VARIANT, not $MODALITY/$LOSS_VARIANT" >&2
        exit 1
    fi
    if [[ -n "$CAP_ABLATION_RUN_ID_WAS_SET" && "${CAP_ABLATION_RUN_ID:-}" != "$RESUME_RUN_ID" ]]; then
        echo "ERROR: do not set a different CAP_ABLATION_RUN_ID when resuming; checkpoint run-id is $RESUME_RUN_ID" >&2
        exit 1
    fi
    if [[ -n "$OUTPUT_DIR_WAS_SET" ]] \
        && [[ "$(readlink -m -- "${OUTPUT_DIR:-}")" != "$RESUME_OUTPUT_DIR" ]]; then
        echo "ERROR: do not set a different OUTPUT_DIR when resuming; checkpoint output is $RESUME_OUTPUT_DIR" >&2
        exit 1
    fi

    HAS_MODEL_STATE=0
    if [[ -s "$RESUME_FROM_CHECKPOINT/diffusion_pytorch_model.safetensors" ]] \
        || [[ -s "$RESUME_FROM_CHECKPOINT/pytorch_model.bin" ]] \
        || [[ -d "$RESUME_FROM_CHECKPOINT/transformer" ]] \
        || [[ -s "$RESUME_FROM_CHECKPOINT/pytorch_model_fsdp_0/.metadata" ]]; then
        HAS_MODEL_STATE=1
    fi
    HAS_OPTIMIZER_STATE=0
    if [[ -s "$RESUME_FROM_CHECKPOINT/optimizer_0/.metadata" ]] \
        || [[ -d "$RESUME_FROM_CHECKPOINT/optimizer" ]]; then
        HAS_OPTIMIZER_STATE=1
    fi
    if [[ "$HAS_MODEL_STATE" != "1" || "$HAS_OPTIMIZER_STATE" != "1" ]] \
        || [[ ! -s "$RESUME_FROM_CHECKPOINT/scheduler.bin" ]] \
        || [[ ! -s "$RESUME_FROM_CHECKPOINT/sampler_pos_start.pkl" ]]; then
        echo "ERROR: checkpoint is incomplete; model, optimizer, scheduler, and sampler state are required: $RESUME_FROM_CHECKPOINT" >&2
        exit 1
    fi
    for rank in {0..7}; do
        if [[ ! -s "$RESUME_FROM_CHECKPOINT/random_states_${rank}.pkl" ]]; then
            echo "ERROR: checkpoint is missing rank $rank RNG state: $RESUME_FROM_CHECKPOINT/random_states_${rank}.pkl" >&2
            exit 1
        fi
    done

    for sibling in "$RESUME_OUTPUT_DIR"/checkpoint-*; do
        [[ -d "$sibling" ]] || continue
        sibling_name="$(basename -- "$sibling")"
        if [[ "$sibling_name" =~ ^checkpoint-([1-9][0-9]*)$ ]] \
            && (( 10#${BASH_REMATCH[1]} > 10#$RESUME_STEP )); then
            echo "ERROR: a newer checkpoint already exists: $sibling" >&2
            echo "Resume from the newest checkpoint to avoid overwriting later training state." >&2
            exit 1
        fi
    done

    CAP_ABLATION_RUN_ID="$RESUME_RUN_ID"
    OUTPUT_DIR="$RESUME_OUTPUT_DIR"
    export RESUME_FROM_CHECKPOINT
    if (( CAP_SELECTED_METADATA_SAMPLES < CAP_METADATA_SAMPLES )); then
        CAP_RESUME_DATASET_ARGS+=(--resume_with_new_dataset)
    fi
else
    CAP_ABLATION_RUN_ID="${CAP_ABLATION_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
    if [[ -z "${OUTPUT_DIR:-}" ]]; then
        OUTPUT_DIR="$CAP_ABLATION_OUTPUT_ROOT/$MODALITY/$LOSS_VARIANT/$CAP_ABLATION_RUN_ID"
    fi
fi

CAP_RESUME_PREFIX_EPOCHS=5
CAP_RESUME_TARGET_STEP=""
if (( ${#CAP_RESUME_DATASET_ARGS[@]} > 0 )); then
    CAP_RESUME_TARGET_STEP=$((
        RESUME_STEP + CAP_RESUME_PREFIX_EPOCHS * CAP_SCHEDULED_TRAIN_SAMPLES / 32
    ))
    export MAX_TRAIN_STEPS="$CAP_RESUME_TARGET_STEP"
fi

if [[ -n "$CAP_ABLATION_RUN_ID" && ! "$CAP_ABLATION_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: CAP_ABLATION_RUN_ID may contain only letters, digits, dot, underscore, and hyphen" >&2
    exit 1
fi
export OUTPUT_DIR
export BENCHMARK_TIMING_PATH="$OUTPUT_DIR/train_metrics.jsonl"
export METHOD1_SAMPLE_LOSS_DIR="$OUTPUT_DIR/sample_losses"
if [[ "$RESUMING" != "1" && "$DRY_RUN" != "1" && -d "$OUTPUT_DIR" ]] \
    && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: output directory is not empty: $OUTPUT_DIR" >&2
    echo "Set a new CAP_ABLATION_RUN_ID, OUTPUT_DIR, or CAP_ABLATION_OUTPUT_ROOT." >&2
    exit 1
fi

CAP_CONDA_ENV="${CAP_CONDA_ENV:-}"
if [[ -n "$CAP_CONDA_ENV" ]]; then
    DEFAULT_PYTHON_BIN="$CAP_CONDA_ENV/bin/python"
else
    DEFAULT_PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
export PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 1
fi

if [[ -z "${CAP_STAGE_LOCAL_SITE:-}" ]]; then
    CAP_STAGE_LOCAL_SITE=0
fi
if [[ "$CAP_STAGE_LOCAL_SITE" != "0" && "$CAP_STAGE_LOCAL_SITE" != "1" ]]; then
    echo "ERROR: CAP_STAGE_LOCAL_SITE must be 0 or 1; got $CAP_STAGE_LOCAL_SITE" >&2
    exit 1
fi
if [[ "$CAP_STAGE_LOCAL_SITE" == "1" ]]; then
    if [[ -z "$CAP_CONDA_ENV" ]]; then
        echo "ERROR: CAP_STAGE_LOCAL_SITE=1 requires CAP_CONDA_ENV" >&2
        exit 1
    fi
    export VIDEOX_LOCAL_SITE_PACKAGES="${VIDEOX_LOCAL_SITE_PACKAGES:?Set VIDEOX_LOCAL_SITE_PACKAGES to a writable cache directory}"
    CONDA_ENV="$CAP_CONDA_ENV" bash "$PREPARE_LOCAL_SITE"
fi
if [[ -n "${VIDEOX_LOCAL_SITE_PACKAGES:-}" ]]; then
    for package in torch diffusers accelerate safetensors nvidia transformers; do
        if [[ ! -e "$VIDEOX_LOCAL_SITE_PACKAGES/$package" ]]; then
            echo "ERROR: VIDEOX_LOCAL_SITE_PACKAGES is incomplete: missing $package" >&2
            exit 1
        fi
    done
    export PYTHONPATH="$METHOD1_COMPAT_DIR:$VIDEOX_LOCAL_SITE_PACKAGES:$REPO_ROOT:${PYTHONPATH:-}"
else
    export PYTHONPATH="$METHOD1_COMPAT_DIR:$REPO_ROOT:${PYTHONPATH:-}"
fi

# The runtime lives on a shared filesystem. Concurrent Volcano jobs must not
# read or write its shared __pycache__ files, which can be observed mid-write.
CAP_NODE_CACHE_ROOT="${CAP_NODE_CACHE_ROOT:-/dev/shm/cap-runtime-cache/${UID}-${MODALITY}-${LOSS_VARIANT}-${CAP_ABLATION_RUN_ID}}"
if [[ -n "${CAP_PYTHONPYCACHEPREFIX:-}" ]]; then
    PYTHONPYCACHEPREFIX="$CAP_PYTHONPYCACHEPREFIX"
    mkdir -p "$PYTHONPYCACHEPREFIX"
else
    PYTHONPYCACHEPREFIX="$CAP_NODE_CACHE_ROOT/pycache"
    mkdir -p "$PYTHONPYCACHEPREFIX"
fi
export PYTHONPYCACHEPREFIX
export PYTHONDONTWRITEBYTECODE=1

CAP_VOLC_CACHE_ROOT="${CAP_VOLC_CACHE_ROOT:-$CAP_NODE_CACHE_ROOT/external_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CAP_VOLC_CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CAP_VOLC_CACHE_ROOT/huggingface}"
export TMPDIR="${TMPDIR:-$CAP_NODE_CACHE_ROOT/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CAP_NODE_CACHE_ROOT/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CAP_NODE_CACHE_ROOT/torch_extensions}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$CAP_NODE_CACHE_ROOT/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$CAP_NODE_CACHE_ROOT/cuda}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$CAP_NODE_CACHE_ROOT/numba}"
export VIDEOX_METHOD1_ISOLATE_RUNTIME_CACHE=1
export VIDEOX_RUNTIME_CACHE_ROOT="${VIDEOX_RUNTIME_CACHE_ROOT:-$CAP_NODE_CACHE_ROOT/per_rank}"
export RUN_TAG="${RUN_TAG:-cap-${MODALITY}-${LOSS_VARIANT}-${CAP_ABLATION_RUN_ID}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p \
    "$XDG_CACHE_HOME" \
    "$HF_HOME" \
    "$TMPDIR" \
    "$TRITON_CACHE_DIR" \
    "$TORCH_EXTENSIONS_DIR" \
    "$TORCHINDUCTOR_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$NUMBA_CACHE_DIR"

export SEED=42
export METHOD1_LOSS_VARIANT="$LOSS_VARIANT"
export METHOD1_TAU_S=0.50
export METHOD1_EPS=1e-6
export METHOD1_MSE_THRESHOLD=0
export CONTROL_MODEL="${CONTROL_MODEL:?Set CONTROL_MODEL to a Wan2.2 TI2V control directory}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29611}}"

if [[ -n "${CAP_NONFINITE_AUDIT_MAX_TRAIN_STEPS:-}" ]]; then
    if [[ ! "$CAP_NONFINITE_AUDIT_MAX_TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: CAP_NONFINITE_AUDIT_MAX_TRAIN_STEPS must be a positive integer" >&2
        exit 2
    fi
    export MAX_TRAIN_STEPS="$CAP_NONFINITE_AUDIT_MAX_TRAIN_STEPS"
fi

if [[ "$DRY_RUN" == "1" || "$SMOKE" == "1" ]]; then
    export NUM_GPUS="${NUM_GPUS:-1}"
    export DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
    export LOW_VRAM="${LOW_VRAM:-1}"
    export SKIP_SANITY_CHECK="${SKIP_SANITY_CHECK:-1}"
    if [[ "$SMOKE" == "1" ]]; then
        export METHOD1_ACTION_DROPOUT_PROB=0
    else
        export METHOD1_ACTION_DROPOUT_PROB=0.10
    fi
else
    export METHOD1_ACTION_DROPOUT_PROB=0.10
    export NUM_GPUS=8
    export USE_FSDP=1
    export FRAMES=17
    export GRAD_ACCUM=4
    export DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
    export LOW_VRAM="${LOW_VRAM:-1}"
    export SKIP_SANITY_CHECK="${SKIP_SANITY_CHECK:-1}"
    export LEARNING_RATE=2e-5
    export LR_WARMUP_STEPS=100
    export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-1000}"
    export CHECKPOINTS_TOTAL_LIMIT=10
    export EPOCHS=5
    export HEIGHT=704
    export WIDTH=1280
fi

if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

if [[ "$MODALITY" == "arm" ]]; then
    CAP_MODE="current_arm"
else
    CAP_MODE="camera"
fi

echo "CAP ablation: modality=$MODALITY variant=$LOSS_VARIANT GPUs=$NUM_GPUS output=$OUTPUT_DIR"
echo "CAP runtime: python=$PYTHON_BIN local_site=${VIDEOX_LOCAL_SITE_PACKAGES:-disabled} staged=$CAP_STAGE_LOCAL_SITE node_cache=$CAP_NODE_CACHE_ROOT"
echo "CAP Python bytecode: writes=disabled prefix=$PYTHONPYCACHEPREFIX scope=node-local"
echo "CAP caches: xdg=$XDG_CACHE_HOME hf=$HF_HOME tmp=$TMPDIR cuda=$CUDA_CACHE_PATH inductor=$TORCHINDUCTOR_CACHE_DIR"
echo "CAP performance: low_vram=$LOW_VRAM dataloader_workers_per_rank=$DATALOADER_WORKERS dataloader_workers_total=$((NUM_GPUS * DATALOADER_WORKERS)) skip_sanity_check=$SKIP_SANITY_CHECK"
echo "CAP data selection: requested=$CAP_ABLATION_TRAIN_SAMPLES aligned_limit=$CAP_ALIGNED_TRAIN_SAMPLES metadata_samples=$CAP_SELECTED_METADATA_SAMPLES scheduled_samples=$CAP_SCHEDULED_TRAIN_SAMPLES"
echo "CAP records: metrics=$BENCHMARK_TIMING_PATH sample_losses=$METHOD1_SAMPLE_LOSS_DIR"
if [[ "$RESUMING" == "1" ]]; then
    echo "CAP resume: checkpoint=$RESUME_FROM_CHECKPOINT step=$RESUME_STEP output=$OUTPUT_DIR"
fi
if [[ -n "$CAP_RESUME_TARGET_STEP" ]]; then
    echo "CAP prefix resume: additional_epochs=$CAP_RESUME_PREFIX_EPOCHS target_step=$CAP_RESUME_TARGET_STEP sampler_pos_start=0"
fi
if [[ "$DRY_RUN" == "1" ]]; then
    exec bash "$RUN_CAP" "$CAP_MODE" "${CAP_RESUME_DATASET_ARGS[@]}" "$@"
fi

LOG_FILE="$OUTPUT_DIR/console.log"
LOG_FLUSH_SECONDS="${CAP_LOG_FLUSH_SECONDS:-2}"
if [[ "$RESUMING" == "1" ]]; then
    {
        printf '\n'
        echo "===== CAP resume $(date -u +%Y-%m-%dT%H:%M:%SZ) checkpoint=$RESUME_FROM_CHECKPOINT ====="
    } >>"$LOG_FILE"
else
    : >"$LOG_FILE"
fi
{
    echo "CAP ablation: modality=$MODALITY variant=$LOSS_VARIANT GPUs=$NUM_GPUS output=$OUTPUT_DIR"
    echo "CAP runtime: python=$PYTHON_BIN local_site=${VIDEOX_LOCAL_SITE_PACKAGES:-disabled} staged=$CAP_STAGE_LOCAL_SITE node_cache=$CAP_NODE_CACHE_ROOT"
    echo "CAP Python bytecode: writes=disabled prefix=$PYTHONPYCACHEPREFIX scope=node-local"
    echo "CAP caches: xdg=$XDG_CACHE_HOME hf=$HF_HOME tmp=$TMPDIR cuda=$CUDA_CACHE_PATH inductor=$TORCHINDUCTOR_CACHE_DIR"
    echo "CAP performance: low_vram=$LOW_VRAM dataloader_workers_per_rank=$DATALOADER_WORKERS dataloader_workers_total=$((NUM_GPUS * DATALOADER_WORKERS)) skip_sanity_check=$SKIP_SANITY_CHECK"
    echo "CAP data selection: requested=$CAP_ABLATION_TRAIN_SAMPLES aligned_limit=$CAP_ALIGNED_TRAIN_SAMPLES metadata_samples=$CAP_SELECTED_METADATA_SAMPLES scheduled_samples=$CAP_SCHEDULED_TRAIN_SAMPLES"
    echo "CAP records: metrics=$BENCHMARK_TIMING_PATH sample_losses=$METHOD1_SAMPLE_LOSS_DIR"
    if [[ "$RESUMING" == "1" ]]; then
        echo "CAP resume: checkpoint=$RESUME_FROM_CHECKPOINT step=$RESUME_STEP output=$OUTPUT_DIR"
    fi
    if [[ -n "$CAP_RESUME_TARGET_STEP" ]]; then
        echo "CAP prefix resume: additional_epochs=$CAP_RESUME_PREFIX_EPOCHS target_step=$CAP_RESUME_TARGET_STEP sampler_pos_start=0"
    fi
} >>"$LOG_FILE"

set +e
PYTHONUNBUFFERED=1 bash "$RUN_CAP" "$CAP_MODE" "${CAP_RESUME_DATASET_ARGS[@]}" "$@" 2>&1 \
    | "$PYTHON_BIN" "$REALTIME_TEE" "$LOG_FILE" --flush-seconds "$LOG_FLUSH_SECONDS"
PIPE_STATUSES=("${PIPESTATUS[@]}")
STATUS="${PIPE_STATUSES[0]}"
LOGGER_STATUS="${PIPE_STATUSES[1]}"
set -e
if [[ "$STATUS" == "0" && "$LOGGER_STATUS" != "0" ]]; then
    echo "ERROR: realtime log writer failed with exit code $LOGGER_STATUS" >&2
    STATUS="$LOGGER_STATUS"
fi
printf '%s\n' "$STATUS" >"$OUTPUT_DIR/exit_code"
echo "CAP ablation exit_code=$STATUS log=$LOG_FILE"
exit "$STATUS"
