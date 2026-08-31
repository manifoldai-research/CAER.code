#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: run_cap_train.sh {arm|current_arm|action_map|camera|poseanything} [extra training args]

Environment:
  DRY_RUN=1              Run metadata/model preflight and print the command only.
  SMOKE=1                Use 1 sample, 5 frames, 64x96, and 1 optimizer step.
  CONTROL_MODEL=PATH     Wan2.2 base (48ch) or prepared control model (100ch).
  POSE_MODEL=PATH        Required 48-channel Wan2.2 base for PoseAnything.
  PYTHON_BIN=PATH        Python runtime (default: python3 on PATH).
  NUM_GPUS=N             torchrun workers (default: 8; smoke default: 1).
  TRAIN_BATCH_SIZE=N     Per-device batch size (default: 1).
  USE_FSDP=0|1           Use FSDP (default: 1 for multi-GPU, 0 for one GPU).
  LOW_VRAM=0|1           Move VAE/text encoder between CPU/GPU each batch (default: 1).
  SKIP_SANITY_CHECK=0|1  Skip first-batch GIF/PNG generation (default: 0).
  TRAIN_DATA_META=PATH   Override the selected dataset metadata.
  TRAIN_DATA_DIR=PATH    Resolve relative media paths under this directory.
  OUTPUT_DIR=PATH        Override the run output directory.
  METHOD1_LOSS_VARIANT   MSE|CAER (default: CAER).
EOF
}

if [[ $# -gt 0 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
    usage
    exit 0
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
    MODE="$1"
    shift
else
    MODE="${CAP_MODE:-}"
fi

if [[ -z "$MODE" ]]; then
    usage
    exit 2
fi
case "$MODE" in
    current_arm)
        MODE="arm"
        ;;
    arm|action_map|camera|poseanything) ;;
    *)
        echo "ERROR: unsupported CAP mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUNDLE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_control_camera_arm_actionmap_method1.py"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/config/wan2.2/wan_civitai_5b.yaml}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
# PoseAnything's formal migration path uses the CAP ablation CAER weights.
# Keep this as the direct-entry default too; callers can still select another
# variant explicitly for non-formal experiments.
if [[ "$MODE" == "poseanything" && -z "${METHOD1_LOSS_VARIANT:-}" ]]; then
    METHOD1_LOSS_VARIANT="CAER"
else
    METHOD1_LOSS_VARIANT="${METHOD1_LOSS_VARIANT:-CAER}"
fi

case "$METHOD1_LOSS_VARIANT" in
    MSE|CAER) ;;
    *)
        echo "ERROR: unsupported METHOD1_LOSS_VARIANT: $METHOD1_LOSS_VARIANT" >&2
        exit 2
        ;;
esac

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_SCRIPT" || ! -f "$CONFIG_PATH" ]]; then
    echo "ERROR: CAP training script or config is missing under $REPO_ROOT" >&2
    exit 1
fi

case "$MODE" in
    arm)
        DEFAULT_HEIGHT=704
        DEFAULT_WIDTH=1280
        ;;
    action_map)
        DEFAULT_HEIGHT=704
        DEFAULT_WIDTH=1280
        ;;
    camera)
        DEFAULT_HEIGHT=704
        DEFAULT_WIDTH=1280
        ;;
    poseanything)
        DEFAULT_HEIGHT=704
        DEFAULT_WIDTH=1280
        ;;
esac

if [[ "$SMOKE" == "1" ]]; then
    HEIGHT="${HEIGHT:-64}"
    WIDTH="${WIDTH:-96}"
    FRAMES="${FRAMES:-5}"
    NUM_GPUS="${NUM_GPUS:-1}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
    GRAD_ACCUM="${GRAD_ACCUM:-1}"
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-0}"
    LOW_VRAM="${LOW_VRAM:-1}"
    SKIP_SANITY_CHECK="${SKIP_SANITY_CHECK:-1}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1}"
    METHOD1_ACTION_DROPOUT_PROB="${METHOD1_ACTION_DROPOUT_PROB:-0}"
else
    HEIGHT="${HEIGHT:-$DEFAULT_HEIGHT}"
    WIDTH="${WIDTH:-$DEFAULT_WIDTH}"
    FRAMES="${FRAMES:-17}"
    NUM_GPUS="${NUM_GPUS:-8}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
    GRAD_ACCUM="${GRAD_ACCUM:-4}"
    DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
    LOW_VRAM="${LOW_VRAM:-1}"
    SKIP_SANITY_CHECK="${SKIP_SANITY_CHECK:-0}"
    MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
    MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
    METHOD1_ACTION_DROPOUT_PROB="${METHOD1_ACTION_DROPOUT_PROB:-0.10}"
fi

if [[ ! "$TRAIN_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: TRAIN_BATCH_SIZE must be a positive integer; got $TRAIN_BATCH_SIZE" >&2
    exit 1
fi
if [[ "$LOW_VRAM" != "0" && "$LOW_VRAM" != "1" ]]; then
    echo "ERROR: LOW_VRAM must be 0 or 1; got $LOW_VRAM" >&2
    exit 1
fi
if [[ "$SKIP_SANITY_CHECK" != "0" && "$SKIP_SANITY_CHECK" != "1" ]]; then
    echo "ERROR: SKIP_SANITY_CHECK must be 0 or 1; got $SKIP_SANITY_CHECK" >&2
    exit 1
fi

if (( HEIGHT % 32 != 0 || WIDTH % 32 != 0 )); then
    echo "ERROR: HEIGHT and WIDTH must both be divisible by 32; got ${HEIGHT}x${WIDTH}" >&2
    exit 1
fi
if (( FRAMES < 1 || (FRAMES - 1) % 4 != 0 )); then
    echo "ERROR: FRAMES must have form 4n+1; got $FRAMES" >&2
    exit 1
fi

if [[ -z "${USE_FSDP:-}" ]]; then
    if (( NUM_GPUS > 1 )); then
        USE_FSDP=1
    else
        USE_FSDP=0
    fi
fi
if [[ "$USE_FSDP" != "0" && "$USE_FSDP" != "1" ]]; then
    echo "ERROR: USE_FSDP must be 0 or 1; got $USE_FSDP" >&2
    exit 1
fi

if [[ "$MODE" == "poseanything" ]]; then
    PRETRAINED_MODEL="${PRETRAINED_MODEL:-${POSE_MODEL:?Set POSE_MODEL or PRETRAINED_MODEL to the 48-channel Wan2.2-TI2V-5B directory}}"
    EXPECTED_SOURCE_CHANNELS="48"
else
    PRETRAINED_MODEL="${PRETRAINED_MODEL:-${CONTROL_MODEL:?Set CONTROL_MODEL to a Wan2.2 TI2V control directory}}"
    EXPECTED_SOURCE_CHANNELS="48,100"
fi

MODEL_CONFIG="$PRETRAINED_MODEL/config.json"
if [[ ! -f "$MODEL_CONFIG" ]]; then
    echo "ERROR: model config is missing: $MODEL_CONFIG" >&2
    if [[ "$MODE" != "poseanything" ]]; then
        echo "Set CONTROL_MODEL to a persistent 100-channel TI2V-control checkpoint." >&2
    fi
    exit 1
fi

CAP_SKIP_RUNTIME_PREFLIGHT="${CAP_SKIP_RUNTIME_PREFLIGHT:-0}"
if [[ "$CAP_SKIP_RUNTIME_PREFLIGHT" != "0" && "$CAP_SKIP_RUNTIME_PREFLIGHT" != "1" ]]; then
    echo "ERROR: CAP_SKIP_RUNTIME_PREFLIGHT must be 0 or 1" >&2
    exit 1
fi
if [[ "$CAP_SKIP_RUNTIME_PREFLIGHT" == "1" && "$DRY_RUN" != "1" ]]; then
    echo "ERROR: CAP_SKIP_RUNTIME_PREFLIGHT=1 is allowed only with DRY_RUN=1" >&2
    exit 1
fi
if [[ "$CAP_SKIP_RUNTIME_PREFLIGHT" == "1" ]]; then
    echo "CAP model preflight: covered by four-modal-repro/scripts/preflight.py"
else
"$PYTHON_BIN" - "$MODEL_CONFIG" "$EXPECTED_SOURCE_CHANNELS" "$MODE" "$DRY_RUN" <<'PY'
import glob
import json
import os
import sys

path, expected_raw, mode, dry_run = sys.argv[1:]
with open(path, "r", encoding="utf-8") as handle:
    config = json.load(handle)
config_channels = config.get("in_dim", config.get("in_channels"))
expected = {int(value) for value in expected_raw.split(",")}
if config_channels not in expected:
    raise SystemExit(
        f"ERROR: {mode} source model must have in_dim in {sorted(expected)}, "
        f"got {config_channels!r}: {path}"
    )

model_root = os.path.dirname(path)
index_path = os.path.join(model_root, "diffusion_pytorch_model.safetensors.index.json")
weight_files = []
if os.path.isfile(index_path):
    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    patch_file = index.get("weight_map", {}).get("patch_embedding.weight")
    if patch_file is None:
        raise SystemExit(f"ERROR: patch_embedding.weight is absent from {index_path}")
    weight_files = [os.path.join(model_root, patch_file)]
else:
    weight_files = sorted(glob.glob(os.path.join(model_root, "*.safetensors")))

for weight_file in weight_files:
    if not os.path.isfile(weight_file) or os.path.getsize(weight_file) <= 0:
        raise SystemExit(f"ERROR: model weight shard is missing or empty: {weight_file}")

weight_channels = config_channels if dry_run == "1" else None
has_camera_adapter_weights = False
if dry_run != "1":
    from safetensors import safe_open

    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            has_camera_adapter_weights = has_camera_adapter_weights or any(
                key.startswith("control_adapter.") for key in keys
            )
            if "patch_embedding.weight" in keys:
                shape = tuple(handle.get_slice("patch_embedding.weight").get_shape())
                if len(shape) != 5:
                    raise SystemExit(
                        f"ERROR: patch_embedding.weight must be rank 5, got {shape}: {weight_file}"
                    )
                weight_channels = int(shape[1])

if weight_channels is None:
    raise SystemExit(f"ERROR: could not locate patch_embedding.weight under {model_root}")
if weight_channels not in expected:
    raise SystemExit(
        f"ERROR: {mode} checkpoint patch channels must be in {sorted(expected)}, "
        f"got {weight_channels}: {model_root}"
    )

target = 96 if mode == "poseanything" else 100
if weight_channels > target:
    raise SystemExit(
        f"ERROR: refusing to truncate patch channels from {weight_channels} to {target}: {model_root}"
    )
initialization = (
    "direct" if weight_channels == target else f"zero_expand_{weight_channels}_to_{target}"
)
adapter_source = "not_used"
if mode == "camera":
    adapter_source = "checkpoint" if has_camera_adapter_weights else "new_zero_output_initialization"
print(
    f"CAP model preflight: mode={mode} config_in_dim={config_channels} "
    f"weight_in_dim={weight_channels} target_in_dim={target} "
    f"patch_initialization={initialization} camera_adapter={adapter_source} config={path}"
    f" inspection={'index' if dry_run == '1' else 'safetensors'}"
)
PY
fi

if [[ "$CAP_SKIP_RUNTIME_PREFLIGHT" == "1" ]]; then
    echo "CAP runtime preflight: skipped for command-only dry-run"
else
"$PYTHON_BIN" - "$REPO_ROOT" "$DRY_RUN" <<'PY'
import sys

repo_root, dry_run = sys.argv[1:]
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from videox_fun.runtime_compat import configure_attention_runtime

attention_runtime = configure_attention_runtime()

import accelerate
import diffusers
import torch
import transformers
if dry_run != "1":
    from videox_fun.models import AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel
    from videox_fun.pipeline import Wan2_2FunControlPipeline

print(
    "CAP runtime preflight: "
    f"python={sys.executable} torch={torch.__version__} accelerate={accelerate.__version__} "
    f"diffusers={diffusers.__version__} transformers={transformers.__version__} "
    f"cuda={int(torch.cuda.is_available())} devices={torch.cuda.device_count()} "
    f"attention={attention_runtime['backend']} "
    f"inspection={'core-packages' if dry_run == '1' else 'training-classes'}"
)
PY
fi

DATASET_NAME="$MODE"
if [[ "$MODE" == "arm" ]]; then
    DATASET_NAME="current_arm"
fi
DATA_ARGS=(--dataset_name "$DATASET_NAME" --action_injection "$MODE")
if [[ -n "${TRAIN_DATA_META:-}" ]]; then
    DATA_ARGS+=(--train_data_meta "$TRAIN_DATA_META")
fi
if [[ -n "${TRAIN_DATA_DIR:-}" ]]; then
    DATA_ARGS+=(--train_data_dir "$TRAIN_DATA_DIR")
fi

if [[ "$CAP_SKIP_RUNTIME_PREFLIGHT" == "1" ]]; then
    echo "CAP metadata preflight: covered by four-modal-repro/scripts/preflight.py"
elif [[ "$DRY_RUN" == "1" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/cap_metadata_preflight.py" "${DATA_ARGS[@]}"
else
    "$PYTHON_BIN" "$TRAIN_SCRIPT" \
        --metadata_preflight_only \
        "${DATA_ARGS[@]}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-${BUNDLE_ROOT}/outputs/${MODE}-method1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-960}"
MASTER_PORT="${MASTER_PORT:-29611}"

RUN_CMD=(
    "$PYTHON_BIN" -m torch.distributed.run
    "--nproc_per_node=$NUM_GPUS"
    --nnodes=1
    --node_rank=0
    --master_addr=127.0.0.1
    "--master_port=$MASTER_PORT"
    "$TRAIN_SCRIPT"
)
TRAIN_ARGS=(
    --pretrained_model_name_or_path "$PRETRAINED_MODEL"
    --config_path "$CONFIG_PATH"
    "${DATA_ARGS[@]}"
    --image_sample_size "$SAMPLE_SIZE"
    --video_sample_size "$SAMPLE_SIZE"
    --token_sample_size "$SAMPLE_SIZE"
    --fix_sample_size "$HEIGHT" "$WIDTH"
    --require_input_resolution "$HEIGHT" "$WIDTH"
    --video_sample_stride 1
    --video_sample_n_frames "$FRAMES"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --video_repeat 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --dataloader_num_workers "$DATALOADER_WORKERS"
    --num_train_epochs "${EPOCHS:-5}"
    --checkpointing_steps "${CHECKPOINTING_STEPS:-1000}"
    --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-10}"
    --learning_rate "${LEARNING_RATE:-2e-5}"
    --lr_scheduler constant_with_warmup
    --lr_warmup_steps "${LR_WARMUP_STEPS:-100}"
    --seed "${SEED:-42}"
    --logging_dir tensorboard
    --report_to tensorboard
    --mixed_precision bf16
    --adam_weight_decay 3e-2
    --adam_epsilon 1e-10
    --vae_mini_batch 1
    --max_grad_norm 0.05
    --gradient_checkpointing
    --random_hw_adapt
    --training_with_video_token_length
    --enable_bucket
    --uniform_sampling
    --boundary_type full
    --add_inpaint_info
    --add_full_ref_image_in_self_attention
    --disable_moe
    --enable_method1_focused_loss
    --method1_loss_variant "$METHOD1_LOSS_VARIANT"
    --method1_action_dropout_prob "$METHOD1_ACTION_DROPOUT_PROB"
    --method1_tau_s "${METHOD1_TAU_S:-0.50}"
    --method1_eps "${METHOD1_EPS:-1e-6}"
    --method1_mse_threshold "${METHOD1_MSE_THRESHOLD:-0}"
    --method1_log_stats
    --output_dir "$OUTPUT_DIR"
    --trainable_modules .
    --require_all_transformer_trainable
)

if [[ "${METHOD1_SKIP_NONFINITE_UPDATES:-0}" == "1" ]]; then
    TRAIN_ARGS+=(
        --method1_skip_nonfinite_updates
        --method1_max_nonfinite_update_skips "${METHOD1_MAX_NONFINITE_UPDATE_SKIPS:-10}"
    )
fi

if [[ "$LOW_VRAM" == "1" ]]; then
    TRAIN_ARGS+=(--low_vram)
fi
if [[ "$SKIP_SANITY_CHECK" == "1" ]]; then
    TRAIN_ARGS+=(--skip_sanity_check)
fi
if [[ -n "${BENCHMARK_TIMING_PATH:-}" ]]; then
    TRAIN_ARGS+=(--benchmark_timing_path "$BENCHMARK_TIMING_PATH")
fi
if [[ -n "${METHOD1_SAMPLE_LOSS_DIR:-}" ]]; then
    TRAIN_ARGS+=(
        --method1_sample_loss_dir "$METHOD1_SAMPLE_LOSS_DIR"
        --require_method1_sample_loss_recording
    )
fi

if [[ "$USE_FSDP" == "1" ]]; then
    TRAIN_ARGS+=(--use_fsdp)
fi

if [[ -n "$MAX_TRAIN_STEPS" ]]; then
    TRAIN_ARGS+=(--max_train_steps "$MAX_TRAIN_STEPS")
fi
if [[ -n "$MAX_TRAIN_SAMPLES" ]]; then
    TRAIN_ARGS+=(--max_train_samples "$MAX_TRAIN_SAMPLES")
fi
if [[ -n "${TRANSFORMER_PATH:-}" ]]; then
    TRAIN_ARGS+=(--transformer_path "$TRANSFORMER_PATH")
    if [[ "$MODE" == "poseanything" ]]; then
        TRAIN_ARGS+=(--poseanything_resume_checkpoint)
    fi
fi
if [[ "$MODE" == "arm" ]]; then
    TRAIN_ARGS+=(--zero_init_arm_action_output)
    if [[ -z "${TRANSFORMER_PATH:-}" ]]; then
        TRAIN_ARGS+=(--require_arm_action_zero_init)
    fi
fi
if [[ "$MODE" == "camera" ]]; then
    TRAIN_ARGS+=(--zero_init_camera_adapter_output)
    if [[ -z "${TRANSFORMER_PATH:-}" ]]; then
        TRAIN_ARGS+=(--require_camera_adapter_zero_init)
    fi
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
fi
if [[ "$SMOKE" == "1" ]]; then
    TRAIN_ARGS+=(
        --skip_sanity_check
        --skip_final_checkpoint
        --method1_force_exit_after_training
        --require_cap_condition_gradient
    )
fi
TRAIN_ARGS+=("$@")

if [[ "$DRY_RUN" == "1" ]]; then
    printf 'CAP training command:'
    printf ' %q' "${RUN_CMD[@]}" "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

echo "Starting CAP mode=$MODE variant=$METHOD1_LOSS_VARIANT GPUs=$NUM_GPUS frames=$FRAMES resolution=${HEIGHT}x${WIDTH} per_device_batch=$TRAIN_BATCH_SIZE grad_accum=$GRAD_ACCUM effective_batch=$((NUM_GPUS * TRAIN_BATCH_SIZE * GRAD_ACCUM)) low_vram=$LOW_VRAM dataloader_workers_per_rank=$DATALOADER_WORKERS dataloader_workers_total=$((NUM_GPUS * DATALOADER_WORKERS))"
exec "${RUN_CMD[@]}" "${TRAIN_ARGS[@]}"
