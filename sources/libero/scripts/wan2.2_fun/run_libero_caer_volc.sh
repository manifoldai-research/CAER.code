#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_libero_ti2v_loss_volc.sh {MSE|CAER}

One-line, single-node TI2V LIBERO arm training.

Modes:
  DRY_RUN=1  Validate and print the resolved launch without starting Python.
  SMOKE=1    Run one optimizer step on one GPU at 64x64; save no checkpoint.

Common overrides:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  OUTPUT_DIR=/absolute/new/output/directory
  LIBERO_TI2V_LOSS_OUTPUT_ROOT=/path/to/outputs/libero
  LIBERO_TI2V_LOSS_RESUME_CHECKPOINT=/absolute/run/checkpoint-N
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -ge 1 ]] && exit 0
  exit 2
fi

if [[ "$1" == "ti2v" ]]; then
  # Backward-compatible spelling for the original CAER-only launcher.
  shift
  set -- CAER "$@"
fi
LOSS_VARIANT=$1
shift
case "${LOSS_VARIANT}" in
  MSE|CAER) ;;
  *)
    echo "ERROR: loss must be MSE or CAER; got: ${LOSS_VARIANT}" >&2
    exit 2
    ;;
esac
INIT_KIND=ti2v

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
CORE_LAUNCHER="${SCRIPT_DIR}/train_singleffn_libero_single_arm_ablation_5b_8gpu.sh"
ASSETS_DIR=${ASSETS_DIR:-${PROJECT_ROOT}/datasets/libero_single_arm_ablation}
DRY_RUN=${DRY_RUN:-0}
SMOKE=${SMOKE:-0}

for binary_flag in DRY_RUN SMOKE; do
  value=${!binary_flag}
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "ERROR: ${binary_flag} must be 0 or 1; got: ${value}" >&2
    exit 2
  fi
done
if [[ ! -f "${CORE_LAUNCHER}" ]]; then
  echo "ERROR: missing core launcher: ${CORE_LAUNCHER}" >&2
  exit 1
fi

WORKER_NUM=${MLP_WORKER_NUM:-1}
WORKER_RANK=${MLP_ROLE_INDEX:-0}
if [[ "${WORKER_NUM}" != "1" || "${WORKER_RANK}" != "0" ]]; then
  echo "ERROR: use one 8-GPU Volcano node per run; got MLP_WORKER_NUM=${WORKER_NUM}, MLP_ROLE_INDEX=${WORKER_RANK}" >&2
  exit 1
fi

required_assets=(
  "${ASSETS_DIR}/metadata_libero_single_arm_train.json"
  "${ASSETS_DIR}/stat.json"
  "${ASSETS_DIR}/ti2v_control_init_model/config.json"
)
for asset in "${required_assets[@]}"; do
  if [[ ! -s "${asset}" ]]; then
    echo "ERROR: required LIBERO asset is missing or empty: ${asset}" >&2
    echo "Prepare the assets first as described in docs/TRAINING.md." >&2
    exit 1
  fi
done

RUN_ID=${LIBERO_TI2V_LOSS_RUN_ID:-$(date -u +%Y%m%dT%H%M%S%NZ)}
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: LIBERO_TI2V_LOSS_RUN_ID contains unsupported characters: ${RUN_ID}" >&2
  exit 1
fi
OUTPUT_ROOT=${LIBERO_TI2V_LOSS_OUTPUT_ROOT:-${PROJECT_ROOT}/../../outputs/libero}
RESUME_CHECKPOINT=${LIBERO_TI2V_LOSS_RESUME_CHECKPOINT:-}
OUTPUT_DIR_WAS_SET=${OUTPUT_DIR+x}

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ "${SMOKE}" == "1" ]]; then
    echo "ERROR: SMOKE=1 cannot resume a formal run" >&2
    exit 1
  fi
  if [[ "${RESUME_CHECKPOINT}" != /* || ! -d "${RESUME_CHECKPOINT}" ]]; then
    echo "ERROR: resume checkpoint must be an existing absolute directory: ${RESUME_CHECKPOINT}" >&2
    exit 1
  fi
  checkpoint_name=$(basename "${RESUME_CHECKPOINT}")
  if [[ ! "${checkpoint_name}" =~ ^checkpoint-[1-9][0-9]*$ ]]; then
    echo "ERROR: resume path must end in checkpoint-N: ${RESUME_CHECKPOINT}" >&2
    exit 1
  fi
  resolved_output=$(dirname "$(readlink -f "${RESUME_CHECKPOINT}")")
  if [[ -n "${OUTPUT_DIR:-}" && "$(readlink -m "${OUTPUT_DIR}")" != "${resolved_output}" ]]; then
    echo "ERROR: OUTPUT_DIR must be the checkpoint parent when resuming: ${resolved_output}" >&2
    exit 1
  fi
  OUTPUT_DIR=${resolved_output}
  RESUME_FROM_CHECKPOINT=$(readlink -f "${RESUME_CHECKPOINT}")
else
  if [[ "${SMOKE}" == "1" && -z "${OUTPUT_DIR_WAS_SET}" ]]; then
    OUTPUT_DIR=/tmp/libero-ti2v-${LOSS_VARIANT}-smoke-${RUN_ID}
  else
    OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT}/${LOSS_VARIANT}/${RUN_ID}}
  fi
  if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "ERROR: refusing to reuse nonempty output directory: ${OUTPUT_DIR}" >&2
    exit 1
  fi
  RESUME_FROM_CHECKPOINT=
fi

if [[ "${SMOKE}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}
  NPROC_PER_NODE=1
  MAX_TRAIN_STEPS=1
  CHECKPOINTING_STEPS=999999
  CHECKPOINTS_TOTAL_LIMIT=1
  GRADIENT_ACCUMULATION_STEPS=1
  DATALOADER_NUM_WORKERS=0
  IMAGE_SAMPLE_SIZE=64
  VIDEO_SAMPLE_SIZE=64
  TOKEN_SAMPLE_SIZE=64
else
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
  NPROC_PER_NODE=8
  IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  if [[ ${#visible_gpus[@]} -ne 8 ]]; then
    echo "ERROR: formal training requires exactly 8 visible GPUs; got: ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
fi

export INIT_KIND LOSS_VARIANT RUN_ID OUTPUT_DIR RESUME_FROM_CHECKPOINT
export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
export PREPARE_ASSETS=0
export ASSETS_DIR
export RUN_TAG="libero-ti2v-${LOSS_VARIANT}-${RUN_ID}"
export TMP_LOG_ROOT="${OUTPUT_DIR}/runtime"
export LOG_DIR="${OUTPUT_DIR}/logs"
export TRAIN_LOGGING_DIR="${OUTPUT_DIR}/tensorboard"
export MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-5000}
export CHECKPOINTING_STEPS=${CHECKPOINTING_STEPS:-1000}
export CHECKPOINTS_TOTAL_LIMIT=${CHECKPOINTS_TOTAL_LIMIT:-10}
export GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export SEED=${SEED:-42}
export MIXED_PRECISION=${MIXED_PRECISION:-bf16}

# These are the controlled Method1 settings for this launcher.
export ENABLE_METHOD1_FOCUSED_LOSS=1
export METHOD1_LOSS_VARIANT="${LOSS_VARIANT}"
export METHOD1_ACTION_DROPOUT_PROB=0.10
export METHOD1_TAU_S=0.50
export METHOD1_EPS=1e-6
export METHOD1_MSE_THRESHOLD=0.0
export METHOD1_LOG_STATS=1
export ZERO_INIT_ARM_ACTION_OUTPUT=1

NODE_CACHE_ROOT=${LIBERO_TI2V_LOSS_NODE_CACHE_ROOT:-/dev/shm/libero-ti2v-${LOSS_VARIANT}-${UID}-${RUN_ID}}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${NODE_CACHE_ROOT}/xdg}
export HF_HOME=${HF_HOME:-${NODE_CACHE_ROOT}/huggingface}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${NODE_CACHE_ROOT}/triton}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-${NODE_CACHE_ROOT}/torch_extensions}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-${NODE_CACHE_ROOT}/pycache}
export TMPDIR=${TMPDIR:-${NODE_CACHE_ROOT}/tmp}
mkdir -p "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}" "${PYTHONPYCACHEPREFIX}" "${TMPDIR}"

EXTRA_ARGS=("$@")
if [[ "${SMOKE}" == "1" ]]; then
  EXTRA_ARGS+=(
    --fix_sample_size 64 64
    --max_train_samples 1
    --skip_sanity_check
    --skip_final_checkpoint
  )
fi

echo "LIBERO TI2V loss launch"
echo "  loss=${LOSS_VARIANT} smoke=${SMOKE} dry_run=${DRY_RUN}"
echo "  gpus=${CUDA_VISIBLE_DEVICES} processes=${NPROC_PER_NODE}"
echo "  output=${OUTPUT_DIR}"
echo "  log=${LOG_DIR}"
echo "  loss=${LOSS_VARIANT} dropout=0.10 sigma=0.50 eps=1e-6 future_only=1 zero_init_arm=1"
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  echo "  resume=${RESUME_FROM_CHECKPOINT}"
fi

COMMAND=(bash "${CORE_LAUNCHER}" "${EXTRA_ARGS[@]}")
if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'command='
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

exec "${COMMAND[@]}"
