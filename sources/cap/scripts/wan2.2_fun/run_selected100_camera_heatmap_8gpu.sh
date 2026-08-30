#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${CAP_CAMERA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
HOST_PROJECT_DIR="${CAP_CAMERA_HOST_PROJECT_ROOT:-$PROJECT_DIR}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
SCRIPT_PATH="${PROJECT_DIR}/scripts/wan2.2_fun/infer_selected100_camera_heatmap.py"
MERGE_SCRIPT="${PROJECT_DIR}/scripts/wan2.2_fun/merge_selected100_camera_heatmap_results.py"
MODEL_NAME="${MODEL_NAME:-${CAP_CONTROL_MODEL:?Set MODEL_NAME or CAP_CONTROL_MODEL}}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config/wan2.2/wan_civitai_5b.yaml}"
SELECTED_CSV="${SELECTED_CSV:?Set SELECTED_CSV to the Camera-100 case list}"
CAMERA_ROOT="${CAMERA_ROOT:?Set CAMERA_ROOT to the camera trajectory directory}"
ACTION_CSV="${ACTION_CSV:?Set ACTION_CSV to the camera action CSV}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
SAMPLE_COUNT="${SAMPLE_COUNT:-100}"
START_INDEX="${START_INDEX:-0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
FRAME_STRIDE="${FRAME_STRIDE:-4}"
HEATMAP_OUTPUT_FORMAT="${HEATMAP_OUTPUT_FORMAT:-frames}"
SKIP_COMPLETE="${SKIP_COMPLETE:-1}"

if [[ -z "${CHECKPOINT_PATH:-}" || -z "${OUTPUT_ROOT:-}" ]]; then
  echo "ERROR: CHECKPOINT_PATH and OUTPUT_ROOT are required" >&2
  exit 1
fi
if [[ ! -s "${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint missing or empty: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ "${SAMPLE_COUNT}" -ne 100 || "${START_INDEX}" -ne 0 ]]; then
  echo "ERROR: this launcher requires SAMPLE_COUNT=100 START_INDEX=0" >&2
  exit 1
fi
if [[ "${HEATMAP_OUTPUT_FORMAT}" != "frames" && "${HEATMAP_OUTPUT_FORMAT}" != "video" ]]; then
  echo "ERROR: HEATMAP_OUTPUT_FORMAT must be frames or video" >&2
  exit 1
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export PYTHONPATH="${PROJECT_DIR}:${HOST_PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -ne 8 ]]; then
  echo "ERROR: GPU_LIST must contain exactly 8 GPUs" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/logs"
PIDS=()
SHARD_IDS=()
CURRENT_START=0
BASE_COUNT=$((SAMPLE_COUNT / 8))
REMAINDER=$((SAMPLE_COUNT % 8))

terminate_children() {
  local pid
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap 'terminate_children; exit 130' INT TERM

for shard_index in "${!GPUS[@]}"; do
  shard_count="${BASE_COUNT}"
  if [[ "${shard_index}" -lt "${REMAINDER}" ]]; then
    shard_count=$((shard_count + 1))
  fi
  gpu="${GPUS[${shard_index}]}"
  shard_id="$(printf '%02d' "${shard_index}")"
  log_path="${OUTPUT_ROOT}/logs/shard${shard_id}_gpu${gpu}.log"
  skip_args=()
  if [[ "${SKIP_COMPLETE}" == "1" ]]; then
    skip_args+=(--skip_complete)
  fi
  echo "[$(date -u +%FT%TZ)] launch shard=${shard_id} gpu=${gpu} start=${CURRENT_START} count=${shard_count}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u "${SCRIPT_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --model_name "${MODEL_NAME}" \
    --config_path "${CONFIG_PATH}" \
    --selected_csv "${SELECTED_CSV}" \
    --camera_root "${CAMERA_ROOT}" \
    --action_csv "${ACTION_CSV}" \
    --output_dir "${OUTPUT_ROOT}" \
    --device cuda:0 \
    --sample_count "${shard_count}" \
    --start_index "${CURRENT_START}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --frame_stride "${FRAME_STRIDE}" \
    --heatmap_output_format "${HEATMAP_OUTPUT_FORMAT}" \
    --result_file_name "results_shard${shard_id}.json" \
    --selection_file_name "selection_shard${shard_id}.json" \
    --camera_moe_root "" \
    --defer_root_manifest \
    "${skip_args[@]}" \
    > "${log_path}" 2>&1 &
  PIDS+=("$!")
  SHARD_IDS+=("${shard_id}")
  CURRENT_START=$((CURRENT_START + shard_count))
done

STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[${index}]}"; then
    echo "[$(date -u +%FT%TZ)] shard=${SHARD_IDS[${index}]} complete"
  else
    code="$?"
    echo "[$(date -u +%FT%TZ)] shard=${SHARD_IDS[${index}]} failed exit=${code}" >&2
    STATUS=1
  fi
done

if ! "${PYTHON_BIN}" "${MERGE_SCRIPT}" \
  --output-dir "${OUTPUT_ROOT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --expected-count "${SAMPLE_COUNT}"; then
  STATUS=1
fi

exit "${STATUS}"
