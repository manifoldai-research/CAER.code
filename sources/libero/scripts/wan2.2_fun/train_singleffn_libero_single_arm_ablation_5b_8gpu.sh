#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
EXTRA_ARGS=("$@")
if [[ -n "${CONDA_ENV:-}" ]]; then
  PYTHON_BIN=${PYTHON_BIN:-${CONDA_ENV}/bin/python}
  TORCHRUN_BIN=${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}
  export PATH="${CONDA_ENV}/bin:${PATH}"
else
  PYTHON_BIN=${PYTHON_BIN:-$(command -v python3 || command -v python || true)}
  TORCHRUN_BIN=${TORCHRUN_BIN:-$(command -v torchrun || true)}
fi
if [[ -z "${TORCHRUN_BIN}" ]]; then
  echo "ERROR: set TORCHRUN_BIN or install torchrun on PATH" >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  NVCC_PATH=$(command -v nvcc 2>/dev/null || true)
  if [[ -n "${NVCC_PATH}" ]]; then
    export CUDA_HOME="$(dirname "$(dirname "${NVCC_PATH}")")"
  elif [[ -d "/usr/local/cuda" ]]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
export DS_BUILD_OPS=${DS_BUILD_OPS:-0}

INIT_KIND=${INIT_KIND:-shared}
if [[ "${INIT_KIND}" != "shared" && "${INIT_KIND}" != "ti2v" ]]; then
  echo "INIT_KIND must be shared or ti2v, got: ${INIT_KIND}" >&2
  exit 1
fi

MODEL_NAME=${MODEL_NAME:-}
CONTROL_TEMPLATE_MODEL_NAME=${CONTROL_TEMPLATE_MODEL_NAME:-}
ASSETS_DIR=${ASSETS_DIR:-${PROJECT_ROOT}/datasets/libero_single_arm_ablation}
LIBERO_TRAIN_ROOT=${LIBERO_TRAIN_ROOT:-}
LIBERO_VAL_ROOT=${LIBERO_VAL_ROOT:-}
SHARED_MOE_CHECKPOINT=${SHARED_MOE_CHECKPOINT:-}
SHARED_EXPORT_MODE=${SHARED_EXPORT_MODE:-ffn_only}
LIBERO_WINDOW_SIZE=${LIBERO_WINDOW_SIZE:-17}
LIBERO_VIDEO_SAMPLE_STRIDE=${LIBERO_VIDEO_SAMPLE_STRIDE:-1}
LIBERO_METADATA_STRIDE=${LIBERO_METADATA_STRIDE:-1}
ASSETS_READY=1
[[ -f "${ASSETS_DIR}/metadata_libero_single_arm_train.json" ]] || ASSETS_READY=0
[[ -f "${ASSETS_DIR}/metadata_libero_single_arm_val.json" ]] || ASSETS_READY=0
[[ -f "${ASSETS_DIR}/stat.json" ]] || ASSETS_READY=0
[[ -f "${ASSETS_DIR}/ti2v_control_init_model/config.json" ]] || ASSETS_READY=0
if [[ "${INIT_KIND}" == "shared" && ! -f "${ASSETS_DIR}/checkpoints/shared_expert_single_ffn_from_moe.safetensors" ]]; then
  ASSETS_READY=0
fi
if [[ -z "${PREPARE_ASSETS+x}" ]]; then
  if [[ "${ASSETS_READY}" == "1" ]]; then
    PREPARE_ASSETS=0
  else
    PREPARE_ASSETS=1
  fi
fi

ROLE_INDEX=${MLP_ROLE_INDEX:-${RANK:-0}}
SKIP_SHARED_EXPORT_ARG=()
if [[ "${INIT_KIND}" == "ti2v" ]]; then
  SKIP_SHARED_EXPORT_ARG=(--skip_shared_export)
fi
if [[ "${PREPARE_ASSETS}" == "1" ]]; then
  if [[ -z "${LIBERO_TRAIN_ROOT}" || -z "${MODEL_NAME}" || -z "${CONTROL_TEMPLATE_MODEL_NAME}" ]]; then
    echo "ERROR: asset preparation requires LIBERO_TRAIN_ROOT, MODEL_NAME, and CONTROL_TEMPLATE_MODEL_NAME" >&2
    exit 1
  fi
  if [[ "${INIT_KIND}" == "shared" && -z "${SHARED_MOE_CHECKPOINT}" ]]; then
    echo "ERROR: shared init requires SHARED_MOE_CHECKPOINT" >&2
    exit 1
  fi
  if [[ "${ROLE_INDEX}" == "0" ]]; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_libero_single_arm_ablation_assets.py" \
      --assets_dir "${ASSETS_DIR}" \
      --train_root "${LIBERO_TRAIN_ROOT}" \
      --val_root "${LIBERO_VAL_ROOT}" \
      --shared_moe_checkpoint "${SHARED_MOE_CHECKPOINT}" \
      --ti2v_model_dir "${MODEL_NAME}" \
      --control_template_model_dir "${CONTROL_TEMPLATE_MODEL_NAME}" \
      --shared_export_mode "${SHARED_EXPORT_MODE}" \
      --window_size "${LIBERO_WINDOW_SIZE}" \
      --video_sample_stride "${LIBERO_VIDEO_SAMPLE_STRIDE}" \
      --metadata_stride "${LIBERO_METADATA_STRIDE}" \
      "${SKIP_SHARED_EXPORT_ARG[@]}"
  else
    MANIFEST_PATH="${ASSETS_DIR}/assets_manifest.json"
    until [[ -f "${MANIFEST_PATH}" ]]; do
      echo "Waiting for rank 0 to prepare ${MANIFEST_PATH} ..."
      sleep 10
    done
  fi
fi

TI2V_CONTROL_MODEL_NAME=${TI2V_CONTROL_MODEL_NAME:-${ASSETS_DIR}/ti2v_control_init_model}
SHARED_FFN_CHECKPOINT=${SHARED_FFN_CHECKPOINT:-${ASSETS_DIR}/checkpoints/shared_expert_single_ffn_from_moe.safetensors}
DATASET_META_NAME=${DATASET_META_NAME:-${ASSETS_DIR}/metadata_libero_single_arm_train.json}
ARM_ACTION_STAT_PATH=${ARM_ACTION_STAT_PATH:-${ASSETS_DIR}/stat.json}

if [[ "${INIT_KIND}" == "shared" ]]; then
  RUN_TAG=${RUN_TAG:-singleffn-shared-libero-single-arm-8gpu}
  TRANSFORMER_PATH=${TRANSFORMER_PATH:-${SHARED_FFN_CHECKPOINT}}
else
  RUN_TAG=${RUN_TAG:-singleffn-ti2v-libero-single-arm-8gpu}
  TRANSFORMER_PATH=${TRANSFORMER_PATH:-}
fi
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/../../outputs/videox-fun-5B-${RUN_TAG}}
TMP_LOG_ROOT=${TMP_LOG_ROOT:-${OUTPUT_DIR}/runtime}
LOG_DIR=${LOG_DIR:-${TMP_LOG_ROOT}/logs}
TRAIN_LOGGING_DIR=${TRAIN_LOGGING_DIR:-${TMP_LOG_ROOT}/tensorboard}
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}" "${TRAIN_LOGGING_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE=${LOG_FILE:-${LOG_DIR}/train_${RUN_TAG}_${TIMESTAMP}_node${MLP_ROLE_INDEX:-0}.log}
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "============================================="
echo "INIT_KIND=${INIT_KIND}"
echo "RUN_TAG=${RUN_TAG}"
echo "PREPARE_ASSETS=${PREPARE_ASSETS}"
echo "LIBERO_TRAIN_ROOT=${LIBERO_TRAIN_ROOT}"
echo "TI2V_CONTROL_MODEL_NAME=${TI2V_CONTROL_MODEL_NAME}"
echo "TRANSFORMER_PATH=${TRANSFORMER_PATH:-<none>}"
echo "DATASET_META_NAME=${DATASET_META_NAME}"
echo "ARM_ACTION_STAT_PATH=${ARM_ACTION_STAT_PATH}"
if [[ "${INIT_KIND}" == "shared" ]]; then
  echo "SHARED_MOE_CHECKPOINT=${SHARED_MOE_CHECKPOINT}"
fi
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "TRAIN_LOGGING_DIR=${TRAIN_LOGGING_DIR}"
echo "LOG_FILE=${LOG_FILE}"
echo "ENABLE_METHOD1_FOCUSED_LOSS=${ENABLE_METHOD1_FOCUSED_LOSS:-0}"
echo "ZERO_INIT_ARM_ACTION_OUTPUT=${ZERO_INIT_ARM_ACTION_OUTPUT:-0}"
if [[ "${ENABLE_METHOD1_FOCUSED_LOSS:-0}" == "1" ]]; then
  echo "METHOD1_LOSS_VARIANT=${METHOD1_LOSS_VARIANT:-s_max1}"
  echo "METHOD1_ACTION_DROPOUT_PROB=${METHOD1_ACTION_DROPOUT_PROB:-0.10}"
  echo "METHOD1_TAU_S=${METHOD1_TAU_S:-0.50}"
  echo "METHOD1_EPS=${METHOD1_EPS:-1e-6}"
fi
echo "CUDA_HOME=${CUDA_HOME:-not set}"
echo "============================================="

export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_14:1,mlx5_15:1,mlx5_16:1,mlx5_17:1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

export TRAINABLE_MODULES=${TRAINABLE_MODULES:-"ffn arm_action_embedder arm_condition_mask_emb"}
export TRAINABLE_MODULES_LOW_LEARNING_RATE=${TRAINABLE_MODULES_LOW_LEARNING_RATE:-"ffn.control_moe.shared_expert control_adapter"}
export FREEZE_CONTROL_ADAPTER=${FREEZE_CONTROL_ADAPTER:-0}
read -r -a TRAINABLE_MODULES_VALUES <<< "${TRAINABLE_MODULES}"
read -r -a TRAINABLE_MODULES_LOW_VALUES <<< "${TRAINABLE_MODULES_LOW_LEARNING_RATE}"
TRAINABLE_MODULES_ARG=(--trainable_modules "${TRAINABLE_MODULES_VALUES[@]}")
TRAINABLE_MODULES_LOW_ARG=()
if [[ ${#TRAINABLE_MODULES_LOW_VALUES[@]} -gt 0 ]]; then
  TRAINABLE_MODULES_LOW_ARG=(--trainable_modules_low_learning_rate "${TRAINABLE_MODULES_LOW_VALUES[@]}")
fi
FREEZE_CONTROL_ADAPTER_ARG=()
if [[ "${FREEZE_CONTROL_ADAPTER}" == "1" ]]; then
  FREEZE_CONTROL_ADAPTER_ARG=(--freeze_control_adapter)
fi

ENABLE_METHOD1_FOCUSED_LOSS=${ENABLE_METHOD1_FOCUSED_LOSS:-0}
METHOD1_ARGS=()
if [[ "${ENABLE_METHOD1_FOCUSED_LOSS}" == "1" ]]; then
  METHOD1_ARGS=(
    --enable_method1_focused_loss
    --method1_loss_variant "${METHOD1_LOSS_VARIANT:-s_max1}"
    --method1_action_dropout_prob "${METHOD1_ACTION_DROPOUT_PROB:-0.10}"
    --method1_tau_s "${METHOD1_TAU_S:-0.50}"
    --method1_eps "${METHOD1_EPS:-1e-6}"
    --method1_mse_threshold "${METHOD1_MSE_THRESHOLD:-0.0}"
  )
  if [[ "${METHOD1_LOG_STATS:-1}" == "1" ]]; then
    METHOD1_ARGS+=(--method1_log_stats)
  fi
elif [[ "${ENABLE_METHOD1_FOCUSED_LOSS}" != "0" ]]; then
  echo "ENABLE_METHOD1_FOCUSED_LOSS must be 0 or 1, got: ${ENABLE_METHOD1_FOCUSED_LOSS}" >&2
  exit 1
fi

ZERO_INIT_ARM_ACTION_OUTPUT=${ZERO_INIT_ARM_ACTION_OUTPUT:-0}
ZERO_INIT_ARM_ACTION_ARG=()
if [[ "${ZERO_INIT_ARM_ACTION_OUTPUT}" == "1" ]]; then
  ZERO_INIT_ARM_ACTION_ARG=(--zero_init_arm_action_output)
elif [[ "${ZERO_INIT_ARM_ACTION_OUTPUT}" != "0" ]]; then
  echo "ZERO_INIT_ARM_ACTION_OUTPUT must be 0 or 1, got: ${ZERO_INIT_ARM_ACTION_OUTPUT}" >&2
  exit 1
fi

export WAN22_MOE_MODE=${WAN22_MOE_MODE:-${MOE_MODE:-control_expert}}
export WAN22_MOE_ALL_BLOCKS=${WAN22_MOE_ALL_BLOCKS:-${MOE_ALL_BLOCKS:-1}}
export WAN22_MOE_VERBOSE=${WAN22_MOE_VERBOSE:-0}
export WAN22_CAMERA_MOE_ROOT=${WAN22_CAMERA_MOE_ROOT:-${CAMERA_MOE_ROOT:-}}
export WAN22_ACTION_MAP_DEBUG=${WAN22_ACTION_MAP_DEBUG:-0}
MOE_ALL_BLOCKS_ARG=()
if [[ "${WAN22_MOE_ALL_BLOCKS}" == "1" ]]; then
  MOE_ALL_BLOCKS_ARG=(--moe_all_blocks)
fi

RANK_ENV=${MLP_ROLE_INDEX:-0}
MASTER_ADDR_ENV=${MLP_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT_ENV=${MLP_WORKER_0_PORT:-${MASTER_PORT:-29503}}
NNODES_ENV=${MLP_WORKER_NUM:-1}
NPROC_PER_NODE_ENV=${NPROC_PER_NODE:-8}

RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
TRANSFORMER_PATH_ARG=()
if [[ -n "${TRANSFORMER_PATH}" ]]; then
  TRANSFORMER_PATH_ARG=(--transformer_path "${TRANSFORMER_PATH}")
fi
RESUME_ARG=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARG=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

cd "${PROJECT_ROOT}"
"${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE_ENV}" \
  --nnodes="${NNODES_ENV}" \
  --node_rank="${RANK_ENV}" \
  --master_addr="${MASTER_ADDR_ENV}" \
  --master_port="${MASTER_PORT_ENV}" \
  "${SCRIPT_DIR}/train_control_camera_arm_actionmap.py" \
  --pretrained_model_name_or_path "${TI2V_CONTROL_MODEL_NAME}" \
  --config_path "${CONFIG_PATH:-${PROJECT_ROOT}/config/wan2.2/wan_civitai_5b.yaml}" \
  --train_data_meta "${DATASET_META_NAME}" \
  --train_data_dir "${TRAIN_DATA_DIR:-/}" \
  --train_mode "${TRAIN_MODE:-control_camera_ref}" \
  --control_ref_image "${CONTROL_REF_IMAGE:-first_frame}" \
  --image_sample_size "${IMAGE_SAMPLE_SIZE:-256}" \
  --video_sample_size "${VIDEO_SAMPLE_SIZE:-256}" \
  --token_sample_size "${TOKEN_SAMPLE_SIZE:-256}" \
  --video_sample_stride "${VIDEO_SAMPLE_STRIDE:-1}" \
  --video_sample_n_frames "${VIDEO_SAMPLE_N_FRAMES:-17}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --video_repeat "${VIDEO_REPEAT:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-10}" \
  --max_train_steps "${MAX_TRAIN_STEPS:-5000}" \
  --checkpointing_steps "${CHECKPOINTING_STEPS:-1000}" \
  --checkpoints_total_limit "${CHECKPOINTS_TOTAL_LIMIT:-10}" \
  --learning_rate "${LEARNING_RATE:-2e-05}" \
  --lr_scheduler "${LR_SCHEDULER:-constant_with_warmup}" \
  --lr_warmup_steps "${LR_WARMUP_STEPS:-100}" \
  --seed "${SEED:-42}" \
  --logging_dir "${TRAIN_LOGGING_DIR:-tensorboard}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  --adam_weight_decay "${ADAM_WEIGHT_DECAY:-3e-2}" \
  --adam_epsilon "${ADAM_EPSILON:-1e-10}" \
  --vae_mini_batch "${VAE_MINI_BATCH:-1}" \
  --max_grad_norm "${MAX_GRAD_NORM:-0.05}" \
  --gradient_checkpointing \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --boundary_type "${BOUNDARY_TYPE:-full}" \
  --add_inpaint_info \
  --use_fsdp \
  --enable_arm_info \
  --arm_action_key "${ARM_ACTION_KEY:-state}" \
  --arm_action_dim "${ARM_ACTION_DIM:-7}" \
  --arm_action_num_frames "${ARM_ACTION_NUM_FRAMES:-17}" \
  --arm_action_stat_path "${ARM_ACTION_STAT_PATH}" \
  --moe_mode "${WAN22_MOE_MODE}" \
  "${MOE_ALL_BLOCKS_ARG[@]}" \
  --camera_moe_root "${WAN22_CAMERA_MOE_ROOT}" \
  --disable_moe \
  "${METHOD1_ARGS[@]}" \
  "${ZERO_INIT_ARM_ACTION_ARG[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  "${TRAINABLE_MODULES_ARG[@]}" \
  "${TRAINABLE_MODULES_LOW_ARG[@]}" \
  "${FREEZE_CONTROL_ADAPTER_ARG[@]}" \
  "${TRANSFORMER_PATH_ARG[@]}" \
  "${RESUME_ARG[@]}" \
  "${EXTRA_ARGS[@]}"
