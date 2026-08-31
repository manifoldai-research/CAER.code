#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG_FILE=${REPRO_CONFIG:-${ROOT_DIR}/config/paths.env}
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: missing configuration: ${CONFIG_FILE}" >&2
  echo "Create it from ${ROOT_DIR}/config/paths.env.example" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${CONFIG_FILE}"
set +a

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {libero|arm|camera|poseanything} [loss_variant] [extra training args]" >&2
  exit 2
fi
MODE=$1
shift
case "${MODE}" in
  libero) DEFAULT_VARIANT=CAER ;;
  arm|camera|poseanything) DEFAULT_VARIANT=CAER ;;
  *) echo "ERROR: unsupported mode: ${MODE}" >&2; exit 2 ;;
esac

LOSS_VARIANT=${1:-${DEFAULT_VARIANT}}
if [[ $# -gt 0 && "${1}" != --* ]]; then
  shift
fi
case "${LOSS_VARIANT}" in
  MSE|CAER) ;;
  *) echo "ERROR: unsupported loss variant: ${LOSS_VARIANT}" >&2; exit 2 ;;
esac
if [[ "${MODE}" == "libero" && "${LOSS_VARIANT}" != "CAER" && "${LOSS_VARIANT}" != "MSE" ]]; then
  echo "ERROR: LIBERO supports only MSE or CAER" >&2
  exit 2
fi

"${ROOT_DIR}/scripts/preflight.sh" "${MODE}"

RUN_ID=${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" && -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR=$(dirname "$(readlink -f "${RESUME_FROM_CHECKPOINT}")")
else
  OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_ROOT:?OUTPUT_ROOT is required}/${MODE}/${LOSS_VARIANT}/${RUN_ID}}
fi
export OUTPUT_DIR

if [[ "${MODE}" == "libero" ]]; then
  export ASSETS_DIR=${LIBERO_ASSETS_DIR:?LIBERO_ASSETS_DIR is required}
  export CONDA_ENV=$(cd "$(dirname "${LIBERO_PYTHON_BIN:?LIBERO_PYTHON_BIN is required}")/.." && pwd)
  export PYTHON_BIN=${LIBERO_PYTHON_BIN}
  export TORCHRUN_BIN=${LIBERO_TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}
  export TRAIN_DATA_DIR=${LIBERO_TRAIN_DATA_ROOT:-/}
  export CONFIG_PATH=${ROOT_DIR}/sources/libero/config/wan2.2/wan_civitai_5b.yaml
  export LIBERO_TI2V_LOSS_OUTPUT_ROOT=${OUTPUT_ROOT}/libero
  export LIBERO_TI2V_LOSS_RUN_ID=${RUN_ID}
  export LIBERO_TI2V_LOSS_RESUME_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
  export WAN22_CAMERA_MOE_ROOT=${LIBERO_ASSETS_DIR}
  exec bash "${ROOT_DIR}/sources/libero/scripts/wan2.2_fun/run_libero_ti2v_loss_volc.sh" "${LOSS_VARIANT}" "$@"
fi

export PYTHON_BIN=${CAP_PYTHON_BIN:?CAP_PYTHON_BIN is required}
export CONFIG_PATH=${ROOT_DIR}/sources/cap/config/wan2.2/wan_civitai_5b.yaml
export PYTHONPATH=${ROOT_DIR}/sources/cap/scripts/wan2.2_fun/method1_compat:${ROOT_DIR}/sources/cap:${PYTHONPATH:-}
export METHOD1_LOSS_VARIANT=${LOSS_VARIANT}
case "${MODE}" in
  arm)
    export CONTROL_MODEL=${CAP_CONTROL_MODEL:?CAP_CONTROL_MODEL is required}
    export TRAIN_DATA_META=${ARM_METADATA:?ARM_METADATA is required}
    export TRAIN_DATA_DIR=${ARM_DATA_ROOT:-/}
    EXTRA_ARGS=("$@")
    if [[ -n "${ARM_ACTION_STAT_PATH:-}" ]]; then
      EXTRA_ARGS+=(--arm_action_stat_path "${ARM_ACTION_STAT_PATH}")
    fi
    exec bash "${ROOT_DIR}/sources/cap/scripts/wan2.2_fun/run_cap_train.sh" arm "${EXTRA_ARGS[@]}"
    ;;
  camera)
    export CONTROL_MODEL=${CAP_CONTROL_MODEL:?CAP_CONTROL_MODEL is required}
    export TRAIN_DATA_META=${CAMERA_METADATA:?CAMERA_METADATA is required}
    export TRAIN_DATA_DIR=${CAMERA_DATA_ROOT:-/}
    exec bash "${ROOT_DIR}/sources/cap/scripts/wan2.2_fun/run_cap_train.sh" camera "$@"
    ;;
  poseanything)
    export POSE_MODEL=${POSE_BASE_MODEL:?POSE_BASE_MODEL is required}
    export PRETRAINED_MODEL=${POSE_BASE_MODEL}
    export TRAIN_DATA_META=${POSE_METADATA:?POSE_METADATA is required}
    export TRAIN_DATA_DIR=${POSE_DATA_ROOT:-/}
    exec bash "${ROOT_DIR}/sources/cap/scripts/wan2.2_fun/run_cap_train.sh" poseanything "$@"
    ;;
esac
