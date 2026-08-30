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

MODE=${1:-all}
case "${MODE}" in
  libero|arm|camera|poseanything) MODES=("${MODE}") ;;
  all) MODES=(libero arm camera poseanything) ;;
  *) echo "Usage: $0 {libero|arm|camera|poseanything|all}" >&2; exit 2 ;;
esac

for selected_mode in "${MODES[@]}"; do
  if [[ "${selected_mode}" == "libero" ]]; then
    runtime=${LIBERO_PYTHON_BIN:-}
  else
    runtime=${CAP_PYTHON_BIN:-}
  fi
  if [[ ! -x "${runtime}" ]]; then
    echo "ERROR: configured Python is not executable for ${selected_mode}: ${runtime}" >&2
    exit 1
  fi
  check_python=${PREFLIGHT_PYTHON_BIN:-python3}
  timeout_seconds=${PREFLIGHT_TIMEOUT_SECONDS:-120}
  timeout "${timeout_seconds}" "${check_python}" "${ROOT_DIR}/scripts/preflight.py" "${selected_mode}"
  echo "PYTHON EXECUTABLE OK mode=${selected_mode} python=${runtime}"
done
