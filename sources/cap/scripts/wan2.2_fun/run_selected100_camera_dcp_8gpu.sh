#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR=${CAP_CAMERA_PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
HOST_PROJECT_DIR=${CAP_CAMERA_HOST_PROJECT_ROOT:-$PROJECT_DIR}
if [[ -z "${CAP_CAMERA_PYTHON:-${PYTHON_BIN:-}}" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
else
  PYTHON_BIN=${CAP_CAMERA_PYTHON:-$PYTHON_BIN}
fi
SCRIPT_PATH="$PROJECT_DIR/scripts/wan2.2_fun/infer_selected100_camera_dcp.py"
MODEL_NAME=${MODEL_NAME:-${CAP_CONTROL_MODEL:?Set MODEL_NAME or CAP_CONTROL_MODEL}}
CONFIG_PATH=${CONFIG_PATH:-$PROJECT_DIR/config/wan2.2/wan_civitai_5b.yaml}
SELECTED_CSV=${SELECTED_CSV:?Set SELECTED_CSV to the Camera-100 case list}
CAMERA_ROOT=${CAMERA_ROOT:?Set CAMERA_ROOT to the camera trajectory directory}
ACTION_CSV=${ACTION_CSV:?Set ACTION_CSV to the camera action CSV}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}
SAMPLE_COUNT=${SAMPLE_COUNT:-100}
START_INDEX=${START_INDEX:-0}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
SAMPLE_HEIGHT=${SAMPLE_HEIGHT:-704}
SAMPLE_WIDTH=${SAMPLE_WIDTH:-1280}
START_IMAGE_CENTER_CROP_SIZE=${START_IMAGE_CENTER_CROP_SIZE:-720}
VIDEO_LENGTH=${VIDEO_LENGTH:-81}
FPS=${FPS:-24}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-6.0}
SEED=${SEED:-50025}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}
WORKER_STAGGER_SECONDS=${CAP_CAMERA_WORKER_STAGGER_SECONDS:-5}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -eq 2 ]] || die "usage: $0 CHECKPOINT_DIR OUTPUT_ROOT"
CHECKPOINT=$1
OUTPUT_ROOT=$2

[[ -x "$PYTHON_BIN" ]] || die "Python is not executable: $PYTHON_BIN"
[[ -f "$SCRIPT_PATH" ]] || die "inference script is missing: $SCRIPT_PATH"
[[ -f "$CONFIG_PATH" ]] || die "config is missing: $CONFIG_PATH"
[[ -f "$SELECTED_CSV" ]] || die "Camera-100 CSV is missing: $SELECTED_CSV"
[[ -d "$CAMERA_ROOT" ]] || die "camera trajectory root is missing: $CAMERA_ROOT"
[[ -d "$CHECKPOINT/pytorch_model_fsdp_0" ]] || die "FSDP weights are missing: $CHECKPOINT"
[[ -s "$CHECKPOINT/pytorch_model_fsdp_0/.metadata" ]] || die "FSDP metadata is missing: $CHECKPOINT"
[[ -s "$CHECKPOINT/scheduler.bin" ]] || die "scheduler state is missing: $CHECKPOINT"
mapfile -t DCP_SHARDS < <(find "$CHECKPOINT/pytorch_model_fsdp_0" -maxdepth 1 -type f -name '*.distcp' -size +0c | sort)
(( ${#DCP_SHARDS[@]} == 8 )) || die "expected 8 non-empty DCP shards, found ${#DCP_SHARDS[@]}: $CHECKPOINT"
[[ "$SAMPLE_COUNT" == 100 && "$START_INDEX" == 0 ]] || die 'Camera-100 requires SAMPLE_COUNT=100 and START_INDEX=0'
[[ "$SAMPLE_HEIGHT" == 704 && "$SAMPLE_WIDTH" == 1280 ]] || die 'Camera-100 requires 704x1280 output'
[[ "$START_IMAGE_CENTER_CROP_SIZE" == 720 && "$VIDEO_LENGTH" == 81 && "$FPS" == 24 ]] \
  || die 'Camera-100 requires crop=720, video_length=81, and fps=24'
[[ "$NUM_INFERENCE_STEPS" == 50 && "$GUIDANCE_SCALE" == 6.0 && "$SEED" == 50025 ]] \
  || die 'formal Camera-100 requires steps=50, guidance=6.0, and seed=50025'
[[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+){7}$ ]] || die "exactly 8 GPU ids are required: $GPU_LIST"
[[ "$WORKER_STAGGER_SECONDS" =~ ^[0-9]+$ ]] || die "invalid worker stagger: $WORKER_STAGGER_SECONDS"

export TMPDIR=/tmp TMP=/tmp TEMP=/tmp CAP_WORLDARENA_TMPDIR=/tmp
[[ ${#CAP_WORLDARENA_TMPDIR} -le 70 ]] || die "CAP_WORLDARENA_TMPDIR is too long"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$PROJECT_DIR:$HOST_PROJECT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 VIDEOX_ATTENTION_TYPE=${VIDEOX_ATTENTION_TYPE:-SDPA}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
unset PYTHONHOME

if [[ -n "${CAP_CAMERA_FFMPEG_SOURCE:-}" ]]; then
  FFMPEG_SOURCE=$CAP_CAMERA_FFMPEG_SOURCE
  FFMPEG_TARGET=${CAP_CAMERA_FFMPEG_TARGET:-/tmp/cap-camera100-ffmpeg-bin/ffmpeg}
  [[ -s "$FFMPEG_SOURCE" ]] || die "bundled ffmpeg is missing: $FFMPEG_SOURCE"
  mkdir -p "$(dirname "$FFMPEG_TARGET")"
  if [[ ! -x "$FFMPEG_TARGET" || "$FFMPEG_TARGET" -ot "$FFMPEG_SOURCE" ]]; then
    cp "$FFMPEG_SOURCE" "$FFMPEG_TARGET.tmp.$$"
    chmod 755 "$FFMPEG_TARGET.tmp.$$"
    mv -f "$FFMPEG_TARGET.tmp.$$" "$FFMPEG_TARGET"
  fi
  timeout 20 "$FFMPEG_TARGET" -version >/dev/null 2>&1 || die "ffmpeg smoke test failed: $FFMPEG_TARGET"
  export IMAGEIO_FFMPEG_EXE="$FFMPEG_TARGET"
  export PATH="$(dirname "$FFMPEG_TARGET"):$PATH"
elif command -v ffmpeg >/dev/null 2>&1; then
  export IMAGEIO_FFMPEG_EXE="$(command -v ffmpeg)"
else
  die "set CAP_CAMERA_FFMPEG_SOURCE or install ffmpeg on PATH"
fi

mkdir -p "$OUTPUT_ROOT/logs"
LAUNCH_LOG="$OUTPUT_ROOT/logs/launcher.log"
exec > >(tee -a "$LAUNCH_LOG") 2>&1
printf 'launch_utc=%s checkpoint=%s output=%s\n' "$(date -u +%FT%TZ)" "$CHECKPOINT" "$OUTPUT_ROOT"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
BASE_COUNT=$((SAMPLE_COUNT / ${#GPUS[@]}))
REMAINDER=$((SAMPLE_COUNT % ${#GPUS[@]}))
CURRENT_START=$START_INDEX
PIDS=()
SHARD_IDS=()

terminate_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap 'terminate_children; exit 130' INT TERM

for shard_index in "${!GPUS[@]}"; do
  shard_count=$BASE_COUNT
  (( shard_index < REMAINDER )) && shard_count=$((shard_count + 1))
  gpu=${GPUS[$shard_index]}
  shard_id=$(printf '%02d' "$shard_index")
  log_path="$OUTPUT_ROOT/logs/shard${shard_id}_gpu${gpu}.log"
  skip_args=()
  [[ "$SKIP_EXISTING" == 1 ]] && skip_args+=(--skip_existing)
  dry_args=()
  [[ "$DRY_RUN" == 1 ]] && dry_args+=(--dry_run)
  printf '[%s] launch shard=%s gpu=%s start=%s count=%s\n' "$(date -u +%FT%TZ)" "$shard_id" "$gpu" "$CURRENT_START" "$shard_count"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u "$SCRIPT_PATH" \
    --checkpoint_path "$CHECKPOINT" \
    --model_name "$MODEL_NAME" \
    --config_path "$CONFIG_PATH" \
    --selected_csv "$SELECTED_CSV" \
    --camera_root "$CAMERA_ROOT" \
    --action_csv "$ACTION_CSV" \
    --output_dir "$OUTPUT_ROOT" \
    --device cuda:0 \
    --sample_count "$shard_count" \
    --start_index "$CURRENT_START" \
    --sample_height "$SAMPLE_HEIGHT" \
    --sample_width "$SAMPLE_WIDTH" \
    --start_image_center_crop_size "$START_IMAGE_CENTER_CROP_SIZE" \
    --video_length "$VIDEO_LENGTH" \
    --fps "$FPS" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --seed "$SEED" \
    --result_file_name "results_shard${shard_id}.json" \
    --camera_moe_root '' \
    "${skip_args[@]}" "${dry_args[@]}" \
    >"$log_path" 2>&1 &
  PIDS+=("$!")
  SHARD_IDS+=("$shard_id")
  CURRENT_START=$((CURRENT_START + shard_count))
  if (( shard_index + 1 < ${#GPUS[@]} )); then
    sleep "$WORKER_STAGGER_SECONDS"
  fi
done

STATUS=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    printf '[%s] shard=%s complete\n' "$(date -u +%FT%TZ)" "${SHARD_IDS[$index]}"
  else
    code=$?
    printf '[%s] shard=%s failed exit=%s\n' "$(date -u +%FT%TZ)" "${SHARD_IDS[$index]}" "$code" >&2
    STATUS=1
  fi
done
(( STATUS == 0 )) || exit "$STATUS"

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$CHECKPOINT" "$DRY_RUN" "$SELECTED_CSV" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
checkpoint = sys.argv[2]
dry_run = sys.argv[3] == "1"
selected_csv = sys.argv[4]
paths = sorted(root.glob("results_shard*.json"))
if len(paths) != 8:
    raise SystemExit(f"expected 8 shard results, found {len(paths)}")
rows = []
successes = []
failures = []
for path in paths:
    data = json.loads(path.read_text())
    if dry_run:
        rows.extend(item["row_index"] for item in data.get("cases", []))
    else:
        successes.extend(data.get("successes", []))
        failures.extend(data.get("failures", []))
        rows.extend(item["row_index"] for item in data.get("successes", []))
if sorted(rows) != list(range(100)):
    raise SystemExit(f"Camera-100 row coverage is invalid: count={len(rows)} unique={len(set(rows))}")
if failures:
    raise SystemExit(f"inference contains {len(failures)} failures")
if not dry_run:
    videos = [path for path in root.glob("*.mp4") if path.stat().st_size > 0]
    if len(videos) != 100 or len(successes) != 100:
        raise SystemExit(f"expected 100 videos/successes, found videos={len(videos)} successes={len(successes)}")
merged = {
    "dry_run": dry_run,
    "checkpoint_path": checkpoint,
    "selected_csv": selected_csv,
    "selected_count": len(rows),
    "num_success": 0 if dry_run else len(successes),
    "num_failures": len(failures),
    "row_indices": sorted(rows),
    "shard_results": [path.name for path in paths],
}
(root / "results.json").write_text(json.dumps(merged, indent=2) + "\n")
if not dry_run:
    (root / ".complete.json").write_text(json.dumps(merged, indent=2) + "\n")
print(f"validated Camera-100: dry_run={dry_run} rows={len(rows)}")
PY
