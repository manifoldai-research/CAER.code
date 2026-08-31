# Training Guide

This page is the operational contract for reproducing the four CAER modalities. The paper overview lives in [README.md](../README.md).

All launch commands resolve paths from `config/paths.env`. They do not depend on any internal cluster filesystem.

## 1. Layout

```text
.
├── config/paths.env.example
├── scripts/
│   ├── preflight.py / preflight.sh / runtime_preflight.py
│   ├── train.sh
│   ├── train_libero.sh
│   ├── train_arm.sh
│   ├── train_camera.sh
│   └── train_poseanything.sh
└── sources/
    ├── libero/    # validated LIBERO trainer
    └── cap/       # Arm / Camera / PoseAnything trainer
```

LIBERO keeps a separate source snapshot. Merging it with the CAP tree would change model construction, trainable modules, and the checkpoint contract.

## 2. Environment

Linux, Python 3.10 or 3.11, a CUDA build of PyTorch, and eight GPUs on one node for formal runs. Formal jobs use bf16 and FSDP.

```bash
python -m venv /path/to/cap-env
/path/to/cap-env/bin/pip install -U pip
/path/to/cap-env/bin/pip install -r sources/cap/requirements.txt

python -m venv /path/to/libero-env
/path/to/libero-env/bin/pip install -U pip
/path/to/libero-env/bin/pip install -r sources/libero/requirements.txt
```

Install PyTorch for the CUDA version on the machine. FFmpeg must be able to decode MP4/AVI.

## 3. Models and Data

Every metadata file must be a top-level JSON array. Absolute paths inside metadata must exist on the new machine. A `*_DATA_ROOT` value only resolves **relative** paths; it does not rewrite absolute paths.

### 3.1 LIBERO

`LIBERO_ASSETS_DIR` must contain:

```text
libero_single_arm_ablation/
├── metadata_libero_single_arm_train.json
├── stat.json
├── annotation/train/*.json
└── ti2v_control_init_model/
    ├── config.json
    ├── diffusion_pytorch_model.safetensors.index.json   # if sharded
    └── *.safetensors
```

Each record needs at least:

```json
{
  "file_path": "/path/to/episode.mp4",
  "ann_file": "/path/to/annotation.json",
  "text": "task instruction",
  "type": "video",
  "start_frame": 0,
  "window_size": 17,
  "control_type": "arm",
  "arm_action_key": "state"
}
```

`stat.json` must provide 7-D `state_01` and `state_99`. The paper contract is 400 training episodes and 44,166 windows of 17 frames. `ti2v_control_init_model` is a prepared TI2V control directory, not a single training checkpoint file.

### 3.2 Arm (RoboTwin)

- `CAP_CONTROL_MODEL`: Wan2.2 TI2V control directory (48-channel base expanded at startup, or a 100-channel control init).
- `ARM_METADATA`: `file_path`, `ann_file`, `control_type=arm`.
- Referenced RGB videos and 14-D state/action annotations.

Leave `ARM_ACTION_STAT_PATH` empty unless the dataset requires percentile normalization.

### 3.3 Camera

Same `CAP_CONTROL_MODEL`, plus:

- `CAMERA_METADATA`: target video and an aligned camera-pose `.txt`.
- Every referenced RGB video and pose file.

The pose file must cover the selected frames. Preflight checks the first media/control pair and the `.txt` suffix.

### 3.4 PoseAnything

- `POSE_BASE_MODEL`: original 48-channel Wan2.2-TI2V-5B directory.
- `POSE_METADATA`: XPose-converted metadata.
- Referenced RGB videos and aligned skeleton videos.

A record stores 81 aligned candidate indices; the formal forward pass takes 17 evenly spaced frames. At startup the patch input expands from 48 to 96 channels; the extra 48 skeleton channels are zero-initialized.

## 4. Configure and Preflight

```bash
cp config/paths.env.example config/paths.env
```

Replace every `/path/to/...` value. This is the only file that must be edited. Or point `REPRO_CONFIG` at another env file.

```bash
bash scripts/preflight.sh all
bash scripts/preflight.sh libero
bash scripts/preflight.sh arm
bash scripts/preflight.sh camera
bash scripts/preflight.sh poseanything
```

Preflight checks model shards, input channels, the first metadata record, the first media/control files, and LIBERO 7-D statistics. It does not walk hundreds of thousands of media paths.

## 5. Launch

Run from the repository root. Submit each formal command as one 8-GPU job.

```bash
bash scripts/train_libero.sh CAER          # paper CAER setting
bash scripts/train_libero.sh MSE           # matched MSE baseline
bash scripts/train_arm.sh CAER
bash scripts/train_camera.sh CAER
bash scripts/train_poseanything.sh CAER
bash scripts/train.sh camera CAER         # equivalent unified entry
bash scripts/train_arm.sh CAER --max_train_steps 2000
```

Accepted loss variants: `MSE`, `CAER`. LIBERO accepts only `MSE` or `CAER`.

## 6. Dry-run, Smoke, Resume

```bash
DRY_RUN=1 bash scripts/train_camera.sh CAER
DRY_RUN=1 CAP_SKIP_RUNTIME_PREFLIGHT=1 bash scripts/train_camera.sh CAER
SMOKE=1 bash scripts/train_poseanything.sh CAER

RESUME_FROM_CHECKPOINT=/path/to/run/checkpoint-5000 \
bash scripts/train_camera.sh CAER
```

`CAP_SKIP_RUNTIME_PREFLIGHT=1` is rejected unless `DRY_RUN=1`. `--max_train_steps` is the absolute final optimizer step. Resume with the same modality, loss, topology, batch, optimizer, and data contract.

## 7. Formal Hyperparameters

| Modality | Model input | Video | Global batch | LR | Default length | Trainable scope |
|---|---|---|---:|---:|---:|---|
| LIBERO | TI2V control + 7-D arm | 17 frames | 32 | `2e-5` | 5000 steps | FFN + arm action branch |
| Arm | 100ch CAP + 14-D arm | 17 frames, `1280x704` | 32 | `2e-5` | 5 epochs | Full Transformer, MoE off |
| Camera | 100ch CAP + camera rays | 17 frames, `1280x704` | 32 | `2e-5` | 5 epochs | Full Transformer, MoE off |
| PoseAnything | 96ch RGB/skeleton | 17 frames, `1280x704` | 32 | `2e-5` | 5 epochs | Full Transformer, MoE off |

CAP global batch is `8 GPUs × 1 × grad accum 4 = 32`. Seed 42, bf16, checkpoint every 1000 steps. CAER defaults: `dropout=0.10`, `tau=0.50`, `eps=1e-6`.

## 8. Outputs

If `OUTPUT_DIR` is unset:

```text
${OUTPUT_ROOT}/<libero|arm|camera|poseanything>/<loss>/<UTC run id>/
```

CAP writes `checkpoint-N/`, TensorBoard, and training logs. LIBERO writes `logs/` and `tensorboard/`. A new LIBERO run refuses a non-empty output directory; resume through `RESUME_FROM_CHECKPOINT`.

## 9. Source Mapping

| Path | Role |
|---|---|
| `sources/cap/scripts/wan2.2_fun/train_control_camera_arm_actionmap_method1.py` | Arm / Camera / PoseAnything trainer |
| `sources/cap/videox_fun/training/method1_focused_loss.py` | CAER focused loss |
| `sources/cap/scripts/wan2.2_fun/run_cap_train.sh` | CAP FSDP launcher |
| `sources/libero/scripts/wan2.2_fun/train_control_camera_arm_actionmap.py` | LIBERO trainer |
| `sources/libero/scripts/wan2.2_fun/run_libero_ti2v_loss_volc.sh` | LIBERO launcher |

Weights, datasets, run outputs, and virtualenvs are not part of this repository.
