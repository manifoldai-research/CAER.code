#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def required_env(name):
    value = os.environ.get(name, "")
    if not value:
        fail(f"{name} is not set; edit config/paths.env")
    return value


def require_file(path, label):
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        fail(f"{label} is missing or empty: {path}")


def require_dir(path, label):
    if not os.path.isdir(path):
        fail(f"{label} directory is missing: {path}")


def first_json_record(path):
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with open(path, "r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            buffer += chunk
            if not started:
                stripped = buffer.lstrip()
                if not stripped or stripped[0] != "[":
                    fail(f"metadata must be a top-level JSON array: {path}")
                buffer = stripped[1:].lstrip()
                started = True
            try:
                record, _ = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                fail(f"first metadata item is not an object: {path}")
            return record
    fail(f"metadata contains no records: {path}")


def first_value(record, names):
    for name in names:
        value = record.get(name)
        if isinstance(value, dict):
            value = value.get("path") or value.get("file_path")
        if value not in (None, ""):
            return str(value)
    return None


def resolved(path, root):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(root or "/", path)


def validate_model(path, label, expected_channels):
    require_dir(path, label)
    config_path = os.path.join(path, "config.json")
    require_file(config_path, f"{label} config")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    channels = config.get("in_dim", config.get("in_channels"))
    if channels not in expected_channels:
        fail(f"{label} input channels must be one of {sorted(expected_channels)}, got {channels!r}")
    index_path = os.path.join(path, "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        shards = sorted(set(index.get("weight_map", {}).values()))
        if not shards:
            fail(f"model index has no weight shards: {index_path}")
        for shard in shards:
            require_file(os.path.join(path, shard), f"{label} weight shard")
    else:
        weights = glob.glob(os.path.join(path, "*.safetensors"))
        if not weights:
            fail(f"{label} has no safetensors weights: {path}")
        for weight in weights:
            require_file(weight, f"{label} weights")
    return channels


def validate_metadata(path, root, mode):
    require_file(path, f"{mode} metadata")
    record = first_json_record(path)
    media = first_value(record, ["file_path", "video_path", "image_path"])
    if not media:
        fail(f"first {mode} metadata item has no target media path")
    control_names = {
        "arm": ["ann_file", "annotation_path", "action_annotation_path"],
        "camera": ["control_file_path", "camera_pose_path", "condition_video_path"],
        "poseanything": ["control_file_path", "skeleton_video_path", "pose_video_path"],
        "libero": ["ann_file", "annotation_path"],
    }
    control = first_value(record, control_names[mode])
    if not control:
        fail(f"first {mode} metadata item has no control/annotation path")
    if mode == "camera" and not control.lower().endswith(".txt"):
        fail(f"camera control path must end in .txt, got: {control}")
    media_path = resolved(media, root)
    control_path = resolved(control, root)
    require_file(media_path, f"first {mode} target media")
    require_file(control_path, f"first {mode} control/annotation")
    return media_path, control_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["libero", "arm", "camera", "poseanything"])
    args = parser.parse_args()

    if args.mode == "libero":
        assets = required_env("LIBERO_ASSETS_DIR")
        require_dir(assets, "LIBERO assets")
        metadata = os.path.join(assets, "metadata_libero_single_arm_train.json")
        stat = os.path.join(assets, "stat.json")
        require_file(stat, "LIBERO action statistics")
        with open(stat, "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        if len(stats.get("state_01", [])) != 7 or len(stats.get("state_99", [])) != 7:
            fail("LIBERO stat.json must contain 7-D state_01 and state_99")
        model = os.path.join(assets, "ti2v_control_init_model")
        channels = validate_model(model, "LIBERO TI2V control model", {48, 100})
        media, control = validate_metadata(
            metadata, os.environ.get("LIBERO_TRAIN_DATA_ROOT", "/"), "libero"
        )
    elif args.mode in {"arm", "camera"}:
        channels = validate_model(required_env("CAP_CONTROL_MODEL"), "CAP control model", {48, 100})
        prefix = args.mode.upper()
        media, control = validate_metadata(
            required_env(f"{prefix}_METADATA"),
            os.environ.get(f"{prefix}_DATA_ROOT", "/"),
            args.mode,
        )
    else:
        channels = validate_model(required_env("POSE_BASE_MODEL"), "PoseAnything base model", {48})
        media, control = validate_metadata(
            required_env("POSE_METADATA"), os.environ.get("POSE_DATA_ROOT", "/"), "poseanything"
        )

    print(
        f"PREFLIGHT OK mode={args.mode} model_channels={channels} "
        f"media={media} control={control}"
    )


if __name__ == "__main__":
    main()

