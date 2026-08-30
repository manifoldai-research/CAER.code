import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from prepare_shared_ffn_ablation_assets import export_shared_ffn, prepare_ti2v_control_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets_dir", default=os.environ.get("LIBERO_ASSETS_DIR", "datasets/libero_single_arm_ablation"))
    parser.add_argument("--train_root", default=os.environ.get("LIBERO_TRAIN_ROOT", ""))
    parser.add_argument("--val_root", default=os.environ.get("LIBERO_VAL_ROOT", ""))
    parser.add_argument("--shared_moe_checkpoint", default=os.environ.get("SHARED_MOE_CHECKPOINT", ""))
    parser.add_argument("--ti2v_model_dir", default=os.environ.get("MODEL_NAME", ""))
    parser.add_argument("--control_template_model_dir", default=os.environ.get("CONTROL_TEMPLATE_MODEL_NAME", ""))
    parser.add_argument("--shared_export_mode", choices=["ffn_only", "full_single_ffn"], default="ffn_only")
    parser.add_argument("--window_size", type=int, default=17)
    parser.add_argument("--video_sample_stride", type=int, default=1)
    parser.add_argument("--metadata_stride", type=int, default=1)
    parser.add_argument("--skip_shared_export", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return value or "item"


def get_instruction(path, fallback):
    if not path.is_file():
        return fallback
    data = load_json(path)
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("instruction", "text", "task", "caption"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        for value in data.values():
            if isinstance(value, str) and value:
                return value
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("instruction", "text", "task", "caption"):
                value = first.get(key)
                if isinstance(value, str) and value:
                    return value
    return fallback


def derive_sibling_path(video_path, sibling_dir, suffix):
    video_path = Path(video_path)
    return video_path.parent.parent / sibling_dir / f"{video_path.stem}{suffix}"


def prepare_split(root, split, assets_dir, window_size, video_sample_stride, metadata_stride):
    root = Path(root)
    metadata_path = root / "metadata.json"
    items = load_json(metadata_path)
    if not isinstance(items, list):
        raise ValueError(f"Expected list metadata: {metadata_path}")

    annotation_dir = assets_dir / "annotation" / split
    annotation_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    train_actions = []
    skipped = []

    for item in items:
        video_path = Path(item["video_path"])
        action_path = derive_sibling_path(video_path, "actions", ".npy")
        if not action_path.is_file():
            action_path = derive_sibling_path(video_path, "states", ".npy")
        if not video_path.is_file() or not action_path.is_file():
            skipped.append({"video_path": str(video_path), "action_path": str(action_path)})
            continue

        action = np.load(action_path).astype(np.float32)
        if action.ndim != 2 or action.shape[1] != 7:
            raise ValueError(f"Expected action shape (T, 7), got {action.shape}: {action_path}")

        task = item.get("task", video_path.parents[3].name if len(video_path.parents) > 3 else "libero")
        episode = item.get("episode", video_path.stem.replace("episode", ""))
        episode_id = f"{safe_name(task)}__episode{episode}"
        instruction_path = derive_sibling_path(video_path, "instructions", ".json")
        text = get_instruction(instruction_path, task)
        ann_path = annotation_dir / f"{episode_id}.json"
        ann = {
            "episode_id": episode_id,
            "task_name": task,
            "episode": episode,
            "state": action.tolist(),
            "state_length": int(action.shape[0]),
            "texts": [text],
            "videos": [{"video_path": str(video_path)}],
            "file_path": str(video_path),
        }
        if split == "train":
            train_actions.append(action)
        dump_json_atomic(ann_path, ann)

        n_frames = int(min(item.get("n_frames", action.shape[0]), action.shape[0]))
        clip_span = (window_size - 1) * video_sample_stride + 1
        max_start = max(n_frames - clip_span, 0)
        for start_frame in range(0, max_start + 1, metadata_stride):
            metadata.append({
                "file_path": str(video_path),
                "ann_file": str(ann_path),
                "text": text,
                "type": "video",
                "task": task,
                "episode": f"episode{episode}",
                "episode_id": episode_id,
                "start_frame": int(start_frame),
                "window_size": int(window_size),
                "control_type": "arm",
                "arm_action_key": "state",
                "video_sample_stride": int(video_sample_stride),
                "video_sample_n_frames": int(window_size),
            })

    return metadata, train_actions, skipped


def prepare_libero_assets(args):
    assets_dir = Path(args.assets_dir)
    train_metadata_path = assets_dir / "metadata_libero_single_arm_train.json"
    val_metadata_path = assets_dir / "metadata_libero_single_arm_val.json"
    stat_path = assets_dir / "stat.json"

    if train_metadata_path.is_file() and val_metadata_path.is_file() and stat_path.is_file() and not args.overwrite:
        print(f"Libero metadata already exists: {train_metadata_path}")
        return train_metadata_path, val_metadata_path, stat_path

    train_metadata, train_actions, train_skipped = prepare_split(
        args.train_root,
        "train",
        assets_dir,
        args.window_size,
        args.video_sample_stride,
        args.metadata_stride,
    )
    val_metadata, _, val_skipped = prepare_split(
        args.val_root,
        "val",
        assets_dir,
        args.window_size,
        args.video_sample_stride,
        args.metadata_stride,
    )

    if not train_metadata:
        raise ValueError("No Libero train windows were generated")
    if not train_actions:
        raise ValueError("No Libero train actions were loaded")

    all_actions = np.concatenate(train_actions, axis=0)
    stat = {
        "state_01": np.percentile(all_actions, 1, axis=0).astype(float).tolist(),
        "state_99": np.percentile(all_actions, 99, axis=0).astype(float).tolist(),
        "state_mean": np.mean(all_actions, axis=0).astype(float).tolist(),
        "state_std": np.std(all_actions, axis=0).astype(float).tolist(),
        "action_dim": 7,
        "window_size": int(args.window_size),
        "video_sample_stride": int(args.video_sample_stride),
        "num_train_episodes": len(train_actions),
        "num_train_windows": len(train_metadata),
        "num_val_windows": len(val_metadata),
        "num_train_skipped": len(train_skipped),
        "num_val_skipped": len(val_skipped),
    }

    dump_json_atomic(train_metadata_path, train_metadata)
    dump_json_atomic(val_metadata_path, val_metadata)
    dump_json_atomic(stat_path, stat)
    print(f"Wrote Libero train metadata: {train_metadata_path} ({len(train_metadata)} windows)")
    print(f"Wrote Libero val metadata: {val_metadata_path} ({len(val_metadata)} windows)")
    print(f"Wrote Libero action stats: {stat_path}")
    if train_skipped or val_skipped:
        skipped_path = assets_dir / "skipped_libero_items.json"
        dump_json_atomic(skipped_path, {"train": train_skipped, "val": val_skipped})
        print(f"Wrote skipped item report: {skipped_path}")
    return train_metadata_path, val_metadata_path, stat_path


def write_manifest(args, train_metadata_path, val_metadata_path, stat_path, model_dir, shared_path):
    manifest = {
        "assets_dir": str(Path(args.assets_dir)),
        "train_metadata": str(train_metadata_path),
        "val_metadata": str(val_metadata_path),
        "action_stat_path": str(stat_path),
        "ti2v_control_init_model": str(model_dir),
        "shared_ffn_checkpoint": str(shared_path) if shared_path is not None else None,
        "shared_export_mode": args.shared_export_mode,
        "shared_moe_checkpoint": args.shared_moe_checkpoint,
        "ti2v_model_dir": args.ti2v_model_dir,
        "control_template_model_dir": args.control_template_model_dir,
        "train_root": args.train_root,
        "val_root": args.val_root,
        "window_size": args.window_size,
        "video_sample_stride": args.video_sample_stride,
        "metadata_stride": args.metadata_stride,
    }
    path = Path(args.assets_dir) / "assets_manifest.json"
    dump_json_atomic(path, manifest)
    print(f"Wrote manifest: {path}")


def main():
    args = parse_args()
    Path(args.assets_dir).mkdir(parents=True, exist_ok=True)
    train_metadata_path, val_metadata_path, stat_path = prepare_libero_assets(args)
    model_dir = prepare_ti2v_control_model(args)
    shared_path = None if args.skip_shared_export else export_shared_ffn(args)
    write_manifest(args, train_metadata_path, val_metadata_path, stat_path, model_dir, shared_path)


if __name__ == "__main__":
    main()
