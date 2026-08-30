#!/usr/bin/env python
import argparse
import importlib.util
import json
from pathlib import Path


def load_presets_module():
    source = Path(__file__).resolve().parents[2] / "videox_fun/data/cap_dataset_presets.py"
    spec = importlib.util.spec_from_file_location("cap_dataset_presets", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_presets_module()
    parser = argparse.ArgumentParser(description="Read-only CAP metadata/path preflight.")
    parser.add_argument("--dataset_name", choices=module.CAP_DATASET_PRESETS, required=True)
    parser.add_argument(
        "--action_injection",
        choices=("arm", "action_map", "camera", "poseanything"),
        default=None,
    )
    parser.add_argument("--train_data_meta", default=None)
    parser.add_argument("--train_data_dir", default=None)
    args = parser.parse_args()
    preset = module.CAP_DATASET_PRESETS[args.dataset_name]
    action_injection = args.action_injection or preset["action_injection"]
    if action_injection != preset["action_injection"]:
        parser.error(
            f"--action_injection={action_injection} does not match "
            f"--dataset_name={args.dataset_name} ({preset['action_injection']})"
        )
    summary = module.preflight_cap_metadata(
        args.train_data_meta or preset["train_data_meta"],
        args.train_data_dir if args.train_data_dir is not None else preset["train_data_dir"],
        action_injection,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
