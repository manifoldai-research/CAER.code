#!/usr/bin/env python3
"""Merge Camera-100 heatmap shard results and validate case coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from visualize_cap_arm_weights import RENDERING_CONFIG, WEIGHT_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def rendering_config_for(output_format: str) -> dict:
    if output_format not in {"frames", "video"}:
        raise RuntimeError(f"unsupported heatmap output format: {output_format!r}")
    config = dict(RENDERING_CONFIG)
    if output_format == "video":
        config["output"] = "mp4"
    return config


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    result_paths = sorted(output_dir.glob("results_shard*.json"))
    if not result_paths:
        raise RuntimeError(f"no shard result files found in {output_dir}")

    successes = []
    failures = []
    shards = []
    for path in result_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        shards.append(
            {
                "result_file": path.name,
                "start_index": data.get("start_index"),
                "selected_count": data.get("selected_count"),
                "num_success": data.get("num_success"),
                "num_failures": data.get("num_failures"),
            }
        )
        for item in data.get("successes", []):
            value = dict(item)
            value["shard_result_file"] = path.name
            successes.append(value)
        for item in data.get("failures", []):
            value = dict(item)
            value["shard_result_file"] = path.name
            failures.append(value)

    successes.sort(key=lambda item: int(item["row_index"]))
    failures.sort(key=lambda item: int(item["row_index"]))
    success_ids = [int(item["row_index"]) for item in successes]
    if len(success_ids) != len(set(success_ids)):
        raise RuntimeError("duplicate successful case IDs across shards")
    expected_ids = list(range(args.expected_count))
    missing_ids = sorted(set(expected_ids).difference(success_ids))
    unexpected_ids = sorted(set(success_ids).difference(expected_ids))

    reports = []
    output_formats = set()
    for item in successes:
        generated_video = Path(item["output_path"])
        manifest_path = Path(item["heatmap_manifest"])
        if not generated_video.is_file() or generated_video.stat().st_size <= 0:
            raise RuntimeError(f"missing generated video: {generated_video}")
        if not manifest_path.is_file():
            raise RuntimeError(f"missing case manifest: {manifest_path}")
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
        if report.get("case_id") != int(item["row_index"]):
            raise RuntimeError(f"case manifest ID mismatch: {manifest_path}")
        if report.get("heatmap_background") != "ground_truth":
            raise RuntimeError(f"heatmap background is not GT: {manifest_path}")
        output_format = report.get("heatmap_output_format", "frames")
        output_formats.add(output_format)
        rendering = rendering_config_for(output_format)
        if any(report.get(key) != value for key, value in rendering.items()):
            raise RuntimeError(f"stale rendering config: {manifest_path}")
        for mode in WEIGHT_MODES:
            spec = report.get("weights", {}).get(mode, {})
            array_path = Path(spec.get("array", ""))
            if not array_path.is_file() or array_path.stat().st_size <= 0:
                raise RuntimeError(f"missing {mode} array: {manifest_path}")
            if output_format == "video":
                video_path = Path(spec.get("video", ""))
                if not video_path.is_file() or video_path.stat().st_size <= 0:
                    raise RuntimeError(f"missing {mode} video: {manifest_path}")
            else:
                pngs = [Path(path) for path in spec.get("pngs", [])]
                if not pngs or not all(path.is_file() and path.stat().st_size > 0 for path in pngs):
                    raise RuntimeError(f"missing {mode} PNGs: {manifest_path}")
        reports.append(report)

    if len(output_formats) > 1:
        raise RuntimeError(f"mixed heatmap output formats: {sorted(output_formats)}")
    heatmap_output_format = next(iter(output_formats), None)

    merged_result = {
        "checkpoint_path": str(checkpoint),
        "merged_shards": True,
        "num_shards": len(result_paths),
        "expected_count": args.expected_count,
        "selected_count": args.expected_count,
        "num_success": len(successes),
        "num_failures": len(failures),
        "heatmap_output_format": heatmap_output_format,
        "missing_case_ids": missing_ids,
        "unexpected_case_ids": unexpected_ids,
        "shards": shards,
        "successes": successes,
        "failures": failures,
        "rendering": (
            rendering_config_for(heatmap_output_format)
            if heatmap_output_format is not None
            else dict(RENDERING_CONFIG)
        ),
    }
    write_json(output_dir / "results.json", merged_result)
    write_json(
        output_dir / "manifest.json",
        {
            "checkpoint": str(checkpoint),
            "diagnostic_sigma": 0.5,
            "mse_used": False,
            "heatmap_background": "ground_truth",
            "heatmap_output_format": heatmap_output_format,
            "case_ids": [int(report["case_id"]) for report in reports],
            "cases": reports,
            "rendering": (
                rendering_config_for(heatmap_output_format)
                if heatmap_output_format is not None
                else dict(RENDERING_CONFIG)
            ),
        },
    )

    print(
        json.dumps(
            {
                "num_shards": len(result_paths),
                "num_success": len(successes),
                "num_failures": len(failures),
                "missing_case_ids": missing_ids,
                "unexpected_case_ids": unexpected_ids,
            },
            indent=2,
        ),
        flush=True,
    )
    if failures or missing_ids or unexpected_ids or len(successes) != args.expected_count:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
