#!/usr/bin/env python3
"""Infer a fixed metadata sample set with one reusable CAP pipeline per GPU."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import infer_cap_arm_sample as single


def parse_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample indices must be comma-separated integers") from exc
    if not indices:
        raise argparse.ArgumentTypeError("sample indices cannot be empty")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("sample indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("sample indices must be unique")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-indices", type=parse_indices, required=True)
    parser.add_argument("--metadata-prefix-limit", type=int, required=True)
    parser.add_argument("--selection-seed", type=int, required=True)
    parser.add_argument("--metadata", type=Path, default=single.DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=single.VARIANTS, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=single.DEFAULT_RUNS_ROOT)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--model-root", type=Path, default=single.DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=single.DEFAULT_CACHE_ROOT)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=single.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def sample_complete(
    output_dir: Path,
    metadata_index: int,
    checkpoint: Path,
    args: argparse.Namespace,
) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not all(
        nonempty_file(output_dir / name)
        for name in ("generated.mp4", "target_clip.mp4", "first_frame.png", "manifest.json")
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = {
        "metadata_index": metadata_index,
        "variant": args.variant,
        "checkpoint": str(checkpoint),
        "seed": args.generation_seed + metadata_index,
        "random_selection_seed": args.selection_seed,
        "generation_seed_base": args.generation_seed,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "fps": args.fps,
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def main() -> int:
    args = parse_args()
    if args.metadata_prefix_limit <= 0:
        raise ValueError("--metadata-prefix-limit must be positive")
    if any(index >= args.metadata_prefix_limit for index in args.sample_indices):
        raise ValueError(
            "every sample index must be below --metadata-prefix-limit: "
            f"limit={args.metadata_prefix_limit} indices={args.sample_indices}"
        )
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must be in [0, world-size)")
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    if not args.metadata.is_file():
        raise FileNotFoundError(f"metadata is missing: {args.metadata}")

    # validate_args expects a scalar sample_id; each item sets it again below.
    args.sample_id = args.sample_indices[0]
    args.seed = args.generation_seed + args.sample_id
    args.random_seed = args.selection_seed
    single.validate_args(args)
    run_dir, checkpoint, checkpoint_step = single.resolve_run_and_checkpoint(args)
    if checkpoint_step != args.checkpoint_step:
        raise RuntimeError(
            f"resolved checkpoint step mismatch: expected={args.checkpoint_step} actual={checkpoint_step}"
        )

    args.output_dir = args.output_dir.expanduser().resolve()
    args.status_dir = args.status_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    assigned = args.sample_indices[args.rank :: args.world_size]
    pending = [
        index
        for index in assigned
        if not sample_complete(
            args.output_dir / f"sample-{index:05d}", index, checkpoint, args
        )
    ]
    cache_dir = single.cache_directory(args, run_dir, checkpoint_step)

    pipeline = config = device = None
    if pending and not args.dry_run:
        pipeline, config, device = single.build_pipeline(args, checkpoint, cache_dir)

    write_json_atomic(
        args.status_dir / f"worker-{args.rank}-ready.json",
        {
            "rank": args.rank,
            "world_size": args.world_size,
            "assigned": list(assigned),
            "pending": pending,
            "pipeline_loaded": bool(pipeline is not None),
            "checkpoint": str(checkpoint),
            "ready_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(
        f"rank={args.rank}/{args.world_size} ready assigned={list(assigned)} pending={pending}",
        flush=True,
    )

    successes: list[int] = []
    skipped: list[int] = []
    failures: list[dict[str, Any]] = []
    for metadata_index in assigned:
        final_dir = args.output_dir / f"sample-{metadata_index:05d}"
        if sample_complete(final_dir, metadata_index, checkpoint, args):
            skipped.append(metadata_index)
            print(f"rank={args.rank} skip complete metadata_index={metadata_index}", flush=True)
            continue
        try:
            args.sample_id = metadata_index
            args.seed = args.generation_seed + metadata_index
            record = single.read_json_array_item(args.metadata, metadata_index)
            single.sample_paths(record)
            sample = single.extract_sample(record, args)
            if args.dry_run:
                successes.append(metadata_index)
                print(f"rank={args.rank} dry-run metadata_index={metadata_index}", flush=True)
                continue
            temporary_dir = args.output_dir / (
                f".sample-{metadata_index:05d}.tmp-r{args.rank}-{os.getpid()}"
            )
            if temporary_dir.exists():
                raise FileExistsError(f"temporary output already exists: {temporary_dir}")
            single.run_inference_with_pipeline(
                args,
                sample,
                checkpoint,
                cache_dir,
                temporary_dir,
                pipeline,
                config,
                device,
            )
            if final_dir.exists():
                raise FileExistsError(f"incomplete final output blocks atomic publish: {final_dir}")
            os.replace(temporary_dir, final_dir)
            successes.append(metadata_index)
            print(f"rank={args.rank} complete metadata_index={metadata_index}", flush=True)
        except Exception as exc:  # keep other assigned samples diagnosable
            traceback.print_exc()
            failures.append(
                {
                    "metadata_index": metadata_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    status = {
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned": list(assigned),
        "successes": successes,
        "skipped": skipped,
        "failures": failures,
        "checkpoint": str(checkpoint),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(args.status_dir / f"worker-{args.rank}-status.json", status)
    if failures:
        raise RuntimeError(f"rank {args.rank} failed {len(failures)} metadata samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
