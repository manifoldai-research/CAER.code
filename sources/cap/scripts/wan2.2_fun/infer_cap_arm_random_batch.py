#!/usr/bin/env python3
"""Prepare and run a reproducible random Arm CAP inference batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import infer_cap_arm_sample as single


DEFAULT_METADATA_SIZE = 365831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select fixed random sample IDs and infer one deterministic GPU shard."
    )
    parser.add_argument("--batch-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--metadata-size", type=int, default=DEFAULT_METADATA_SIZE)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--variant", choices=single.VARIANTS, default="current")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=single.DEFAULT_RUNS_ROOT)
    parser.add_argument("--metadata", type=Path, default=single.DEFAULT_METADATA)
    parser.add_argument("--model-root", type=Path, default=single.DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=single.DEFAULT_CACHE_ROOT)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--negative-prompt", default=single.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_count <= 0 or args.sample_count > args.metadata_size:
        raise SystemExit("--sample-count must be in [1, metadata-size]")
    if args.metadata_size <= 0:
        raise SystemExit("--metadata-size must be > 0")
    if args.world_size <= 0:
        raise SystemExit("--world-size must be > 0")
    if not args.prepare_only and (args.rank is None or not 0 <= args.rank < args.world_size):
        raise SystemExit("worker mode requires --rank in [0, world-size)")
    validation_args = SimpleNamespace(
        sample_id=0,
        checkpoint_step=args.checkpoint_step,
        height=args.height,
        width=args.width,
        frames=args.frames,
        fps=args.fps,
        inference_steps=args.inference_steps,
    )
    single.validate_args(validation_args)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def selection_checksum(sample_ids: list[int]) -> str:
    encoded = json.dumps(sample_ids, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def metadata_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def select_sample_ids(seed: int, population_size: int, sample_count: int) -> list[int]:
    """Stable, version-independent selection without replacement."""
    selected: list[int] = []
    seen: set[int] = set()
    counter = 0
    seed_bytes = str(seed).encode("ascii")
    while len(selected) < sample_count:
        digest = hashlib.sha256(seed_bytes + b":" + counter.to_bytes(8, "big")).digest()
        counter += 1
        candidate = int.from_bytes(digest[:8], "big") % population_size
        if candidate not in seen:
            seen.add(candidate)
            selected.append(candidate)
    return selected


def write_selected_records(
    metadata: Path,
    records_path: Path,
    sample_ids: list[int],
    expected_size: int,
) -> None:
    with metadata.open("r", encoding="utf-8") as handle:
        all_records = json.load(handle)
    if not isinstance(all_records, list):
        raise ValueError(f"metadata must be a top-level JSON array: {metadata}")
    if len(all_records) != expected_size:
        raise ValueError(
            f"--metadata-size is {expected_size}, but metadata contains {len(all_records)} items"
        )
    missing = [sample_id for sample_id in sample_ids if sample_id >= len(all_records)]
    if missing:
        raise IndexError(f"selected sample IDs exceed metadata length {len(all_records)}: {missing[:10]}")

    tmp = records_path.with_name(f".{records_path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(
                json.dumps(
                    {"sample_id": sample_id, "record": all_records[sample_id]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    os.replace(tmp, records_path)


def load_records(path: Path, expected_ids: list[int]) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                sample_id = int(value["sample_id"])
                record = value["record"]
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid selected record at {path}:{line_number}") from exc
            if sample_id in records or not isinstance(record, dict):
                raise ValueError(f"duplicate/invalid sample_id {sample_id} in {path}")
            records[sample_id] = record
    if set(records) != set(expected_ids):
        raise ValueError(f"{path} does not match selected_ids.json")
    return records


def prepare_batch(args: argparse.Namespace) -> dict[str, Any]:
    args.batch_dir = args.batch_dir.expanduser().resolve()
    manifest_path = args.batch_dir / "selected_ids.json"
    records_path = args.batch_dir / "selected_records.jsonl"
    identity = metadata_identity(args.metadata)
    inference_settings = {
        "generation_seed_base": args.generation_seed,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "fps": args.fps,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "negative_prompt": args.negative_prompt,
        "prompt_override": args.prompt,
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "sample_count": args.sample_count,
            "random_seed": args.random_seed,
            "metadata_size": args.metadata_size,
            "variant": args.variant,
            "selection_algorithm": "sha256_counter_v1_without_replacement",
            "inference": inference_settings,
        }
        mismatches = {
            key: (manifest.get(key), value)
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if manifest.get("metadata") != identity:
            mismatches["metadata"] = (manifest.get("metadata"), identity)
        if mismatches:
            raise ValueError(f"existing batch manifest conflicts with requested settings: {mismatches}")
        sample_ids = manifest.get("selected_ids", [])
        if (
            len(sample_ids) != args.sample_count
            or len(set(sample_ids)) != len(sample_ids)
            or any(not isinstance(value, int) or not 0 <= value < args.metadata_size for value in sample_ids)
            or manifest.get("selection_sha256") != selection_checksum(sample_ids)
        ):
            raise ValueError(f"invalid existing selection manifest: {manifest_path}")
        stored_run = Path(manifest["run_dir"])
        stored_step = int(manifest["checkpoint_step"])
        if args.run_dir is not None and args.run_dir.expanduser().resolve() != stored_run:
            raise ValueError("--run-dir conflicts with the existing batch manifest")
        if args.checkpoint_step is not None and args.checkpoint_step != stored_step:
            raise ValueError("--checkpoint-step conflicts with the existing batch manifest")
        args.run_dir = stored_run
        args.checkpoint_step = stored_step
        _, checkpoint, _ = single.resolve_run_and_checkpoint(args)
        if str(checkpoint) != manifest["checkpoint"]:
            raise ValueError("checkpoint resolved differently from the existing batch manifest")
        print(f"Reusing fixed selection manifest: {manifest_path}", flush=True)
    else:
        args.batch_dir.mkdir(parents=True, exist_ok=True)
        sample_ids = select_sample_ids(
            args.random_seed, args.metadata_size, args.sample_count
        )
        run_dir, checkpoint, checkpoint_step = single.resolve_run_and_checkpoint(args)
        manifest = {
            "format_version": 1,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sample_count": args.sample_count,
            "random_seed": args.random_seed,
            "metadata_size": args.metadata_size,
            "metadata": identity,
            "selection_algorithm": "sha256_counter_v1_without_replacement",
            "selection_sha256": selection_checksum(sample_ids),
            "selected_ids": sample_ids,
            "variant": args.variant,
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_step": checkpoint_step,
            "world_size": args.world_size,
            "inference": inference_settings,
            "generation_seed_formula": "generation_seed_base + sample_id",
        }
        atomic_write_json(manifest_path, manifest)
        print(f"Created fixed selection manifest: {manifest_path}", flush=True)

    if records_path.exists():
        load_records(records_path, sample_ids)
        print(f"Reusing selected metadata records: {records_path}", flush=True)
    else:
        print(f"Resolving {len(sample_ids)} selected records in one metadata pass...", flush=True)
        write_selected_records(args.metadata, records_path, sample_ids, args.metadata_size)
        print(f"Created selected metadata records: {records_path}", flush=True)

    counts = [len(sample_ids[rank :: args.world_size]) for rank in range(args.world_size)]
    print(
        json.dumps(
            {
                "batch_dir": str(args.batch_dir),
                "selection_sha256": manifest["selection_sha256"],
                "sample_count": len(sample_ids),
                "unique_count": len(set(sample_ids)),
                "random_seed": args.random_seed,
                "first_20_ids": sample_ids[:20],
                "min_id": min(sample_ids),
                "max_id": max(sample_ids),
                "samples_per_rank": counts,
                "checkpoint": manifest["checkpoint"],
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def output_complete(
    output_dir: Path,
    sample_id: int,
    checkpoint: str,
    inference: dict[str, Any],
    random_seed: int,
) -> bool:
    required = ("generated.mp4", "target_clip.mp4", "first_frame.png", "manifest.json")
    if not all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in required):
        return False
    try:
        value = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        value.get("sample_id") == sample_id
        and value.get("checkpoint") == checkpoint
        and value.get("seed") == inference["generation_seed_base"] + sample_id
        and value.get("random_selection_seed") == random_seed
        and value.get("generation_seed_base") == inference["generation_seed_base"]
        and value.get("height") == inference["height"]
        and value.get("width") == inference["width"]
        and value.get("frames") == inference["frames"]
        and value.get("fps") == inference["fps"]
        and value.get("inference_steps") == inference["inference_steps"]
        and value.get("guidance_scale") == inference["guidance_scale"]
        and value.get("negative_prompt") == inference["negative_prompt"]
    )


def run_worker(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    selected_ids = manifest["selected_ids"]
    worker_ids = selected_ids[args.rank :: args.world_size]
    records = load_records(args.batch_dir / "selected_records.jsonl", selected_ids)
    args.run_dir = Path(manifest["run_dir"])
    args.checkpoint_step = int(manifest["checkpoint_step"])
    run_dir, checkpoint, checkpoint_step = single.resolve_run_and_checkpoint(args)
    if str(checkpoint) != manifest["checkpoint"]:
        raise ValueError("worker checkpoint differs from selected_ids.json")
    cache_dir = single.cache_directory(args, run_dir, checkpoint_step)
    samples_root = args.batch_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    pending = [
        sample_id
        for sample_id in worker_ids
        if not output_complete(
            samples_root / f"sample-{sample_id:06d}",
            sample_id,
            str(checkpoint),
            manifest["inference"],
            manifest["random_seed"],
        )
    ]
    print(
        f"rank={args.rank}/{args.world_size} assigned={len(worker_ids)} "
        f"complete={len(worker_ids) - len(pending)} pending={len(pending)}",
        flush=True,
    )
    if not pending:
        return

    pipeline, config, device = single.build_pipeline(args, checkpoint, cache_dir)
    failures: list[dict[str, Any]] = []
    for position, sample_id in enumerate(pending, 1):
        output_dir = samples_root / f"sample-{sample_id:06d}"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        sample_args = SimpleNamespace(**vars(args))
        sample_args.sample_id = sample_id
        sample_args.seed = args.generation_seed + sample_id
        started = time.monotonic()
        print(
            f"rank={args.rank} sample={position}/{len(pending)} sample_id={sample_id} "
            f"generation_seed={sample_args.seed}",
            flush=True,
        )
        try:
            sample = single.extract_sample(records[sample_id], sample_args)
        except Exception as exc:
            failures.append(
                {"sample_id": sample_id, "stage": "sample_extraction", "error": repr(exc)}
            )
            print(
                f"rank={args.rank} FAILED sample_id={sample_id} "
                f"stage=sample_extraction: {exc!r}",
                flush=True,
            )
            traceback.print_exc()
            shutil.rmtree(output_dir, ignore_errors=True)
            continue

        try:
            path = single.run_inference_with_pipeline(
                sample_args,
                sample,
                checkpoint,
                cache_dir,
                output_dir,
                pipeline,
                config,
                device,
            )
        except Exception as exc:
            failure = {
                "sample_id": sample_id,
                "stage": "pipeline_inference",
                "error": repr(exc),
            }
            failures.append(failure)
            print(
                f"rank={args.rank} FATAL sample_id={sample_id} "
                f"stage=pipeline_inference: {exc!r}",
                flush=True,
            )
            traceback.print_exc()
            shutil.rmtree(output_dir, ignore_errors=True)
            status = {
                "rank": args.rank,
                "world_size": args.world_size,
                "assigned": len(worker_ids),
                "failures": failures,
                "fatal": failure,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            atomic_write_json(
                args.batch_dir / "logs" / f"worker-{args.rank}-status.json", status
            )
            raise

        print(
            f"rank={args.rank} completed sample_id={sample_id} "
            f"elapsed_s={time.monotonic() - started:.1f} output={path}",
            flush=True,
        )

    status = {
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned": len(worker_ids),
        "failures": failures,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_write_json(args.batch_dir / "logs" / f"worker-{args.rank}-status.json", status)
    if failures:
        raise RuntimeError(f"rank {args.rank} had {len(failures)} failed samples")


def main() -> int:
    args = parse_args()
    validate_args(args)
    manifest = prepare_batch(args)
    if args.prepare_only or args.dry_run:
        print("Batch preflight passed: no video was decoded and no GPU/model was used.")
        return 0
    run_worker(args, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
