#!/usr/bin/env python3
"""Stage a deterministic PoseAnything metadata prefix on node-local storage."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--copy-workers", type=int, default=8)
    return parser.parse_args()


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            if not started:
                buffer = buffer.lstrip()
                if not buffer:
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"metadata must be a top-level JSON array: {path}")
                buffer = buffer[1:]
                started = True

            while True:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                if buffer.startswith("]"):
                    return
                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(item, dict):
                    raise ValueError("every metadata entry must be an object")
                yield item
                buffer = buffer[end:]
    if started and buffer.strip() not in ("", "]"):
        raise ValueError(f"truncated JSON metadata: {path}")


def read_prefix(path: Path, count: int) -> list[dict[str, Any]]:
    records = []
    for record in iter_json_array(path):
        records.append(record)
        if len(records) == count:
            break
    if len(records) != count:
        raise ValueError(f"metadata has only {len(records)} entries; requested {count}")
    return records


def path_value(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[str, str]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            return key, value
    raise ValueError(f"metadata entry lacks all required path keys: {keys}")


def resolve_source(raw: str, data_root: Path, subdir: str) -> Path:
    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(data_root / path)
    candidates.append(data_root / subdir / path.name)
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate.resolve()
    raise FileNotFoundError(
        f"cannot resolve {subdir} source {raw!r}; tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def copy_one(source: Path, destination: Path) -> None:
    expected_size = source.stat().st_size
    if destination.is_file() and destination.stat().st_size == expected_size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(
                f"staged file size mismatch: source={source} destination={temporary}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.copy_workers <= 0:
        raise ValueError("--copy-workers must be positive")

    metadata = args.metadata.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not metadata.is_file():
        raise FileNotFoundError(f"metadata is missing: {metadata}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root is missing: {data_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = read_prefix(metadata, args.count)
    localized: list[dict[str, Any]] = []
    copies: list[tuple[Path, Path]] = []
    source_manifest = []
    for index, record in enumerate(records):
        video_key, video_raw = path_value(record, ("file_path", "video_path"))
        skeleton_key, skeleton_raw = path_value(
            record, ("skeleton_video_path", "control_file_path")
        )
        video_source = resolve_source(video_raw, data_root, "rgb")
        skeleton_source = resolve_source(skeleton_raw, data_root, "skeleton")
        case_dir = output_dir / "cases" / f"{index:05d}"
        video_destination = case_dir / f"rgb{video_source.suffix.lower()}"
        skeleton_destination = case_dir / f"skeleton{skeleton_source.suffix.lower()}"
        copies.extend(
            (
                (video_source, video_destination),
                (skeleton_source, skeleton_destination),
            )
        )
        local_record = dict(record)
        local_record[video_key] = str(video_destination)
        local_record[skeleton_key] = str(skeleton_destination)
        local_record["poseanything_metadata_index"] = index
        local_record["poseanything_original_video_path"] = str(video_source)
        local_record["poseanything_original_skeleton_path"] = str(skeleton_source)
        localized.append(local_record)
        source_manifest.append(
            {
                "metadata_index": index,
                "video": str(video_source),
                "video_bytes": video_source.stat().st_size,
                "skeleton": str(skeleton_source),
                "skeleton_bytes": skeleton_source.stat().st_size,
            }
        )

    print(
        f"staging {len(copies)} PoseAnything files with {args.copy_workers} workers",
        flush=True,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.copy_workers) as executor:
        futures = [executor.submit(copy_one, source, destination) for source, destination in copies]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            future.result()
            if completed % 10 == 0 or completed == len(futures):
                print(f"staged {completed}/{len(futures)} files", flush=True)

    for source, destination in copies:
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"staged file failed validation: {destination}")

    local_metadata = output_dir / "metadata_prefix.json"
    write_json_atomic(local_metadata, localized)
    write_json_atomic(
        output_dir / "stage_manifest.json",
        {
            "source_metadata": str(metadata),
            "source_metadata_bytes": metadata.stat().st_size,
            "data_root": str(data_root),
            "count": args.count,
            "files": source_manifest,
            "local_metadata": str(local_metadata),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(f"PoseAnything prefix ready: {local_metadata}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
