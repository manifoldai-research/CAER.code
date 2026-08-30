#!/usr/bin/env python3
"""Validate the original Camera-100 inputs without importing ML packages."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    return parser.parse_args()


def code(value: str, name: str, row_index: int) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"row {row_index}: empty {name}")
    return int(float(text))


def camera_path(row: dict[str, str], root: Path, row_index: int) -> Path:
    direct = str(row.get("control_camera_txt", "")).strip()
    if direct:
        path = Path(direct)
        if not path.is_file():
            raise FileNotFoundError(f"row {row_index}: camera file is missing: {path}")
        return path
    level = code(row.get("level", ""), "level", row_index)
    translation = code(row.get("translation_code", ""), "translation_code", row_index)
    rotation = code(row.get("rotation_code", ""), "rotation_code", row_index)
    exact = root / f"camera_{level}_{translation}_{rotation}.txt"
    if exact.is_file():
        return exact
    candidates = sorted(root.glob(f"camera_*_{translation}_{rotation}.txt"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"row {row_index}: expected one Camera trajectory for "
        f"camera_{level}_{translation}_{rotation}, found {len(candidates)}"
    )


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z._-]+", "_", str(value)).strip("_")
    return cleaned or "camera_case"


def main() -> None:
    args = parse_args()
    if not args.selected_csv.is_file():
        raise FileNotFoundError(args.selected_csv)
    if not args.camera_root.is_dir():
        raise FileNotFoundError(args.camera_root)
    names: set[str] = set()
    with args.selected_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise RuntimeError(f"Camera-100 CSV must contain exactly 100 rows, found {len(rows)}")
    for row_index, row in enumerate(rows):
        source = Path(str(row.get("source_absolute_path", "")).strip())
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"row {row_index}: source video is missing or empty: {source}")
        trajectory = camera_path(row, args.camera_root, row_index)
        if trajectory.stat().st_size == 0:
            raise RuntimeError(f"row {row_index}: camera trajectory is empty: {trajectory}")
        stem = str(row.get("case_name", "")).strip() or Path(
            row.get("source_filename") or source
        ).stem
        output_name = f"{row_index:03d}_{sanitize(stem)}_{trajectory.stem}.mp4"
        if output_name in names:
            raise RuntimeError(f"duplicate Camera-100 output name: {output_name}")
        names.add(output_name)
    print(
        f"Camera-100 preflight passed: rows={len(rows)} sources=100 trajectories=100 outputs=100"
    )


if __name__ == "__main__":
    main()
