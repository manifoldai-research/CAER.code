#!/usr/bin/env python3
"""Export full-frame S/E heatmap videos for GT cases.

Existing cases are rendered from their saved NPZ weights. Cases that are not
present in the previous random-10 directory use the same fixed-sigma
diagnostic forward as visualize_cap_gt_weights.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

import arm_mse_heatmap as weight_viz
import infer_cap_arm_sample as single
from visualize_cap_arm_weights import RENDERING_CONFIG, WEIGHT_MODES
from visualize_cap_gt_weights import (
    _architecture_kwargs,
    build_pipeline,
    diagnose_case,
    extract_case,
    load_transformer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("poseanything", "libero"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("POSE_BASE_MODEL", "models/Wan2.2-TI2V-5B")))
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("MODEL_CACHE_ROOT", "outputs/model-cache")))
    parser.add_argument("--transformer-cache", type=Path)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260723)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--diagnostic-sigma", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_records(path: Path, count: int, seed: int):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or count <= 0 or count > len(records):
        raise ValueError(f"invalid metadata/sample count: records={len(records)} count={count}")
    ids = random.Random(seed).sample(range(len(records)), count)
    return ids, [records[index] for index in ids]


def _case_dir(root: Path, sample_id: int, case_name: str) -> Path:
    return root / f"case_{sample_id:06d}_{case_name}"


def _source_indices(case_manifest: dict[str, Any]) -> list[int]:
    return [int(value) for value in case_manifest["frame_indices"]]


def _read_frames(path: Path, indices: list[int], height: int, width: int) -> np.ndarray:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    frames = reader.get_batch(np.asarray(indices, dtype=np.int64)).asnumpy()
    return np.stack(
        [np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)) for frame in frames]
    )


def _latent_from_raw(raw: np.ndarray) -> np.ndarray:
    """Recover the 16x44x80 latent map from a 17-frame rendered rho map."""

    chunks, error = weight_viz.recover_latent_chunks_from_interpolated_weights(
        raw,
        latent_shape=(16, 44, 80),
        future_frames_per_chunk=16,
    )
    if error > 1e-3:
        raise RuntimeError(f"saved rho interpolation recovery error is too large: {error}")
    return chunks[0]


def _display_response(raw: np.ndarray, normalization: dict[str, Any]) -> np.ndarray:
    latent = _latent_from_raw(raw)
    display = weight_viz.render_map_to_video(
        weight_viz.smooth_latent_spatially(latent, sigma=1.5),
        raw.shape[0],
        output_size=raw.shape[1:],
    )
    vmax = float(normalization["vmax"])
    vmin = float(normalization["vmin"])
    return weight_viz.normalize_weight_response(display, vmax, vmin=vmin)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=float(fps),
        codec="libx264",
        format="FFMPEG",
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


def _render_case(
    args: argparse.Namespace,
    output_case: Path,
    source_manifest: dict[str, Any],
    rho: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_case.mkdir(parents=True, exist_ok=True)
    if rho is None:
        rho = {}
        for mode in WEIGHT_MODES:
            spec = source_manifest["weights"][mode]
            rho[mode] = np.load(spec["array"])["weights"]
    raw_by_mode = {}
    response_by_mode = {}
    for mode in WEIGHT_MODES:
        values = rho[mode]
        if "weights" in source_manifest:
            raw = np.asarray(values, dtype=np.float32)
            normalization = source_manifest["weights"][mode]["normalization"]
            response = _display_response(raw, normalization)
        else:
            latent = values[0, 0].detach().cpu().numpy().astype(np.float32)
            raw = weight_viz.render_map_to_video(
                values,
                args.frames,
                output_size=(args.height, args.width),
            )
            vmax = weight_viz.positive_percentile_vmax(latent, percentile=99.0, exclude_first_frame=False)
            display = weight_viz.render_map_to_video(
                weight_viz.smooth_latent_spatially(latent, sigma=1.5),
                args.frames,
                output_size=(args.height, args.width),
            )
            normalization = {
                "vmax": vmax,
                "vmin": weight_viz.episode_response_vmin(display, vmax),
                "percentile": 99.0,
            }
            response = weight_viz.normalize_weight_response(display, normalization["vmax"], vmin=normalization["vmin"])
        raw_by_mode[mode] = raw
        response_by_mode[mode] = response

    rgb = _read_frames(
        Path(source_manifest["video_path"]),
        _source_indices(source_manifest),
        args.height,
        args.width,
    )
    videos = {}
    for mode, name in (("CAER", "CAER"), ("MSE", "MSE")):
        frames = [
            weight_viz.overlay_weight_response(rgb[index], response_by_mode[mode][index], blur_radius=12.0)
            for index in range(args.frames)
        ]
        video_path = output_case / f"{name}.mp4"
        _write_video(video_path, frames, args.fps)
        videos[mode] = str(video_path)
    report = {
        "sample_id": source_manifest["sample_id"],
        "case_name": source_manifest["case_name"],
        "video_path": source_manifest["video_path"],
        "condition_path": source_manifest.get("condition_path"),
        "frame_indices": source_manifest["frame_indices"],
        "background": "ground_truth",
        "video_fps": args.fps,
        "video_frame_count": args.frames,
        "rendering": dict(RENDERING_CONFIG),
        "videos": videos,
    }
    (output_case / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _load_existing_case(existing_dir: Path, sample_id: int, case_name: str) -> dict[str, Any] | None:
    path = _case_dir(existing_dir, sample_id, case_name) / "manifest.json"
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.existing_dir = args.existing_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids, records = load_records(args.metadata.resolve(), args.sample_count, args.sample_seed)
    (args.output_dir / "selection.json").write_text(json.dumps({"sample_seed": args.sample_seed, "sample_ids": ids}, indent=2) + "\n")

    existing_reports = {}
    for sample_id, record in zip(ids, records):
        case_name = str(record.get("episode_id") or record.get("source", {}).get("sample_name") or Path(record.get("file_path") or record["video_path"]).stem)
        existing = _load_existing_case(args.existing_dir, sample_id, case_name)
        if existing is not None:
            existing_reports[(sample_id, case_name)] = existing

    missing = [(sid, rec) for sid, rec in zip(ids, records) if (sid, str(rec.get("episode_id") or rec.get("source", {}).get("sample_name") or Path(rec.get("file_path") or rec["video_path"]).stem)) not in existing_reports]
    pipeline = None
    if missing:
        transformer = load_transformer(args)
        pipeline = build_pipeline(args, transformer)

    reports = []
    for position, (sample_id, record) in enumerate(zip(ids, records), 1):
        case_name = str(record.get("episode_id") or record.get("source", {}).get("sample_name") or Path(record.get("file_path") or record["video_path"]).stem)
        existing = existing_reports.get((sample_id, case_name))
        output_case = _case_dir(args.output_dir, sample_id, case_name)
        cached_manifest = output_case / "manifest.json"
        if cached_manifest.is_file():
            try:
                cached = json.loads(cached_manifest.read_text())
                if all(Path(path).is_file() for path in cached.get("videos", {}).values()):
                    reports.append(cached)
                    print(f"reused {position}/{len(ids)} sample_id={sample_id}", flush=True)
                    continue
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if existing is not None:
            report = _render_case(args, output_case, existing)
            print(f"rendered existing {position}/{len(ids)} sample_id={sample_id}", flush=True)
        else:
            case = extract_case(record, args)
            rho = diagnose_case(args, pipeline, case, sample_id)
            source_manifest = {
                "sample_id": sample_id,
                "case_name": case_name,
                "video_path": str(case["video_path"]),
                "condition_path": str(case["control_path"]),
                "frame_indices": case["indices"],
            }
            report = _render_case(args, output_case, source_manifest, rho=rho)
            print(f"diagnosed {position}/{len(ids)} sample_id={sample_id}", flush=True)
        reports.append(report)
    root = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "metadata": str(args.metadata.resolve()),
        "sample_ids": ids,
        "cases": reports,
        "rendering": dict(RENDERING_CONFIG),
        "output": "mp4",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(root, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
