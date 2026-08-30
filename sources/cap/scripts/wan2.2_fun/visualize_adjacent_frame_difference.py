#!/usr/bin/env python3
"""Render CPU-only heatmaps from lagged RGB frame differences."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import arm_mse_heatmap as heatmap


REFERENCE_LEVELS = np.asarray(
    [0.0, 0.125, 0.375, 0.625, 0.875, 1.0], dtype=np.float32
)
REFERENCE_COLORS = np.asarray(
    [
        [0, 0, 128],
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 0],
        [128, 0, 0],
    ],
    dtype=np.float32,
)


def colorize_reference(response: np.ndarray) -> np.ndarray:
    """Apply the requested nonuniform blue-cyan-yellow-red reference scale."""

    values = np.clip(np.asarray(response, dtype=np.float32), 0.0, 1.0)
    channels = [
        np.interp(values, REFERENCE_LEVELS, REFERENCE_COLORS[:, channel])
        for channel in range(3)
    ]
    return np.uint8(np.clip(np.stack(channels, axis=-1), 0, 255))


def overlay_reference(rgb_frame: np.ndarray, response: np.ndarray) -> np.ndarray:
    smooth = heatmap.smooth_weight_response(response, blur_radius=12.0)
    colored = colorize_reference(smooth)
    return np.uint8(np.clip(
        rgb_frame.astype(np.float32) * 0.55
        + colored.astype(np.float32) * 0.65,
        0,
        255,
    ))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--frame-lag", type=int, default=4)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    return parser.parse_args()


def decode_video(path: Path, height: int, width: int) -> np.ndarray:
    raw = subprocess.check_output(
        [
            "ffmpeg", "-v", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ]
    )
    frame_bytes = height * width * 3
    if len(raw) % frame_bytes:
        raise RuntimeError(f"unexpected decoded byte count for {path}: {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def render_episode(args: argparse.Namespace, episode_id: int) -> dict:
    episode = f"episode{episode_id}"
    video_path = args.video_dir / f"{episode}.mp4"
    frames = decode_video(video_path, args.height, args.width)
    if len(frames) <= args.frame_lag:
        raise ValueError(
            f"{video_path} needs more than {args.frame_lag} frames"
        )

    frame_float = frames.astype(np.float32)
    differences = np.abs(
        frame_float[args.frame_lag:] - frame_float[:-args.frame_lag]
    ).mean(axis=-1)
    positive = differences[differences > 0]
    vmax = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    response = np.clip(differences / max(vmax, 1e-8), 0.0, 1.0)

    selected = list(
        range(
            max(args.frame_stride, args.frame_lag),
            len(frames),
            args.frame_stride,
        )
    )
    if selected[-1] != len(frames) - 1:
        selected.append(len(frames) - 1)
    mode_dir = args.output_dir / episode / "Four_frame_difference"
    mode_dir.mkdir(parents=True, exist_ok=True)
    pngs = []
    for index in selected:
        overlay = overlay_reference(
            frames[index], response[index - args.frame_lag]
        )
        png_path = mode_dir / f"frame_{index:04d}.png"
        Image.fromarray(overlay, mode="RGB").save(
            png_path, format="PNG", compress_level=2
        )
        pngs.append(str(png_path))

    report = {
        "episode": episode,
        "source_video": str(video_path),
        "quantity": "mean_rgb_absolute_difference_at_frame_lag",
        "difference": f"mean_c(abs(I_t - I_t_minus_{args.frame_lag}))",
        "frame_lag": args.frame_lag,
        "initial_frames_excluded": args.frame_lag,
        "normalization": "episode_positive_p99_before_blur",
        "vmin": None,
        "percentile": 99.0,
        "vmax": vmax,
        "frame_stride": args.frame_stride,
        "selected_frames": selected,
        "blur": "single_gaussian_radius_12",
        "response_rescaled_after_blur": False,
        "colormap": "six_stop_blue_cyan_yellow_red_reference",
        "colormap_levels": REFERENCE_LEVELS.tolist(),
        "colormap_rgb": REFERENCE_COLORS.astype(np.uint8).tolist(),
        "overlay": "0.55_rgb_plus_0.65_heat",
        "output": "png",
        "output_size": [args.width, args.height],
        "pngs": pngs,
    }
    (mode_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("frame-stride must be positive")
    if args.frame_lag <= 0:
        raise ValueError("frame-lag must be positive")
    episode_ids = [int(value.strip()) for value in args.episode_ids.split(",")]
    reports = [render_episode(args, episode_id) for episode_id in episode_ids]
    report_path = args.output_dir / f"frame_difference_lag_{args.frame_lag}_manifest.json"
    report_path.write_text(
        json.dumps({"episode_ids": episode_ids, "episodes": reports}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"adjacent-frame heatmaps complete: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
