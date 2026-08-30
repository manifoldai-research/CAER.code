#!/usr/bin/env python3
"""Re-render saved CAP weights without loading the inference model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

import arm_mse_heatmap as weight_viz


MODES = (("s_only", "S_only"), ("e_only", "E_only"))
RENDERING = {
    "normalization": "episode_interpolated_global_min_to_latent_positive_p99",
    "vmin": "episode_interpolated_min_excluding_first_frame",
    "percentile": 99.0,
    "latent_spatial_smoothing": "gaussian_sigma_1.5_before_interpolation",
    "blur": "single_gaussian_radius_12",
    "response_rescaled_after_blur": False,
    "color_response_curve": "normalized_sigmoid_k12_after_blur",
    "colormap": "six_stop_blue_cyan_yellow_red_reference",
    "colormap_levels": [0.0, 0.125, 0.375, 0.625, 0.875, 1.0],
    "colormap_rgb": [
        [0, 0, 128],
        [0, 0, 255],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 0],
        [128, 0, 0],
    ],
    "overlay": "0.55_rgb_plus_0.65_heat",
    "output": "png",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", required=True)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument(
        "--only-se-product",
        action="store_true",
        help="Render only S*E/mean(S*E), preserving existing S_only/E_only PNGs.",
    )
    parser.add_argument(
        "--only-se2-product",
        action="store_true",
        help="Render only S*E^2/mean(S*E^2), preserving existing PNGs.",
    )
    parser.add_argument(
        "--only-ee-product",
        action="store_true",
        help="Render only E*E/mean(E*E), preserving existing PNGs.",
    )
    return parser.parse_args()


def read_video(path: Path, height: int, width: int) -> np.ndarray:
    command = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.check_output(command)
    frame_bytes = height * width * 3
    if len(raw) % frame_bytes:
        raise RuntimeError(f"unexpected decoded byte count for {path}: {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(-1, height, width, 3)


def render_product_mode(
    *,
    report: dict,
    episode_dir: Path,
    frames: np.ndarray,
    selected: list[int],
    chunks: list[np.ndarray],
    frame_count: int,
    output_size: tuple[int, int],
    key: str,
    name: str,
    quantity: str,
    reconstruction_error: float,
) -> None:
    latent = np.concatenate(chunks, axis=0)
    vmax = weight_viz.positive_percentile_vmax(
        latent, percentile=99.0, exclude_first_frame=False
    )
    raw_weights = weight_viz.render_latent_chunks_to_video(
        chunks, frame_count, output_size=output_size, spatial_sigma=0.0
    )
    display_weights = weight_viz.render_latent_chunks_to_video(
        chunks, frame_count, output_size=output_size, spatial_sigma=1.5
    )
    vmin = weight_viz.episode_response_vmin(display_weights, vmax)
    response = weight_viz.normalize_weight_response(display_weights, vmax, vmin=vmin)
    mode_dir = episode_dir / name
    mode_dir.mkdir(parents=True, exist_ok=True)
    array_path = episode_dir / f"{name}_weights.npz"
    np.savez_compressed(array_path, weights=raw_weights)
    pngs = []
    for index in selected:
        overlay = weight_viz.overlay_weight_response(
            frames[index], response[index], blur_radius=12.0
        )
        png_path = mode_dir / f"frame_{index:04d}.png"
        Image.fromarray(overlay, mode="RGB").save(
            png_path, format="PNG", compress_level=2
        )
        pngs.append(str(png_path))
    report.setdefault("weights", {})[key] = {
        "name": name,
        "quantity": quantity,
        "normalization_domain": "each_future_latent_chunk",
        "array": str(array_path),
        "stats": weight_viz.heatmap_stats(raw_weights),
        "normalization": {
            "method": RENDERING["normalization"],
            "vmin": vmin,
            "percentile": 99.0,
            "vmax": vmax,
            "source_latent_reconstruction_max_error": reconstruction_error,
        },
        "pngs": pngs,
    }


def main() -> int:
    args = parse_args()
    only_modes = (
        args.only_se_product,
        args.only_se2_product,
        args.only_ee_product,
    )
    if sum(bool(value) for value in only_modes) > 1:
        raise ValueError("only one product-only mode can be selected")
    episode_ids = [int(value) for value in args.episode_ids.split(",")]
    reports = []
    for episode_id in episode_ids:
        episode = f"episode{episode_id}"
        episode_dir = args.output_dir / episode
        report_path = episode_dir / "manifest.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        arrays = {
            mode: np.load(episode_dir / f"{name}_weights.npz")["weights"]
            for mode, name in MODES
        }
        frame_count = len(arrays["s_only"])
        frames = read_video(
            args.video_dir / f"{episode}.mp4", args.height, args.width
        )
        if len(frames) < frame_count:
            raise RuntimeError(f"video has {len(frames)} frames, need {frame_count}")
        selected = list(range(args.frame_stride, frame_count, args.frame_stride))
        if selected[-1] != frame_count - 1:
            selected.append(frame_count - 1)

        recovered = {}
        reconstruction_errors = {}
        for mode, name in MODES:
            latent_chunks, error = (
                weight_viz.recover_latent_chunks_from_interpolated_weights(
                    arrays[mode]
                )
            )
            if error > 2e-5:
                raise RuntimeError(f"{episode} {name} inverse error {error:.8g}")
            recovered[mode] = latent_chunks
            reconstruction_errors[mode] = error
            if any(only_modes):
                continue
            latent = np.concatenate(latent_chunks, axis=0)
            vmax = weight_viz.positive_percentile_vmax(
                latent, percentile=99.0, exclude_first_frame=False
            )
            display_weights = weight_viz.render_latent_chunks_to_video(
                latent_chunks,
                frame_count,
                output_size=(args.height, args.width),
                spatial_sigma=1.5,
            )
            vmin = weight_viz.episode_response_vmin(display_weights, vmax)
            response = weight_viz.normalize_weight_response(
                display_weights, vmax, vmin=vmin
            )
            mode_dir = episode_dir / name
            mode_dir.mkdir(parents=True, exist_ok=True)
            pngs = []
            for index in selected:
                overlay = weight_viz.overlay_weight_response(
                    frames[index], response[index], blur_radius=12.0
                )
                png_path = mode_dir / f"frame_{index:04d}.png"
                Image.fromarray(overlay, mode="RGB").save(
                    png_path, format="PNG", compress_level=2
                )
                pngs.append(str(png_path))
            spec = report["weights"][mode]
            spec["pngs"] = pngs
            spec["normalization"] = {
                "method": RENDERING["normalization"],
                "vmin": vmin,
                "percentile": 99.0,
                "vmax": vmax,
                "latent_reconstruction_max_error": error,
            }

        product_error = max(reconstruction_errors.values())
        if not args.only_se2_product and not args.only_ee_product:
            render_product_mode(
                report=report,
                episode_dir=episode_dir,
                frames=frames,
                selected=selected,
                chunks=weight_viz.normalize_latent_product_chunks(
                    recovered["s_only"], recovered["e_only"]
                ),
                frame_count=frame_count,
                output_size=(args.height, args.width),
                key="se_only",
                name="SE_only",
                quantity="S * E / mean(S * E)",
                reconstruction_error=product_error,
            )
        if args.only_se2_product:
            squared_e = [chunk * chunk for chunk in recovered["e_only"]]
            render_product_mode(
                report=report,
                episode_dir=episode_dir,
                frames=frames,
                selected=selected,
                chunks=weight_viz.normalize_latent_product_chunks(
                    recovered["s_only"], squared_e
                ),
                frame_count=frame_count,
                output_size=(args.height, args.width),
                key="se2_only",
                name="SE2_only",
                quantity="S * E^2 / mean(S * E^2)",
                reconstruction_error=product_error,
            )
        if args.only_ee_product:
            render_product_mode(
                report=report,
                episode_dir=episode_dir,
                frames=frames,
                selected=selected,
                chunks=weight_viz.normalize_latent_product_chunks(
                    recovered["e_only"], recovered["e_only"]
                ),
                frame_count=frame_count,
                output_size=(args.height, args.width),
                key="ee_only",
                name="EE_only",
                quantity="E * E / mean(E * E)",
                reconstruction_error=product_error,
            )

        for stale in (
            "alpha_range", "neighborhood_radius", "neighborhood_mix"
        ):
            report.pop(stale, None)
        report.update(RENDERING)
        report["blur_radius"] = 12.0
        report["selected_frames"] = selected
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        reports.append(report)
        print(f"rendered {episode}", flush=True)

    root_report = {
        "episode_ids": episode_ids,
        "episodes": reports,
        "rendering": dict(RENDERING),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
