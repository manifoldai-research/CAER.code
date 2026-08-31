#!/usr/bin/env python3
"""Export CAER and MSE weight overlays for one WorldArena episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import arm_mse_heatmap as weight_viz
import infer_cap_arm_sample as single
import infer_cap_arm_worldarena_batch as batch


WEIGHT_MODES = ("MSE", "CAER")
RENDERING_CONFIG = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--episode-ids",
        default="50,7,48",
        help="Comma-separated episode IDs processed sequentially with one model load.",
    )
    parser.add_argument("--variant", choices=single.VARIANTS, default="CAER")
    parser.add_argument("--checkpoint-step", type=int, default=5000)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=single.DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=single.DEFAULT_CACHE_ROOT)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--action-downsample", type=int, default=3)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--diagnostic-sigma", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--blur-radius", type=float, default=12.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--negative-prompt", default=single.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument(
        "--skip-complete",
        action="store_true",
        help="Skip episodes whose manifest and all referenced artifacts already exist.",
    )
    parser.add_argument(
        "--defer-root-manifest",
        action="store_true",
        help="Write episode manifests only; merge the root manifest after parallel workers finish.",
    )
    return parser.parse_args()


def select_case(manifest_path: Path, episode_id: int) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode = f"episode{episode_id}"
    matches = [case for case in manifest.get("cases", []) if case.get("episode") == episode]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {episode} in {manifest_path}")
    return matches[0]


def parse_episode_ids(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part or not part.isdigit() or int(part) <= 0 for part in parts):
        raise ValueError("episode-ids must be comma-separated positive integers")
    episode_ids = tuple(int(part) for part in parts)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode-ids must not contain duplicates")
    return episode_ids


def capture_episode_weights(args, case, pipeline, config, device):
    import torch
    from PIL import Image
    from videox_fun.utils.utils import get_image_to_video_latent

    action = np.load(case["action_path"]).astype(np.float32)[:: args.action_downsample]
    if action.ndim != 2 or action.shape[1] != 14 or len(action) < 2:
        raise ValueError(f"invalid downsampled action for {case['episode']}: {action.shape}")
    current_frame = Image.open(case["first_frame_path"]).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.BILINEAR
    )
    gt_reader, gt_indices, _, _ = batch.open_ground_truth(
        case, len(action), args.action_downsample
    )
    chunks = {mode: [] for mode in WEIGHT_MODES}
    display_chunks = {mode: [] for mode in WEIGHT_MODES}
    latent_chunks = {mode: [] for mode in WEIGHT_MODES}
    stride = args.frames - 1
    frame_index = 0
    chunk_index = 0
    while frame_index < len(action) - 1:
        arm = batch.action_chunk(action, frame_index, args.frames)
        video, mask, _ = get_image_to_video_latent(
            [current_frame], None, video_length=args.frames,
            sample_size=[args.height, args.width],
        )
        target_video = batch.load_ground_truth_chunk(
            gt_reader, gt_indices, frame_index, args.frames, args.height, args.width
        )
        target_latents = weight_viz.encode_video_to_latents(pipeline, target_video, device)
        capture = weight_viz.Method1HeatmapCapture(
            pipeline, target_latents, WEIGHT_MODES,
            sigma=args.diagnostic_sigma, eps=args.eps,
        )
        seed = args.generation_seed + int(case["episode_id"]) * 1000 + chunk_index
        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode(), capture:
            sample = pipeline(
                case["instruction"],
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                video=video,
                mask_video=mask,
                control_video=None,
                arm_action=torch.from_numpy(arm).unsqueeze(0),
                arm_action_mask=torch.ones((1,), dtype=torch.float32),
                num_frames=args.frames,
                num_inference_steps=args.inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                boundary=float(config["transformer_additional_kwargs"].get("boundary", 0.9)),
                shift=int(config["scheduler_kwargs"].get("shift", 5)),
                use_empty_control_latents=False,
            ).videos.cpu()
        actual = min(args.frames, len(action) - frame_index)
        for mode in WEIGHT_MODES:
            latent_chunks[mode].append(
                capture.rho_maps[mode][0, 0].numpy().astype(np.float32, copy=False)
            )
            rendered = weight_viz.render_map_to_video(
                capture.rho_maps[mode], args.frames,
                output_size=(args.height, args.width),
            )
            smoothed_rendered = weight_viz.render_map_to_video(
                weight_viz.smooth_latent_spatially(
                    capture.rho_maps[mode][0, 0].numpy(), sigma=1.5
                ),
                args.frames,
                output_size=(args.height, args.width),
            )
            chunks[mode].append(
                rendered[:actual] if chunk_index == 0 else rendered[1:actual]
            )
            display_chunks[mode].append(
                smoothed_rendered[:actual]
                if chunk_index == 0
                else smoothed_rendered[1:actual]
            )
        current_frame = batch.sample_to_frame(sample)
        frame_index += stride
        chunk_index += 1
        print(
            f"captured {case['episode']} chunk={chunk_index} frames={actual}",
            flush=True,
        )
    weights = {
        mode: np.concatenate(mode_chunks, axis=0).astype(np.float32, copy=False)
        for mode, mode_chunks in chunks.items()
    }
    display_weights = {
        mode: np.concatenate(mode_chunks, axis=0).astype(np.float32, copy=False)
        for mode, mode_chunks in display_chunks.items()
    }
    latent_vmax = {
        mode: weight_viz.positive_percentile_vmax(
            np.concatenate(mode_chunks, axis=0),
            percentile=RENDERING_CONFIG["percentile"],
            # compute_rho_maps already removed each chunk's fixed first frame.
            exclude_first_frame=False,
        )
        for mode, mode_chunks in latent_chunks.items()
    }
    return weights, display_weights, latent_vmax


def read_selected_video_frames(video_path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    if not indices or indices[-1] >= len(reader):
        raise ValueError(f"selected frame exceeds video length {len(reader)}: {video_path}")
    decoded = reader.get_batch(np.asarray(indices, dtype=np.int64)).asnumpy()
    return {index: frame for index, frame in zip(indices, decoded)}


def load_complete_episode_report(output_dir: Path, episode: str) -> dict | None:
    manifest_path = output_dir / episode / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if report.get("episode") != episode:
        return None
    if any(report.get(key) != value for key, value in RENDERING_CONFIG.items()):
        return None
    if report.get("blur_radius") != 12.0:
        return None
    weights = report.get("weights", {})
    for mode in WEIGHT_MODES:
        spec = weights.get(mode)
        if not isinstance(spec, dict) or not Path(spec.get("array", "")).is_file():
            return None
        pngs = spec.get("pngs")
        if not isinstance(pngs, list) or not pngs or not all(Path(path).is_file() for path in pngs):
            return None
    return report


def write_root_manifest(args, checkpoint: Path, episode_ids, reports: list[dict]) -> Path:
    report = {
        "checkpoint": str(checkpoint),
        "diagnostic_sigma": args.diagnostic_sigma,
        "mse_used": False,
        "episode_ids": list(episode_ids),
        "episodes": reports,
        "rendering": dict(RENDERING_CONFIG),
    }
    report_path = args.output_dir / "manifest.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(report_path)
    return report_path


def export_overlays(
    args,
    case,
    weights: dict[str, np.ndarray],
    latent_vmax: dict[str, float],
    output_dir: Path,
    display_weights: dict[str, np.ndarray] | None = None,
) -> dict:
    from PIL import Image

    video_path = args.video_dir / f"{case['episode']}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    lengths = {mode: len(values) for mode, values in weights.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"weight length mismatch: {lengths}")
    frame_count = next(iter(lengths.values()))
    selected = list(range(args.frame_stride, frame_count, args.frame_stride))
    if frame_count > 1 and (not selected or selected[-1] != frame_count - 1):
        selected.append(frame_count - 1)
    original_frames = read_selected_video_frames(video_path, selected)
    output = {}
    for mode in WEIGHT_MODES:
        mode_name = mode
        mode_dir = output_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        array_path = output_dir / f"{mode_name}_weights.npz"
        np.savez_compressed(array_path, weights=weights[mode])
        vmax = float(latent_vmax[mode])
        rendered_weights = weights[mode] if display_weights is None else display_weights[mode]
        vmin = weight_viz.episode_response_vmin(rendered_weights, vmax)
        response = weight_viz.normalize_weight_response(
            rendered_weights, vmax, vmin=vmin
        )
        pngs = []
        for index in selected:
            frame = original_frames[index]
            if tuple(frame.shape[:2]) != (args.height, args.width):
                raise RuntimeError(
                    f"source frame size changed: {frame.shape[:2]} != {(args.height, args.width)}"
                )
            overlay = weight_viz.overlay_weight_response(
                frame, response[index],
                blur_radius=args.blur_radius,
            )
            png_path = mode_dir / f"frame_{index:04d}.png"
            Image.fromarray(overlay, mode="RGB").save(
                png_path, format="PNG", compress_level=2
            )
            pngs.append(str(png_path))
        output[mode] = {
            "name": mode_name,
            "quantity": "S / mean(S)" if mode == "CAER" else "MSE",
            "array": str(array_path),
            "stats": weight_viz.heatmap_stats(weights[mode]),
            "normalization": {
                "method": RENDERING_CONFIG["normalization"],
                "vmin": vmin,
                "percentile": RENDERING_CONFIG["percentile"],
                "vmax": vmax,
            },
            "pngs": pngs,
        }
    return {
        "episode": case["episode"],
        "source_video": str(video_path),
        "source_frame_count": frame_count,
        "selected_frames": selected,
        "output_size": [args.width, args.height],
        "blur_radius": args.blur_radius,
        **RENDERING_CONFIG,
        "weights": output,
    }


def main() -> int:
    args = parse_args()
    episode_ids = parse_episode_ids(args.episode_ids)
    if args.frame_stride <= 0:
        raise ValueError("frame-stride must be positive")
    if args.blur_radius != 12.0:
        raise ValueError("blur-radius is fixed at 12 for PNG rendering")
    if args.frames != 17 or args.action_downsample <= 0:
        raise ValueError("frames must be 17 and action-downsample must be positive")
    args.case_manifest = args.case_manifest.expanduser().resolve()
    args.video_dir = args.video_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    cases = [select_case(args.case_manifest, episode_id) for episode_id in episode_ids]
    _, checkpoint, step = single.resolve_run_and_checkpoint(args)
    if step != args.checkpoint_step:
        raise RuntimeError("resolved checkpoint step changed")
    cache_dir = single.cache_directory(args, args.run_dir, step)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports_by_episode = {}
    pending_cases = []
    for case in cases:
        complete = (
            load_complete_episode_report(args.output_dir, case["episode"])
            if args.skip_complete
            else None
        )
        if complete is not None:
            reports_by_episode[case["episode"]] = complete
            print(f"skipping complete {case['episode']}", flush=True)
        else:
            pending_cases.append(case)
    if pending_cases:
        pipeline, config, device = single.build_pipeline(args, checkpoint, cache_dir)
    for case in pending_cases:
        weights, display_weights, latent_vmax = capture_episode_weights(
            args, case, pipeline, config, device
        )
        episode_output = args.output_dir / case["episode"]
        report = export_overlays(
            args, case, weights, latent_vmax, episode_output,
            display_weights=display_weights,
        )
        reports_by_episode[case["episode"]] = report
        episode_manifest = episode_output / "manifest.json"
        episode_manifest.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not args.defer_root_manifest:
            ordered_reports = [
                reports_by_episode[f"episode{episode_id}"]
                for episode_id in episode_ids
                if f"episode{episode_id}" in reports_by_episode
            ]
            write_root_manifest(args, checkpoint, episode_ids, ordered_reports)
    ordered_reports = [reports_by_episode[f"episode{episode_id}"] for episode_id in episode_ids]
    if args.defer_root_manifest:
        print(
            f"weight visualization worker complete: {len(ordered_reports)} episodes; "
            "root manifest deferred",
            flush=True,
        )
    else:
        report_path = write_root_manifest(args, checkpoint, episode_ids, ordered_reports)
        print(f"weight visualization complete: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
