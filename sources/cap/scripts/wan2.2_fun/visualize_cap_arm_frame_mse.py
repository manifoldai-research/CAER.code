#!/usr/bin/env python3
"""Render globally normalized per-frame Arm diagnostic MSE heatmaps."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import arm_mse_heatmap as weight_viz
import visualize_cap_ablation_gt_videos as gt_viz


COLORMAP_STOPS = np.asarray(
    [
        [74, 20, 134],
        [35, 61, 188],
        [20, 145, 212],
        [250, 205, 45],
        [218, 35, 35],
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-visualization-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--metadata-count", type=int, default=365831)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=10)
    parser.add_argument("--model-root", type=Path, default=Path("/dev/shm/wan22-ti2v-local"))
    parser.add_argument("--config", type=Path, default=gt_viz.single.DEFAULT_CONFIG)
    parser.add_argument("--transformer-cache", type=Path, required=True)
    parser.add_argument(
        "--input-cache-root",
        type=Path,
        default=Path("/dev/shm/cap-ablation-gt-inputs"),
    )
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--diagnostic-sigma", type=float, default=0.5)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--overlay-alpha", type=float, default=0.52)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.case_count <= 0:
        raise ValueError("case-count must be positive")
    if args.frames != 17:
        raise ValueError("the Arm checkpoint requires exactly 17 frames")
    if not 0.0 < args.diagnostic_sigma < 1.0:
        raise ValueError("diagnostic-sigma must be in (0, 1)")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha must be in [0, 1]")
    for path, label in (
        (args.source_visualization_dir, "source visualization directory"),
        (args.checkpoint, "checkpoint"),
        (args.metadata, "metadata"),
        (args.model_root, "model root"),
        (args.config, "config"),
        (args.transformer_cache, "transformer cache"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def selected_source_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    root = json.loads(
        (args.source_visualization_dir / "manifest.json").read_text(encoding="utf-8")
    )
    cases = root.get("cases", [])
    if len(cases) < args.case_count:
        raise ValueError(
            f"source manifest has {len(cases)} cases, fewer than {args.case_count}"
        )
    selected = cases[: args.case_count]
    if any(case.get("mode") != "arm" for case in selected):
        raise ValueError("source cases must all use mode=arm")
    return selected


def diagnostic_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        mode="arm",
        checkpoint=args.checkpoint.resolve(),
        metadata=args.metadata.resolve(),
        metadata_count=args.metadata_count,
        output_dir=args.output_dir.resolve(),
        model_root=args.model_root.resolve(),
        config=args.config.resolve(),
        transformer_cache=args.transformer_cache.resolve(),
        input_cache_root=args.input_cache_root.resolve(),
        sample_count=args.case_count,
        sample_seed=0,
        noise_seed=args.noise_seed,
        height=args.height,
        width=args.width,
        frames=args.frames,
        camera_frame_stride=4,
        diagnostic_sigma=args.diagnostic_sigma,
        fps=args.fps,
        device=args.device,
    )


def interpolate_latent_metric(values, frame_count: int):
    import torch.nn.functional as F

    return F.interpolate(
        values[None, None],
        size=frame_count,
        mode="linear",
        # Wan's five latent frames align with RGB frames 0, 4, 8, 12, and 16.
        align_corners=True,
    )[0, 0]


def compute_frame_metrics(args: argparse.Namespace, pipeline, case, sample_id: int):
    import torch

    device = torch.device(args.device)
    clean = weight_viz.encode_video_to_latents(
        pipeline, gt_viz.video_tensor(case["frames"]), device
    )
    generator = torch.Generator(device=device).manual_seed(args.noise_seed + sample_id)
    noise = torch.randn(clean.shape, generator=generator, device=device, dtype=clean.dtype)
    sigma = float(args.diagnostic_sigma)
    noisy = (1.0 - sigma) * clean + sigma * noise
    noisy[:, :, :1] = clean[:, :, :1]
    target = noise.float() - clean.float()
    context, _ = pipeline.encode_prompt(
        case["prompt"], do_classifier_free_guidance=False, device=device
    )
    seq_len = math.ceil(
        clean.shape[2]
        * clean.shape[3]
        * clean.shape[4]
        / (
            pipeline.transformer.config.patch_size[1]
            * pipeline.transformer.config.patch_size[2]
        )
    )
    timestep = torch.full((1,), sigma * 1000.0, device=device, dtype=torch.float32)
    mask = torch.ones((1, 4, *clean.shape[2:]), device=device, dtype=clean.dtype)
    mask[:, :, :1] = 0
    reference = torch.zeros_like(clean)
    reference[:, :, :1] = clean[:, :, :1]
    control = torch.cat([mask, reference], dim=1)
    arm_action = case["action"].to(device=device, dtype=clean.dtype)
    arm_mask = torch.ones((1,), device=device, dtype=torch.float32)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = pipeline.transformer(
            x=noisy,
            context=context,
            t=timestep,
            seq_len=seq_len,
            y=control,
            y_camera=None,
            y_camera_mask=None,
            arm_action=arm_action,
            arm_action_mask=arm_mask,
            full_ref=None,
        )
        null_prediction = pipeline.transformer(
            x=noisy,
            context=context,
            t=timestep,
            seq_len=seq_len,
            y=control,
            y_camera=None,
            y_camera_mask=None,
            arm_action=arm_action,
            arm_action_mask=torch.zeros_like(arm_mask),
            full_ref=None,
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}"
        )
    residual = prediction.float() - target
    latent_frame_mse = residual.square().mean(dim=(1, 3, 4))[0]
    frame_mse = interpolate_latent_metric(latent_frame_mse, args.frames)

    # Match the detached caer training weight and differentiate its normalized
    # weighted loss analytically with respect to the transformer prediction.
    effect = torch.linalg.vector_norm(
        prediction.float() - null_prediction.float(), ord=2, dim=1, keepdim=True
    )
    effect_mean = effect[:, :, 1:].mean(dim=(1, 2, 3, 4), keepdim=True)
    rho = torch.where(
        effect_mean > 1e-6,
        effect / effect_mean.clamp_min(1e-6),
        torch.ones_like(effect),
    )
    prediction_gradient = torch.zeros_like(residual)
    future_rho = rho[:, :, 1:]
    denominator = future_rho.expand_as(residual[:, :, 1:]).sum(
        dim=(1, 2, 3, 4), keepdim=True
    ).clamp_min(1e-6)
    prediction_gradient[:, :, 1:] = (
        2.0 * future_rho * residual[:, :, 1:] / denominator
    )
    latent_frame_gradient_l2 = prediction_gradient.square().sum(
        dim=(1, 3, 4)
    ).sqrt()[0]
    frame_gradient_l2 = interpolate_latent_metric(
        latent_frame_gradient_l2, args.frames
    )
    result = {
        "frame_mse": frame_mse.detach().cpu().numpy().astype(np.float32),
        "latent_frame_mse": latent_frame_mse.detach().cpu().numpy().astype(np.float32),
        "frame_gradient_l2": frame_gradient_l2.detach().cpu().numpy().astype(np.float32),
        "latent_frame_gradient_l2": latent_frame_gradient_l2.detach().cpu().numpy().astype(np.float32),
    }
    del clean, noise, noisy, target, context, control, prediction, null_prediction
    del residual, effect, rho, prediction_gradient
    return result


def color_for_value(value: float, vmin: float, vmax: float) -> np.ndarray:
    if vmax <= vmin:
        position = 0.0
    else:
        position = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    scaled = position * (len(COLORMAP_STOPS) - 1)
    low = min(int(np.floor(scaled)), len(COLORMAP_STOPS) - 1)
    high = min(low + 1, len(COLORMAP_STOPS) - 1)
    fraction = scaled - low
    return COLORMAP_STOPS[low] * (1.0 - fraction) + COLORMAP_STOPS[high] * fraction


def load_font(size: int = 30):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def render_frame(
    rgb: np.ndarray,
    value: float,
    frame_index: int,
    frame_count: int,
    vmin: float,
    vmax: float,
    alpha: float,
    metric_label: str = "MSE",
    value_format: str = ".6f",
) -> np.ndarray:
    color = color_for_value(value, vmin, vmax)
    tinted = np.clip(
        np.asarray(rgb, dtype=np.float32) * (1.0 - alpha) + color * alpha,
        0,
        255,
    ).astype(np.uint8)
    image = Image.fromarray(tinted, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = load_font()
    formatted_value = format(value, value_format)
    label = (
        f"Frame {frame_index + 1:02d}/{frame_count:02d}   "
        f"{metric_label}: {formatted_value}"
    )
    box = draw.textbbox((0, 0), label, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    x, y, pad = 18, 18, 10
    draw.rectangle(
        (x - pad, y - pad, x + width + pad, y + height + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), label, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def raw_paths(output_dir: Path, source_case: dict[str, Any]):
    sample_id = int(source_case["sample_id"])
    case_name = str(source_case["case_name"])
    case_dir = output_dir / f"case_{sample_id:06d}_{case_name}"
    return case_dir, case_dir / "frame_mse_raw.npz"


def render_case(
    args: argparse.Namespace,
    source_case: dict[str, Any],
    record: dict[str, Any],
    mse_vmin: float,
    mse_vmax: float,
    gradient_vmin: float,
    gradient_vmax: float,
) -> dict[str, Any]:
    sample_id = int(source_case["sample_id"])
    case_dir, raw_path = raw_paths(args.output_dir, source_case)
    with np.load(raw_path) as payload:
        frame_mse = payload["frame_mse"].astype(np.float32)
        latent_frame_mse = payload["latent_frame_mse"].astype(np.float32)
        frame_gradient_l2 = payload["frame_gradient_l2"].astype(np.float32)
        latent_frame_gradient_l2 = payload["latent_frame_gradient_l2"].astype(
            np.float32
        )
    case = gt_viz.extract_arm_case(record, diagnostic_args(args), sample_id)
    if len(case["frames"]) != args.frames:
        raise ValueError(
            f"sample {sample_id} has {len(case['frames'])} RGB frames, expected {args.frames}"
        )
    frames_dir = case_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for index, (rgb, value) in enumerate(zip(case["frames"], frame_mse)):
        frame = render_frame(
            rgb,
            float(value),
            index,
            len(frame_mse),
            mse_vmin,
            mse_vmax,
            args.overlay_alpha,
        )
        Image.fromarray(frame).save(frames_dir / f"frame_{index:03d}.png")
        rendered.append(frame)

    video_path = case_dir / "frame_mse_heatmap.mp4"
    gt_viz.write_video(video_path, rendered, args.fps)

    gradient_frames_dir = case_dir / "gradient_frames"
    gradient_frames_dir.mkdir(parents=True, exist_ok=True)
    gradient_rendered = []
    for index, (rgb, value) in enumerate(
        zip(case["frames"], frame_gradient_l2)
    ):
        frame = render_frame(
            rgb,
            float(value),
            index,
            len(frame_gradient_l2),
            gradient_vmin,
            gradient_vmax,
            args.overlay_alpha,
            metric_label="Grad L2",
            value_format=".6e",
        )
        Image.fromarray(frame).save(
            gradient_frames_dir / f"frame_{index:03d}.png"
        )
        gradient_rendered.append(frame)
    gradient_video_path = case_dir / "frame_gradient_heatmap.mp4"
    gt_viz.write_video(gradient_video_path, gradient_rendered, args.fps)

    original_path = case_dir / "original.mp4"
    gt_viz.write_video(original_path, case["frames"], args.fps)
    condition_path = case_dir / "condition.json"
    condition_path.write_text(
        json.dumps(case["condition_data"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    values_path = case_dir / "frame_mse_values.json"
    values_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "frame_indices": case["indices"],
                "frame_mse": [float(value) for value in frame_mse],
                "latent_frame_mse": [float(value) for value in latent_frame_mse],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    gradient_values_path = case_dir / "frame_gradient_values.json"
    gradient_values_path.write_text(
        json.dumps(
            {
                "sample_id": sample_id,
                "frame_indices": case["indices"],
                "metric": "L2 norm of d(caer weighted loss)/d(transformer prediction)",
                "frame_gradient_l2": [
                    float(value) for value in frame_gradient_l2
                ],
                "latent_frame_gradient_l2": [
                    float(value) for value in latent_frame_gradient_l2
                ],
                "exclude_first_latent_frame": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = {
        "sample_id": sample_id,
        "case_name": source_case["case_name"],
        "source_case_manifest": str(
            args.source_visualization_dir
            / f"case_{sample_id:06d}_{source_case['case_name']}"
            / "manifest.json"
        ),
        "video_path": str(case["video_path"]),
        "condition_path": str(case["condition_path"]),
        "frame_indices": case["indices"],
        "frame_count": args.frames,
        "diagnostic_sigma": args.diagnostic_sigma,
        "metric": "conditioned latent flow-matching residual MSE",
        "reduction": "mean over latent channels and spatial tokens",
        "temporal_mapping": "linear interpolation from latent frames to 17 RGB frames",
        "global_color_scale": {"vmin": mse_vmin, "vmax": mse_vmax},
        "gradient_metric": "L2 norm of d(caer weighted loss)/d(transformer prediction)",
        "gradient_reduction": "L2 norm over latent channels and spatial tokens",
        "gradient_excludes_first_latent_frame": True,
        "gradient_global_color_scale": {
            "vmin": gradient_vmin,
            "vmax": gradient_vmax,
        },
        "overlay_alpha": args.overlay_alpha,
        "low_color": COLORMAP_STOPS[0].astype(int).tolist(),
        "high_color": COLORMAP_STOPS[-1].astype(int).tolist(),
        "frame_mse": [float(value) for value in frame_mse],
        "frame_gradient_l2": [float(value) for value in frame_gradient_l2],
        "heatmap_video": str(video_path),
        "frame_images": str(frames_dir),
        "gradient_heatmap_video": str(gradient_video_path),
        "gradient_frame_images": str(gradient_frames_dir),
        "raw_values": str(raw_path),
        "values_json": str(values_path),
        "gradient_values_json": str(gradient_values_path),
        "original_video": str(original_path),
        "condition_data": str(condition_path),
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.source_visualization_dir = args.source_visualization_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.metadata = args.metadata.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_cases = selected_source_cases(args)
    sample_ids = [int(case["sample_id"]) for case in source_cases]
    records = gt_viz.load_selected_records(args.metadata, sample_ids)
    record_by_id = dict(zip(sample_ids, records))
    selection = {
        "source_visualization_dir": str(args.source_visualization_dir),
        "selection": "first cases in source root manifest order",
        "case_count": args.case_count,
        "sample_ids": sample_ids,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    pending = []
    for source_case in source_cases:
        case_dir, raw_path = raw_paths(args.output_dir, source_case)
        case_dir.mkdir(parents=True, exist_ok=True)
        if raw_path.is_file() and raw_path.stat().st_size > 0:
            try:
                with np.load(raw_path) as payload:
                    valid = (
                        payload["frame_mse"].shape == (args.frames,)
                        and payload["frame_gradient_l2"].shape == (args.frames,)
                    )
            except (OSError, ValueError, KeyError):
                valid = False
            if valid:
                print(f"reused raw sample_id={source_case['sample_id']}", flush=True)
                continue
        pending.append(source_case)

    if pending:
        diag_args = diagnostic_args(args)
        transformer = gt_viz.load_transformer(diag_args)
        pipeline = gt_viz.build_pipeline(diag_args, transformer)
        for position, source_case in enumerate(pending, 1):
            sample_id = int(source_case["sample_id"])
            case = gt_viz.extract_arm_case(record_by_id[sample_id], diag_args, sample_id)
            metrics = compute_frame_metrics(args, pipeline, case, sample_id)
            _, raw_path = raw_paths(args.output_dir, source_case)
            np.savez_compressed(raw_path, **metrics)
            print(
                f"computed {position}/{len(pending)} sample_id={sample_id} "
                f"mse=[{metrics['frame_mse'].min():.6f}, "
                f"{metrics['frame_mse'].max():.6f}] "
                f"grad_l2=[{metrics['frame_gradient_l2'].min():.6e}, "
                f"{metrics['frame_gradient_l2'].max():.6e}]",
                flush=True,
            )
            del case
            import torch

            torch.cuda.empty_cache()

    all_mse_values = []
    all_gradient_values = []
    for source_case in source_cases:
        _, raw_path = raw_paths(args.output_dir, source_case)
        with np.load(raw_path) as payload:
            all_mse_values.append(payload["frame_mse"].astype(np.float32))
            all_gradient_values.append(
                payload["frame_gradient_l2"].astype(np.float32)
            )
    combined_mse = np.concatenate(all_mse_values)
    finite_mse = combined_mse[np.isfinite(combined_mse)]
    if finite_mse.size != combined_mse.size:
        raise ValueError("frame MSE contains non-finite values")
    combined_gradient = np.concatenate(all_gradient_values)
    finite_gradient = combined_gradient[np.isfinite(combined_gradient)]
    if finite_gradient.size != combined_gradient.size:
        raise ValueError("frame gradient contains non-finite values")
    mse_vmin = float(finite_mse.min())
    mse_vmax = float(finite_mse.max())
    gradient_vmin = float(finite_gradient.min())
    gradient_vmax = float(finite_gradient.max())

    reports = []
    for position, source_case in enumerate(source_cases, 1):
        sample_id = int(source_case["sample_id"])
        report = render_case(
            args,
            source_case,
            record_by_id[sample_id],
            mse_vmin,
            mse_vmax,
            gradient_vmin,
            gradient_vmax,
        )
        reports.append(report)
        print(f"rendered {position}/{len(source_cases)} sample_id={sample_id}", flush=True)

    root = {
        "mode": "arm",
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "source_visualization_dir": str(args.source_visualization_dir),
        "sample_ids": sample_ids,
        "metric": "conditioned latent flow-matching residual MSE",
        "diagnostic_sigma": args.diagnostic_sigma,
        "global_color_scale": {"vmin": mse_vmin, "vmax": mse_vmax},
        "gradient_metric": "L2 norm of d(caer weighted loss)/d(transformer prediction)",
        "gradient_excludes_first_latent_frame": True,
        "gradient_global_color_scale": {
            "vmin": gradient_vmin,
            "vmax": gradient_vmax,
        },
        "color_semantics": "low=purple/blue, high=red",
        "cases": reports,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
