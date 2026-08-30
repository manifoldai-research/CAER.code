#!/usr/bin/env python3
"""Generate one Camera-100 video and its canonical CAP S/E overlays.

The camera benchmark inference code remains the source of truth for model and
pipeline construction.  This entrypoint only adds GT VAE encoding, the
``Method1HeatmapCapture`` context, and the documented PNG/video + NPZ export.
Heatmaps are composited over the corresponding preprocessed GT frames, while
the generated MP4 remains the normal inference output.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
import os
from pathlib import Path

import numpy as np
from PIL import Image


CAP_SCRIPT_DIR = Path(__file__).resolve().parent
CAP_PROJECT_ROOT = CAP_SCRIPT_DIR.parents[1]
HOST_SCRIPT_DIR = Path(
    os.environ.get(
        "CAP_CAMERA_HOST_SCRIPT_DIR",
        str(Path(os.environ.get("CAP_CAMERA_HOST_PROJECT_ROOT", CAP_PROJECT_ROOT)) / "scripts/wan2.2_fun"),
    )
)
if str(CAP_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(CAP_SCRIPT_DIR))
if str(CAP_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(CAP_PROJECT_ROOT))
if str(HOST_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HOST_SCRIPT_DIR))

import torch

import arm_mse_heatmap as weight_viz
import infer_selected100_camera_benchmark as camera_infer
from visualize_cap_arm_weights import RENDERING_CONFIG, WEIGHT_MODES

# The protected host module's ``load_first_frame`` references Path in its
# module globals but does not import it.  Supply that missing standard-library
# symbol without changing the host inference implementation.
camera_infer.camera_batch.Path = Path


DEFAULT_MODEL_ROOT = Path(os.environ.get("CAP_CONTROL_MODEL", "models/ti2v_control_init_model"))
DEFAULT_CAMERA_ROOT = Path(os.environ.get("CAMERA_ROOT", "data/camera/trajectories"))
DEFAULT_SELECTED_CSV = Path(os.environ.get("SELECTED_CSV", "data/camera/selected_100.csv"))
DEFAULT_ACTION_CSV = Path(os.environ.get("ACTION_CSV", "data/camera/action.csv"))
DEFAULT_CONFIG = CAP_PROJECT_ROOT / "config/wan2.2/wan_civitai_5b.yaml"
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model_name", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--selected_csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--camera_root", type=Path, default=DEFAULT_CAMERA_ROOT)
    parser.add_argument("--action_csv", type=Path, default=DEFAULT_ACTION_CSV)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--prompt_column", default="scene_description")
    parser.add_argument("--sample_height", type=int, default=704)
    parser.add_argument("--sample_width", type=int, default=1280)
    parser.add_argument("--start_image_center_crop_size", type=int, default=720)
    parser.add_argument("--video_length", type=int, default=81)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=50025)
    parser.add_argument("--sample_count", type=int, default=1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--diagnostic_sigma", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument(
        "--heatmap_output_format",
        choices=("frames", "video"),
        default="frames",
        help="Save selected overlay PNG frames or a full overlay MP4 per mode.",
    )
    parser.add_argument("--result_file_name", default="results.json")
    parser.add_argument("--selection_file_name", default="selection.json")
    parser.add_argument("--skip_complete", action="store_true")
    parser.add_argument("--defer_root_manifest", action="store_true")
    parser.add_argument("--camera_moe_root", default="")
    parser.add_argument("--moe_mode", default="control_expert")
    parser.add_argument("--moe_all_blocks", action="store_true")
    parser.add_argument("--moe_route_temperature", type=float, default=1.0)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def host_case_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the small namespace consumed by the host case loader."""

    return argparse.Namespace(
        selected_csv=str(args.selected_csv),
        camera_root=str(args.camera_root),
        action_csv=str(args.action_csv),
        prompt_column=args.prompt_column,
        sample_count=args.sample_count,
        start_index=args.start_index,
    )


def build_pipeline_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the namespace consumed by the host pipeline loader."""

    return argparse.Namespace(
        config_path=str(args.config_path),
        model_name=str(args.model_name),
        checkpoint_path=str(args.checkpoint_path),
        camera_moe_root=args.camera_moe_root,
        moe_all_blocks=args.moe_all_blocks,
        moe_mode=args.moe_mode,
        moe_route_temperature=args.moe_route_temperature,
        device=args.device,
    )


def load_gt_frames(
    video_path: Path,
    frame_count: int,
    center_crop_size: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Read/pad GT RGB frames using the same crop and output size as inference."""

    from decord import VideoReader, cpu

    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    if len(reader) == 0:
        raise RuntimeError(f"GT video has no frames: {video_path}")
    indices = np.arange(min(len(reader), frame_count), dtype=np.int64)
    frames = reader.get_batch(indices).asnumpy()
    if len(frames) < frame_count:
        pad = np.repeat(frames[-1:], frame_count - len(frames), axis=0)
        frames = np.concatenate([frames, pad], axis=0)

    processed = []
    for frame in frames[:frame_count]:
        image = Image.fromarray(frame).convert("RGB")
        if center_crop_size > 0:
            image = camera_infer.camera_batch.center_crop_and_resize_image(
                image, center_crop_size
            )
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        processed.append(np.asarray(image, dtype=np.uint8))
    return np.stack(processed, axis=0)


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0


def sample_to_rgb_frames(sample: torch.Tensor) -> np.ndarray:
    if not torch.is_tensor(sample) or sample.ndim != 5:
        raise ValueError(f"pipeline sample must be [B,C,T,H,W], got {type(sample)} {getattr(sample, 'shape', None)}")
    values = sample.detach().float().cpu().clamp(0.0, 1.0)
    if values.shape[1] != 3:
        raise ValueError(f"pipeline sample must have 3 RGB channels, got {values.shape[1]}")
    return (
        values[0].permute(1, 2, 3, 0).numpy() * 255.0
    ).round().astype(np.uint8)


def prepare_camera_video(case: dict, args: argparse.Namespace, sample_size: list[int]) -> torch.Tensor:
    camera = camera_infer.camera_batch.process_pose_file(
        case["control_camera_txt"], sample_size[1], sample_size[0]
    )
    if camera.shape[0] < args.video_length:
        camera = torch.cat(
            [camera, camera[-1:].repeat(args.video_length - camera.shape[0], 1, 1, 1)],
            dim=0,
        )
    camera = camera[: args.video_length]
    return camera.permute(3, 0, 1, 2).unsqueeze(0)


def heatmap_rendering_config(args: argparse.Namespace) -> dict:
    config = dict(RENDERING_CONFIG)
    if args.heatmap_output_format == "video":
        config["output"] = "mp4"
    return config


def save_overlay_video(
    background_frames: np.ndarray,
    response: np.ndarray,
    path: Path,
    fps: int,
) -> None:
    """Write every GT-backed overlay frame to an MP4 without a large RGB buffer."""

    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    if response.ndim != 3 or len(response) != len(background_frames):
        raise ValueError(
            f"overlay response/background mismatch: {response.shape} vs {background_frames.shape}"
        )
    writer = imageio.get_writer(path, fps=int(fps), macro_block_size=1)
    try:
        for background, frame_response in zip(background_frames, response):
            overlay = weight_viz.overlay_weight_response(
                background, frame_response, blur_radius=12.0
            )
            writer.append_data(np.asarray(overlay, dtype=np.uint8))
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"video writer produced an empty file: {path}")


def export_heatmaps(
    args: argparse.Namespace,
    case: dict,
    background_frames: np.ndarray,
    rho_maps: dict[str, torch.Tensor],
    case_dir: Path,
) -> dict:
    if background_frames.shape[0] != args.video_length:
        raise RuntimeError(
            f"GT frame count mismatch: {background_frames.shape[0]} != {args.video_length}"
        )
    if tuple(background_frames.shape[1:]) != (args.sample_height, args.sample_width, 3):
        raise RuntimeError(
            "GT frame shape mismatch: "
            f"{background_frames.shape[1:]} != {(args.sample_height, args.sample_width, 3)}"
        )
    if set(rho_maps) != set(WEIGHT_MODES):
        raise RuntimeError(f"expected both heatmap modes, got {sorted(rho_maps)}")

    if args.heatmap_output_format == "video":
        selected = list(range(args.video_length))
    else:
        selected = list(range(args.frame_stride, args.video_length, args.frame_stride))
        if args.video_length > 1 and (not selected or selected[-1] != args.video_length - 1):
            selected.append(args.video_length - 1)

    case_dir.mkdir(parents=True, exist_ok=True)
    report_weights = {}
    rendering = heatmap_rendering_config(args)
    for mode, mode_name in (("s_only", "S_only"), ("e_only", "E_only")):
        rho = rho_maps[mode]
        latent = rho[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
        if latent.ndim != 3 or not np.isfinite(latent).all():
            raise RuntimeError(f"{mode} latent rho is invalid: {latent.shape}")
        raw = weight_viz.render_map_to_video(
            rho, args.video_length,
            output_size=(args.sample_height, args.sample_width),
        )
        display = weight_viz.render_map_to_video(
            weight_viz.smooth_latent_spatially(latent, sigma=1.5),
            args.video_length,
            output_size=(args.sample_height, args.sample_width),
        )
        vmax = weight_viz.positive_percentile_vmax(
            latent, percentile=RENDERING_CONFIG["percentile"], exclude_first_frame=False
        )
        vmin = weight_viz.episode_response_vmin(display, vmax)
        response = weight_viz.normalize_weight_response(display, vmax, vmin=vmin)

        array_path = case_dir / f"{mode_name}_weights.npz"
        np.savez_compressed(array_path, weights=raw.astype(np.float32, copy=False))
        mode_dir = case_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        if args.heatmap_output_format == "video":
            video_path = mode_dir / "heatmap.mp4"
            save_overlay_video(background_frames, response, video_path, args.fps)
            artifacts["video"] = str(video_path)
        else:
            pngs = []
            for index in selected:
                overlay = weight_viz.overlay_weight_response(
                    background_frames[index], response[index], blur_radius=12.0
                )
                png_path = mode_dir / f"frame_{index:04d}.png"
                Image.fromarray(overlay, mode="RGB").save(
                    png_path, format="PNG", compress_level=2
                )
                pngs.append(str(png_path))
            artifacts["pngs"] = pngs
        report_weights[mode] = {
            "name": mode_name,
            "quantity": "S / mean(S)" if mode == "s_only" else "E / mean(E)",
            "array": str(array_path),
            "stats": weight_viz.heatmap_stats(raw),
            "latent_stats": weight_viz.heatmap_stats(latent),
            "normalization": {
                "method": rendering["normalization"],
                "vmin": float(vmin),
                "vmax": float(vmax),
                "percentile": rendering["percentile"],
            },
            **artifacts,
        }

    report = {
        "case_id": int(case["row_index"]),
        "case_name": Path(case["source_absolute_path"]).stem,
        "source_video": str(case["source_absolute_path"]),
        "control_camera_txt": str(case["control_camera_txt"]),
        "generated_video": str(case_dir / "generated.mp4"),
        "heatmap_background": "ground_truth",
        "heatmap_output_format": args.heatmap_output_format,
        "frame_count": args.video_length,
        "selected_frames": selected,
        "output_size": [args.sample_width, args.sample_height],
        "diagnostic_sigma": args.diagnostic_sigma,
        "mse_used": False,
        "blur_radius": 12.0,
        **rendering,
        "weights": report_weights,
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def case_directory(output_dir: Path, case: dict) -> Path:
    return output_dir / (
        f"case_{int(case['row_index']):03d}_"
        f"{camera_infer.camera_batch.sanitize_name(Path(case['source_absolute_path']).stem)}"
    )


def load_complete_case_report(args: argparse.Namespace, case: dict) -> dict | None:
    case_dir = case_directory(args.output_dir, case)
    manifest_path = case_dir / "manifest.json"
    generated_video = case_dir / "generated.mp4"
    if not manifest_path.is_file() or not generated_video.is_file():
        return None
    if generated_video.stat().st_size <= 0:
        return None
    try:
        report = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if report.get("case_id") != int(case["row_index"]):
        return None
    if report.get("heatmap_background") != "ground_truth":
        return None
    if report.get("heatmap_output_format", "frames") != args.heatmap_output_format:
        return None
    rendering = heatmap_rendering_config(args)
    if any(report.get(key) != value for key, value in rendering.items()):
        return None
    weights = report.get("weights", {})
    for mode in WEIGHT_MODES:
        spec = weights.get(mode)
        if not isinstance(spec, dict):
            return None
        array_path = Path(spec.get("array", ""))
        if not array_path.is_file() or array_path.stat().st_size <= 0:
            return None
        if args.heatmap_output_format == "video":
            video_path = Path(spec.get("video", ""))
            if not video_path.is_file() or video_path.stat().st_size <= 0:
                return None
        else:
            pngs = spec.get("pngs")
            if not isinstance(pngs, list) or not pngs:
                return None
            if not all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in pngs):
                return None
    return report


def infer_case(
    args: argparse.Namespace,
    case: dict,
    pipeline,
    boundary: float,
) -> tuple[dict, Path]:
    sample_size = [args.sample_height, args.sample_width]
    first_frame = camera_infer.camera_batch.load_first_frame(case["source_absolute_path"])
    if args.start_image_center_crop_size > 0:
        first_frame = camera_infer.camera_batch.center_crop_and_resize_image(
            first_frame, args.start_image_center_crop_size
        )
    inpaint_video, inpaint_mask, _ = camera_infer.camera_batch.get_image_to_video_latent(
        [first_frame], None, video_length=args.video_length, sample_size=sample_size
    )
    control_camera_video = prepare_camera_video(case, args, sample_size)

    print(
        f"case {case['row_index']}: loading GT frames and encoding with the inference VAE",
        flush=True,
    )
    gt_frames = load_gt_frames(
        Path(case["source_absolute_path"]),
        args.video_length,
        args.start_image_center_crop_size,
        args.sample_height,
        args.sample_width,
    )
    target_latents = weight_viz.encode_video_to_latents(
        pipeline, frames_to_tensor(gt_frames), args.device
    )

    case_dir = case_directory(args.output_dir, case)
    generator = torch.Generator(device=args.device).manual_seed(
        args.seed + int(case["row_index"])
    )
    capture = weight_viz.Method1HeatmapCapture(
        pipeline,
        target_latents,
        WEIGHT_MODES,
        sigma=args.diagnostic_sigma,
        eps=args.eps,
    )
    print(
        f"case {case['row_index']}: sampling with simultaneous S_only/E_only capture",
        flush=True,
    )
    with torch.inference_mode(), capture:
        sample = pipeline(
            case["prompt"],
            num_frames=args.video_length,
            negative_prompt=args.negative_prompt,
            height=args.sample_height,
            width=args.sample_width,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            video=inpaint_video,
            mask_video=inpaint_mask,
            control_camera_video=control_camera_video,
            boundary=boundary,
            shift=5,
        ).videos

    if not capture.rho_maps:
        raise RuntimeError("heatmap capture returned no rho maps")
    case_dir.mkdir(parents=True, exist_ok=True)
    generated_video = case_dir / "generated.mp4"
    camera_infer.camera_batch.save_videos_grid(
        sample, str(generated_video), fps=args.fps
    )
    print(f"case {case['row_index']}: saved generated video: {generated_video}", flush=True)
    report = export_heatmaps(args, case, gt_frames, capture.rho_maps, case_dir)
    del sample, capture, target_latents, gt_frames
    return report, generated_video


def build_result(
    args: argparse.Namespace,
    all_cases: list[dict],
    selected_cases: list[dict],
    successes: list[dict],
    failures: list[dict],
) -> dict:
    return {
        "checkpoint_path": str(args.checkpoint_path),
        "selected_csv": str(args.selected_csv),
        "start_index": args.start_index,
        "sample_count_requested": args.sample_count,
        "total_cases": len(all_cases),
        "selected_count": len(selected_cases),
        "num_success": len(successes),
        "num_failures": len(failures),
        "successes": successes,
        "failures": failures,
        "heatmap_output_format": args.heatmap_output_format,
        "rendering": heatmap_rendering_config(args),
    }


def run(args: argparse.Namespace) -> dict:
    args.checkpoint_path = args.checkpoint_path.expanduser().resolve()
    args.model_name = args.model_name.expanduser().resolve()
    args.config_path = args.config_path.expanduser().resolve()
    args.selected_csv = args.selected_csv.expanduser().resolve()
    args.camera_root = args.camera_root.expanduser().resolve()
    args.action_csv = args.action_csv.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.checkpoint_path.is_file() or args.checkpoint_path.stat().st_size <= 0:
        raise FileNotFoundError(f"checkpoint is missing or empty: {args.checkpoint_path}")
    if not args.camera_root.is_dir():
        raise FileNotFoundError(f"camera root is missing: {args.camera_root}")

    all_cases, selected_cases = camera_infer.load_cases(host_case_args(args))
    if not selected_cases:
        raise RuntimeError("no cases selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / args.selection_file_name,
        {
            "selected_csv": str(args.selected_csv),
            "total_cases": len(all_cases),
            "start_index": args.start_index,
            "sample_count": args.sample_count,
            "cases": selected_cases,
        },
    )

    print(f"checkpoint: {args.checkpoint_path}", flush=True)
    print(f"output: {args.output_dir}", flush=True)
    print(
        f"selected rows: {selected_cases[0]['row_index']}..{selected_cases[-1]['row_index']} "
        f"count={len(selected_cases)}",
        flush=True,
    )

    reports = []
    successes = []
    failures = []
    pending_cases = []
    for case in selected_cases:
        complete = load_complete_case_report(args, case) if args.skip_complete else None
        if complete is None:
            pending_cases.append(case)
            continue
        reports.append(complete)
        generated_video = case_directory(args.output_dir, case) / "generated.mp4"
        successes.append(
            {
                **case,
                "output_path": str(generated_video),
                "heatmap_manifest": str(case_directory(args.output_dir, case) / "manifest.json"),
                "skipped_complete": True,
            }
        )
        print(f"case {case['row_index']}: skipping complete output", flush=True)

    pipeline = boundary = None
    if pending_cases:
        pipeline, boundary = camera_infer.camera_batch.build_pipeline(
            build_pipeline_args(args)
        )
    result_path = args.output_dir / args.result_file_name
    for local_index, case in enumerate(pending_cases, start=1):
        print(
            f"case {case['row_index']} ({local_index}/{len(pending_cases)}): "
            f"camera={case['used_camera_key']} source={case['source_absolute_path']}",
            flush=True,
        )
        try:
            report, generated_video = infer_case(args, case, pipeline, boundary)
            reports.append(report)
            successes.append(
                {
                    **case,
                    "output_path": str(generated_video),
                    "heatmap_manifest": str(case_directory(args.output_dir, case) / "manifest.json"),
                    "skipped_complete": False,
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"case {case['row_index']}: FAILED: {error}", flush=True)
            traceback.print_exc()
            failures.append({**case, "error": error})
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = build_result(
            args, all_cases, selected_cases, successes, failures
        )
        write_json(result_path, result)

    result = build_result(args, all_cases, selected_cases, successes, failures)
    write_json(result_path, result)
    if not args.defer_root_manifest:
        ordered_reports = sorted(reports, key=lambda item: int(item["case_id"]))
        write_json(
            args.output_dir / "manifest.json",
            {
                "checkpoint": str(args.checkpoint_path),
                "diagnostic_sigma": args.diagnostic_sigma,
                "mse_used": False,
                "heatmap_output_format": args.heatmap_output_format,
                "case_ids": [int(report["case_id"]) for report in ordered_reports],
                "cases": ordered_reports,
                "rendering": heatmap_rendering_config(args),
            },
        )
    if failures:
        raise RuntimeError(
            f"generated {len(successes)} / {len(selected_cases)} cases with "
            f"{len(failures)} failures"
        )
    return result


def main() -> int:
    args = parse_args()
    if args.sample_count <= 0 or args.start_index < 0:
        raise ValueError("sample_count must be positive and start_index nonnegative")
    if args.frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
