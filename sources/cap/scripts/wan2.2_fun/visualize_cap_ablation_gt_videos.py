#!/usr/bin/env python3
"""Render Arm/Camera CAP diagnostic weights over random GT clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
from PIL import Image

import arm_mse_heatmap as weight_viz
import infer_cap_arm_sample as single
from visualize_cap_arm_weights import RENDERING_CONFIG, WEIGHT_MODES


DEFAULT_METADATA = {
    "arm": single.DEFAULT_METADATA,
    "camera": Path(os.environ.get("CAMERA_METADATA", "data/camera/metadata.json")),
}
DEFAULT_METADATA_COUNTS = {"arm": 365831, "camera": 505955}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("arm", "camera"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--metadata-count", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/dev/shm/wan22-ti2v-local"))
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument("--transformer-cache", type=Path)
    parser.add_argument(
        "--input-cache-root",
        type=Path,
        default=Path("/dev/shm/cap-ablation-gt-inputs"),
    )
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260723)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--camera-frame-stride", type=int, default=4)
    parser.add_argument("--diagnostic-sigma", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer += chunk
            if not started:
                buffer = buffer.lstrip()
                if not buffer and not eof:
                    continue
                if not buffer.startswith("["):
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
                    raise ValueError("metadata entries must be JSON objects")
                yield item
                buffer = buffer[end:]
            if eof:
                break
    raise ValueError(f"unterminated metadata JSON array: {path}")


def load_selected_records(path: Path, sample_ids: list[int]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"metadata must be a top-level JSON array: {path}")
    missing = [index for index in sample_ids if index >= len(records)]
    if missing:
        raise IndexError(f"metadata does not contain selected IDs: {missing}")
    return [records[index] for index in sample_ids]


def first_value(record: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    return single.first_value(record, keys, default)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned[:120] or "sample"


def stage_input(source: Path, args: argparse.Namespace, sample_id: int, label: str) -> Path:
    root = args.input_cache_root.resolve() / args.mode
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{sample_id:06d}_{label}{source.suffix}"
    if destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return destination
    temporary = root / f".{destination.name}.tmp-{os.getpid()}"
    shutil.copyfile(source, temporary)
    if temporary.stat().st_size != source.stat().st_size:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"input staging size mismatch: {source}")
    os.replace(temporary, destination)
    return destination


def read_video_frames(path: Path, indices: np.ndarray, height: int, width: int) -> np.ndarray:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    if not len(reader):
        raise RuntimeError(f"video has no frames: {path}")
    indices = np.clip(indices, 0, len(reader) - 1)
    frames = reader.get_batch(indices).asnumpy()
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)
            )
            for frame in frames
        ]
    )


def extract_arm_case(
    record: dict[str, Any], args: argparse.Namespace, sample_id: int
) -> dict[str, Any]:
    source_video, source_condition = single.sample_paths(record)
    local_record = dict(record)
    local_record["file_path"] = str(
        stage_input(source_video, args, sample_id, "video")
    )
    local_record["ann_file"] = str(
        stage_input(source_condition, args, sample_id, "condition")
    )
    sample = single.extract_sample(
        local_record,
        SimpleNamespace(
            frames=args.frames,
            height=args.height,
            width=args.width,
            prompt=None,
        ),
    )
    return {
        "frames": sample["frames_uint8"],
        "action": sample["arm_action"],
        "camera": None,
        "video_path": source_video,
        "condition_path": source_condition,
        "condition_local_path": sample["annotation_path"],
        "indices": sample["frame_indices"],
        "prompt": sample["prompt"],
        "case_name": safe_name(
            str(record.get("episode_id") or record.get("episode") or sample["video_path"].stem)
        ),
        "condition_data": {
            "type": "arm_action",
            "action_key": sample["action_key"],
            "frame_indices": sample["frame_indices"],
            "values": sample["arm_action"][0].numpy().tolist(),
        },
    }


def extract_camera_case(
    record: dict[str, Any], args: argparse.Namespace, sample_id: int
) -> dict[str, Any]:
    from decord import VideoReader, cpu
    from videox_fun.data.dataset_image_video import process_pose_file
    from videox_fun.training.cap_conditioning import pack_camera_condition

    source_video = Path(first_value(record, ("file_path", "video_path")))
    source_condition = Path(
        first_value(record, ("control_file_path", "camera_pose_path"))
    )
    if not source_video.is_file() or not source_condition.is_file():
        raise FileNotFoundError(
            f"Camera sample inputs are missing: video={source_video} pose={source_condition}"
        )
    video_path = stage_input(source_video, args, sample_id, "video")
    condition_path = stage_input(source_condition, args, sample_id, "condition")
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    span = (args.frames - 1) * args.camera_frame_stride + 1
    if len(reader) < span:
        indices = np.linspace(0, len(reader) - 1, args.frames, dtype=np.int64)
    else:
        max_start = len(reader) - span
        start = random.Random(args.sample_seed + sample_id).randint(0, max_start)
        indices = start + np.arange(args.frames, dtype=np.int64) * args.camera_frame_stride
    del reader
    frames = read_video_frames(video_path, indices, args.height, args.width)

    # Match the training dataset: build rays relative to the complete source
    # trajectory first, then select the same temporal indices as the RGB clip.
    camera_all = process_pose_file(
        str(condition_path), width=args.width, height=args.height
    )
    camera_values = np.asarray(camera_all)[indices]
    import torch

    camera = pack_camera_condition(
        torch.from_numpy(camera_values.copy())
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
    )
    del camera_all, camera_values
    prompt = str(
        first_value(
            record,
            (
                "text",
                "caption",
                "comprehensive_narrative_caption",
                "comprehensive_narrative_caption_zh",
            ),
            "",
        )
    )
    return {
        "frames": frames,
        "action": None,
        "camera": camera,
        "video_path": source_video,
        "condition_path": source_condition,
        "condition_local_path": condition_path,
        "indices": indices.tolist(),
        "prompt": prompt,
        "case_name": safe_name(video_path.stem),
        "condition_data": {
            "type": "camera_pose",
            "frame_indices": indices.tolist(),
            "source": str(source_condition),
        },
    }


def default_transformer_cache(args: argparse.Namespace) -> Path:
    run_id = args.checkpoint.parent.name
    checkpoint_name = args.checkpoint.name
    if args.mode == "arm":
        root = Path(os.environ.get("MODEL_CACHE_ROOT", "outputs/model-cache"))
    else:
        root = Path(os.environ.get("CAMERA_MODEL_CACHE_ROOT", "outputs/camera-model-cache"))
    return root / "CAER" / run_id / checkpoint_name / "transformer"


def load_transformer(args: argparse.Namespace):
    from omegaconf import OmegaConf

    cache = (
        args.transformer_cache.resolve()
        if args.transformer_cache is not None
        else default_transformer_cache(args).resolve()
    )
    loader_args = SimpleNamespace(
        variant="CAER",
        model_root=args.model_root.resolve(),
        architecture_mode=args.mode,
        arm_action_dim=14,
        arm_action_num_frames=args.frames,
        force_rebuild_cache=False,
    )
    return single.load_or_build_transformer(
        loader_args, args.checkpoint.resolve(), cache, OmegaConf.load(args.config)
    )


def build_pipeline(args: argparse.Namespace, transformer):
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import AutoTokenizer, AutoencoderKLWan3_8, WanT5EncoderModel
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils.utils import filter_kwargs

    config = OmegaConf.load(args.config)
    root = args.model_root.resolve()
    vae = AutoencoderKLWan3_8.from_pretrained(
        str(root / config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(root / config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(root / config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(
            FlowMatchEulerDiscreteScheduler,
            OmegaConf.to_container(config["scheduler_kwargs"]),
        )
    )
    pipeline = Wan2_2FunControlPipeline(
        transformer=transformer,
        transformer_2=None,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    pipeline.to(torch.device(args.device))
    return pipeline


def video_tensor(frames: np.ndarray):
    import torch

    return (
        torch.from_numpy(frames.copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        / 255.0
    )


def diagnose_case(
    args: argparse.Namespace,
    pipeline,
    case: dict[str, Any],
    sample_id: int,
) -> dict[str, Any]:
    import torch

    device = torch.device(args.device)
    clean = weight_viz.encode_video_to_latents(
        pipeline, video_tensor(case["frames"]), device
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
    timestep = torch.full(
        (1,), sigma * 1000.0, device=device, dtype=torch.float32
    )
    mask = torch.ones((1, 4, *clean.shape[2:]), device=device, dtype=clean.dtype)
    mask[:, :, :1] = 0
    reference = torch.zeros_like(clean)
    reference[:, :, :1] = clean[:, :, :1]
    control = torch.cat([mask, reference], dim=1)

    arm_action = None
    arm_mask = None
    camera = None
    camera_mask = None
    if args.mode == "arm":
        arm_action = case["action"].to(device=device, dtype=clean.dtype)
        arm_mask = torch.ones((1,), device=device, dtype=torch.float32)
    else:
        camera = case["camera"].to(device=device, dtype=clean.dtype)
        camera_mask = torch.ones((1,), device=device, dtype=torch.float32)

    kwargs = dict(
        x=noisy,
        context=context,
        t=timestep,
        seq_len=seq_len,
        y=control,
        y_camera=camera,
        y_camera_mask=camera_mask,
        arm_action=arm_action,
        arm_action_mask=arm_mask,
        full_ref=None,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = pipeline.transformer(**kwargs)
        null_kwargs = dict(kwargs)
        if arm_mask is not None:
            null_kwargs["arm_action_mask"] = torch.zeros_like(arm_mask)
        if camera_mask is not None:
            null_kwargs["y_camera_mask"] = torch.zeros_like(camera_mask)
        null_prediction = pipeline.transformer(**null_kwargs)
    effect = torch.linalg.vector_norm(
        prediction.float() - null_prediction.float(), ord=2, dim=1, keepdim=True
    )
    rho = weight_viz.compute_rho_maps(
        prediction, target, effect, WEIGHT_MODES, exclude_first_frame=True
    )
    result = {mode: values.detach().cpu() for mode, values in rho.items()}
    del clean, noise, noisy, target, context, control, prediction, null_prediction, effect
    return result


def write_video(path: Path, frames, fps: float) -> None:
    import imageio.v2 as imageio

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


def complete_case(case_dir: Path) -> dict[str, Any] | None:
    manifest = case_dir / "manifest.json"
    required = ("CAER.mp4", "MSE.mp4", "original.mp4")
    if not manifest.is_file() or not all(
        (case_dir / name).is_file() and (case_dir / name).stat().st_size > 0
        for name in required
    ):
        return None
    try:
        return json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def export_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    sample_id: int,
    rho: dict[str, Any],
) -> dict[str, Any]:
    case_dir = args.output_dir / f"case_{sample_id:06d}_{case['case_name']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    videos: dict[str, str] = {}
    arrays: dict[str, str] = {}
    normalization: dict[str, Any] = {}
    for mode, name in (("CAER", "CAER"), ("MSE", "MSE")):
        latent = rho[mode][0, 0].numpy().astype(np.float32)
        raw = weight_viz.render_map_to_video(
            rho[mode], args.frames, output_size=(args.height, args.width)
        )
        display = weight_viz.render_map_to_video(
            weight_viz.smooth_latent_spatially(latent, sigma=1.5),
            args.frames,
            output_size=(args.height, args.width),
        )
        vmax = weight_viz.positive_percentile_vmax(
            latent, percentile=99.0, exclude_first_frame=False
        )
        vmin = weight_viz.episode_response_vmin(display, vmax)
        response = weight_viz.normalize_weight_response(display, vmax, vmin=vmin)
        array_path = case_dir / f"{name}_weights.npz"
        np.savez_compressed(array_path, weights=raw)
        video_path = case_dir / f"{name}.mp4"
        write_video(
            video_path,
            (
                weight_viz.overlay_weight_response(
                    case["frames"][index], response[index], blur_radius=12.0
                )
                for index in range(args.frames)
            ),
            args.fps,
        )
        videos[mode] = str(video_path)
        arrays[mode] = str(array_path)
        normalization[mode] = {"vmin": vmin, "vmax": vmax, "percentile": 99.0}

    original_path = case_dir / "original.mp4"
    write_video(original_path, case["frames"], args.fps)
    condition_name = "condition.json" if args.mode == "arm" else "condition.txt"
    condition_output = case_dir / condition_name
    if args.mode == "arm":
        condition_output.write_text(
            json.dumps(case["condition_data"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        shutil.copyfile(case["condition_local_path"], condition_output)

    report = {
        "sample_id": sample_id,
        "mode": args.mode,
        "case_name": case["case_name"],
        "video_path": str(case["video_path"]),
        "condition_path": str(case["condition_path"]),
        "frame_indices": case["indices"],
        "diagnostic_sigma": args.diagnostic_sigma,
        "background": "ground_truth",
        "video_fps": args.fps,
        "video_frame_count": args.frames,
        "rendering": dict(RENDERING_CONFIG),
        "normalization": normalization,
        "videos": videos,
        "weight_arrays": arrays,
        "original_video": str(original_path),
        "condition_data": str(condition_output),
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def validate_args(args: argparse.Namespace) -> None:
    if args.metadata is None:
        args.metadata = DEFAULT_METADATA[args.mode]
    if args.metadata_count is None:
        args.metadata_count = DEFAULT_METADATA_COUNTS[args.mode]
    if args.sample_count <= 0 or args.sample_count > args.metadata_count:
        raise ValueError("sample count must be within the metadata size")
    if args.frames != 17:
        raise ValueError("CAP Arm/Camera checkpoints require exactly 17 frames")
    if args.camera_frame_stride <= 0:
        raise ValueError("camera frame stride must be positive")
    if not 0.0 < args.diagnostic_sigma < 1.0:
        raise ValueError("diagnostic sigma must be in (0, 1)")
    for path, label in (
        (args.checkpoint, "checkpoint"),
        (args.metadata, "metadata"),
        (args.model_root, "model root"),
        (args.config, "config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.checkpoint = args.checkpoint.resolve()
    args.metadata = args.metadata.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = random.Random(args.sample_seed).sample(
        range(args.metadata_count), args.sample_count
    )
    selection = {
        "mode": args.mode,
        "sample_seed": args.sample_seed,
        "metadata_count": args.metadata_count,
        "sample_ids": sample_ids,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    records = load_selected_records(args.metadata, sample_ids)

    pending = []
    reports: list[dict[str, Any]] = []
    for sample_id, record in zip(sample_ids, records):
        name_value = (
            record.get("episode_id")
            or record.get("episode")
            or Path(first_value(record, ("file_path", "video_path"))).stem
        )
        case_dir = args.output_dir / f"case_{sample_id:06d}_{safe_name(str(name_value))}"
        cached = complete_case(case_dir)
        if cached is None:
            pending.append((sample_id, record))
        else:
            reports.append(cached)
            print(f"reused sample_id={sample_id}", flush=True)

    pipeline = None
    if pending:
        transformer = load_transformer(args)
        pipeline = build_pipeline(args, transformer)
    for position, (sample_id, record) in enumerate(pending, 1):
        if args.mode == "arm":
            case = extract_arm_case(record, args, sample_id)
        else:
            case = extract_camera_case(record, args, sample_id)
        rho = diagnose_case(args, pipeline, case, sample_id)
        reports.append(export_case(args, case, sample_id, rho))
        print(
            f"completed {position}/{len(pending)} sample_id={sample_id}", flush=True
        )
        if args.mode == "camera":
            del case["camera"]
        import torch

        torch.cuda.empty_cache()

    report_by_id = {int(report["sample_id"]): report for report in reports}
    root = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "sample_ids": sample_ids,
        "cases": [report_by_id[sample_id] for sample_id in sample_ids],
        "rendering": dict(RENDERING_CONFIG),
        "output": "mp4",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
