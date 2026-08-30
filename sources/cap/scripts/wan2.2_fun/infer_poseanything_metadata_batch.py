#!/usr/bin/env python3
"""Generate a deterministic PoseAnything metadata prefix on one GPU shard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
import types
from pathlib import Path
from typing import Any

import infer_cap_arm_sample as single


DEFAULT_NEGATIVE_PROMPT = (
    "Blurring, mutation, deformation, distortion, dark, static, text, subtitles, "
    "comic, line art, low quality, worst quality, malformed body."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(os.environ.get("MODEL_CACHE_ROOT", "outputs/model-cache")),
    )
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_records(path: Path, count: int) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"metadata must be a JSON array: {path}")
    if len(records) < count:
        raise ValueError(f"metadata has {len(records)} entries; requested {count}")
    prefix = records[:count]
    if not all(isinstance(record, dict) for record in prefix):
        raise ValueError("every metadata entry must be an object")
    return prefix


def record_path(record: dict[str, Any], keys: tuple[str, ...]) -> Path:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value:
            path = Path(value).expanduser().resolve()
            if not path.is_file() or path.stat().st_size <= 0:
                raise FileNotFoundError(f"metadata path is missing or empty: {path}")
            return path
    raise ValueError(f"metadata entry lacks all required path keys: {keys}")


def frame_indices(
    record: dict[str, Any], target_frames: int, video_frames: int, skeleton_frames: int
) -> list[int]:
    import numpy as np

    explicit = single.first_value(record, ("sampling.frame_indices", "frame_indices"))
    if explicit is not None:
        aligned = np.asarray(explicit, dtype=np.int64).reshape(-1)
        if len(aligned) < target_frames:
            raise ValueError(
                f"metadata provides only {len(aligned)} aligned frames; need {target_frames}"
            )
        positions = np.linspace(0, len(aligned) - 1, target_frames, dtype=np.int64)
        indices = aligned[positions]
    else:
        start = int(single.first_value(record, ("sampling.start_frame", "start_frame"), 0))
        stride = max(
            int(single.first_value(record, ("sampling.stride", "video_sample_stride"), 1)),
            1,
        )
        indices = start + np.arange(target_frames, dtype=np.int64) * stride
    if len(indices) != target_frames:
        raise ValueError(f"resolved {len(indices)} frames; expected {target_frames}")
    if indices.min() < 0 or indices.max() >= min(video_frames, skeleton_frames):
        raise IndexError(
            "aligned frame indices exceed RGB/skeleton lengths: "
            f"range=[{indices.min()}, {indices.max()}] rgb={video_frames} "
            f"skeleton={skeleton_frames}"
        )
    return indices.tolist()


def read_resized_frames(reader: Any, indices: list[int], height: int, width: int):
    import numpy as np
    from PIL import Image

    decoded = reader.get_batch(np.asarray(indices, dtype=np.int64)).asnumpy()
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            )
            for frame in decoded
        ]
    )


def extract_sample(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    video_path = record_path(record, ("file_path", "video_path"))
    skeleton_path = record_path(
        record, ("skeleton_video_path", "control_file_path")
    )
    video_reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    skeleton_reader = VideoReader(str(skeleton_path), ctx=cpu(0), num_threads=2)
    indices = frame_indices(
        record, args.frames, len(video_reader), len(skeleton_reader)
    )
    target_frames = read_resized_frames(
        video_reader, indices, args.height, args.width
    )
    skeleton_frames = read_resized_frames(
        skeleton_reader, indices, args.height, args.width
    )
    skeleton_tensor = (
        torch.from_numpy(skeleton_frames.copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )
    prompt = str(
        single.first_value(record, ("text", "prompt.text", "prompt", "caption"), "")
    )
    return {
        "video_path": video_path,
        "skeleton_path": skeleton_path,
        "frame_indices": indices,
        "target_frames": target_frames,
        "skeleton_frames": skeleton_frames,
        "skeleton_tensor": skeleton_tensor,
        "first_frame": Image.fromarray(target_frames[0]),
        "prompt": prompt,
    }


def cache_directory(args: argparse.Namespace) -> Path:
    checkpoint = args.checkpoint.resolve()
    return (
        args.cache_root.expanduser().resolve()
        / "poseanything"
        / checkpoint.parent.name
        / checkpoint.name
        / "transformer"
    )


def install_pose_condition_patch(transformer: Any) -> dict[str, int]:
    """Keep only the 48-channel skeleton condition seen during training."""

    patch_channels = int(transformer.patch_embedding.in_channels)
    if patch_channels != 96:
        raise RuntimeError(
            f"PoseAnything checkpoint must use 96 patch channels; got {patch_channels}"
        )
    # Accelerate's CPU-offload hook owns ``forward`` and may restore it after
    # each pipeline call. Patch the persistent callable behind that hook so
    # conditioning stays correct across multiple samples in one process.
    forward_attribute = (
        "_old_forward" if hasattr(transformer, "_old_forward") else "forward"
    )
    original_forward = getattr(transformer, forward_attribute)
    state = {"calls": 0, "trimmed_auxiliary_calls": 0}

    def pose_forward(self: Any, *positional: Any, **kwargs: Any):
        x = kwargs.get("x", positional[0] if positional else None)
        y = kwargs.get("y")
        if x is None:
            raise RuntimeError("PoseAnything transformer forward received no x tensor")
        if y is not None:
            expected_y_channels = patch_channels - int(x.shape[1])
            if int(y.shape[1]) == expected_y_channels:
                pass
            elif int(y.shape[1]) - expected_y_channels in (48, 52):
                # The stock control pipeline appends either an empty 48-channel
                # start latent or the 52-channel TI2V mask/reference condition.
                # PoseAnything was trained with only the skeleton latent in y;
                # the reference frame is preserved directly in x by the mask.
                kwargs["y"] = y[:, :expected_y_channels]
                state["trimmed_auxiliary_calls"] += 1
            else:
                raise RuntimeError(
                    "PoseAnything conditioning channel mismatch: "
                    f"x={x.shape[1]} y={y.shape[1]} patch={patch_channels}"
                )
        state["calls"] += 1
        return original_forward(*positional, **kwargs)

    setattr(
        transformer,
        forward_attribute,
        types.MethodType(pose_forward, transformer),
    )
    return state


def build_pipeline(args: argparse.Namespace):
    args.variant = "poseanything"
    args.architecture_mode = "poseanything"
    args.arm_action_dim = 7
    args.arm_action_num_frames = args.frames
    cache_dir = cache_directory(args)
    pipeline, config, device = single.build_pipeline(
        args, args.checkpoint.resolve(), cache_dir
    )
    patch_state = install_pose_condition_patch(pipeline.transformer)
    return pipeline, config, device, cache_dir, patch_state


def generate_sample(
    args: argparse.Namespace,
    sample: dict[str, Any],
    pipeline: Any,
    config: Any,
    device: Any,
):
    import torch
    from videox_fun.utils.utils import get_image_to_video_latent

    generator = torch.Generator(device=device).manual_seed(args.generation_seed + args.sample_id)
    pipeline.transformer._current_action_map_mask = torch.ones(1, dtype=torch.float32)
    inpaint_video, inpaint_mask, _ = get_image_to_video_latent(
        [sample["first_frame"]],
        None,
        video_length=args.frames,
        sample_size=[args.height, args.width],
    )
    expected_video_shape = (1, 3, args.frames, args.height, args.width)
    expected_mask_shape = (1, 1, args.frames, args.height, args.width)
    if tuple(inpaint_video.shape) != expected_video_shape:
        raise RuntimeError(
            f"invalid first-frame video shape: expected={expected_video_shape} "
            f"actual={tuple(inpaint_video.shape)}"
        )
    if tuple(inpaint_mask.shape) != expected_mask_shape:
        raise RuntimeError(
            f"invalid first-frame mask shape: expected={expected_mask_shape} "
            f"actual={tuple(inpaint_mask.shape)}"
        )
    if torch.count_nonzero(inpaint_mask[:, :, :1]).item() != 0:
        raise RuntimeError("PoseAnything frame-0 mask must preserve the RGB reference")
    if torch.count_nonzero(inpaint_mask[:, :, 1:] != 255).item() != 0:
        raise RuntimeError("PoseAnything frames 1..N must remain generated")
    with torch.inference_mode():
        generated = pipeline(
            sample["prompt"],
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            video=inpaint_video,
            mask_video=inpaint_mask,
            control_video=sample["skeleton_tensor"],
            arm_action=None,
            arm_action_mask=None,
            num_frames=args.frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            boundary=float(config["transformer_additional_kwargs"].get("boundary", 0.9)),
            shift=int(config["scheduler_kwargs"].get("shift", 5)),
            use_empty_control_latents=False,
        ).videos.detach().cpu()
    return generated


def publish_sample(
    args: argparse.Namespace,
    sample: dict[str, Any],
    generated: Any,
    cache_dir: Path,
    patch_state: dict[str, int],
    published_dir: Path,
    output_dir: Path,
) -> None:
    import numpy as np
    import torch
    from videox_fun.utils.utils import save_videos_grid

    output_dir.mkdir(parents=True, exist_ok=False)
    generated_path = output_dir / "generated.mp4"
    target_path = output_dir / "target_clip.mp4"
    condition_path = output_dir / "skeleton_condition.mp4"
    first_frame_path = output_dir / "first_frame.png"
    save_videos_grid(generated, str(generated_path), fps=args.fps)
    target = (
        torch.from_numpy(np.asarray(sample["target_frames"]).copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )
    condition = (
        torch.from_numpy(np.asarray(sample["skeleton_frames"]).copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )
    save_videos_grid(target, str(target_path), fps=args.fps)
    save_videos_grid(condition, str(condition_path), fps=args.fps)
    sample["first_frame"].save(first_frame_path)
    for video_path in (generated_path, target_path, condition_path):
        probe_video(video_path, args.width, args.height, args.frames)
    write_json_atomic(
        output_dir / "manifest.json",
        {
            "metadata_index": args.sample_id,
            "checkpoint": str(args.checkpoint.resolve()),
            "cache": str(cache_dir),
            "video_path": str(sample["video_path"]),
            "skeleton_path": str(sample["skeleton_path"]),
            "frame_indices": sample["frame_indices"],
            "prompt": sample["prompt"],
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "frames": args.frames,
            "fps": args.fps,
            "seed": args.generation_seed + args.sample_id,
            "generation_seed_base": args.generation_seed,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "conditioning": "poseanything_skeleton_latent",
            "reference_frame_conditioning": "clean_target_rgb_latent_in_x",
            "transformer_patch_channels": 96,
            "pipeline_forward_calls": patch_state["calls"],
            "trimmed_auxiliary_condition_calls": patch_state[
                "trimmed_auxiliary_calls"
            ],
            "generated": str(published_dir / generated_path.name),
            "target_clip": str(published_dir / target_path.name),
            "skeleton_condition": str(published_dir / condition_path.name),
            "first_frame": str(published_dir / first_frame_path.name),
        },
    )


def sample_complete(output_dir: Path, metadata_index: int, checkpoint: Path) -> bool:
    required = (
        "generated.mp4",
        "target_clip.mp4",
        "skeleton_condition.mp4",
        "first_frame.png",
        "manifest.json",
    )
    if not all(
        (output_dir / name).is_file() and (output_dir / name).stat().st_size > 0
        for name in required
    ):
        return False
    try:
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("metadata_index") == metadata_index
        and manifest.get("checkpoint") == str(checkpoint)
        and manifest.get("conditioning") == "poseanything_skeleton_latent"
        and manifest.get("reference_frame_conditioning")
        == "clean_target_rgb_latent_in_x"
        and manifest.get("transformer_patch_channels") == 96
        and manifest.get("trimmed_auxiliary_condition_calls", 0) > 0
    )


def probe_video(path: Path, width: int, height: int, frames: int) -> dict[str, Any]:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    actual_frames = len(reader)
    if actual_frames != frames:
        raise RuntimeError(
            f"unexpected video frame count for {path}: "
            f"expected={frames} actual={actual_frames}"
        )
    first_frame = reader[0].asnumpy()
    actual_height, actual_width = first_frame.shape[:2]
    if actual_width != width or actual_height != height:
        raise RuntimeError(
            f"unexpected video dimensions for {path}: "
            f"expected={width}x{height} actual={actual_width}x{actual_height}"
        )
    return {
        "width": actual_width,
        "height": actual_height,
        "nb_frames": actual_frames,
    }


def verify_outputs(args: argparse.Namespace) -> int:
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reports = []
    for index in range(args.sample_count):
        case_dir = output_dir / f"sample-{index:05d}"
        if not sample_complete(case_dir, index, checkpoint):
            raise RuntimeError(f"missing or inconsistent output for metadata index {index}")
        probes = {
            name: probe_video(case_dir / name, args.width, args.height, args.frames)
            for name in ("generated.mp4", "target_clip.mp4", "skeleton_condition.mp4")
        }
        reports.append({"metadata_index": index, "videos": probes})
    report = {
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "verified_cases": len(reports),
        "expected_cases": args.sample_count,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "cases": reports,
        "verified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(output_dir / "verification.json", report)
    print(f"verified {len(reports)}/{args.sample_count} PoseAnything cases: {output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.frames != 17:
        raise ValueError("PoseAnything checkpoint inference requires exactly 17 frames")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("--height and --width must be positive multiples of 32")
    if not 0 <= args.rank < args.world_size or args.world_size <= 0:
        raise ValueError("--rank must be in [0, world-size)")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.status_dir = args.status_dir.expanduser().resolve()
    if args.verify_only:
        return verify_outputs(args)

    for path, label in (
        (args.metadata, "metadata"),
        (args.checkpoint, "checkpoint"),
        (args.model_root, "model root"),
        (args.config, "config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")
    records = load_records(args.metadata.expanduser().resolve(), args.sample_count)
    assigned = list(range(args.rank, args.sample_count, args.world_size))
    pending = [
        index
        for index in assigned
        if not sample_complete(
            args.output_dir / f"sample-{index:05d}", index, args.checkpoint
        )
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    pipeline = config = device = cache_dir = patch_state = None
    if pending:
        pipeline, config, device, cache_dir, patch_state = build_pipeline(args)
    write_json_atomic(
        args.status_dir / f"worker-{args.rank}-ready.json",
        {
            "rank": args.rank,
            "world_size": args.world_size,
            "assigned": assigned,
            "pending": pending,
            "checkpoint": str(args.checkpoint),
            "pipeline_loaded": pipeline is not None,
            "ready_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    print(
        f"rank={args.rank}/{args.world_size} ready assigned={assigned} pending={pending}",
        flush=True,
    )

    successes: list[int] = []
    skipped: list[int] = []
    failures: list[dict[str, Any]] = []
    for index in assigned:
        final_dir = args.output_dir / f"sample-{index:05d}"
        if sample_complete(final_dir, index, args.checkpoint):
            skipped.append(index)
            print(f"rank={args.rank} skip complete metadata_index={index}", flush=True)
            continue
        temporary_dir = args.output_dir / f".sample-{index:05d}.tmp-r{args.rank}-{os.getpid()}"
        try:
            if temporary_dir.exists():
                raise FileExistsError(f"temporary output already exists: {temporary_dir}")
            args.sample_id = index
            sample = extract_sample(records[index], args)
            generated = generate_sample(args, sample, pipeline, config, device)
            publish_sample(
                args,
                sample,
                generated,
                cache_dir,
                patch_state,
                final_dir,
                temporary_dir,
            )
            if final_dir.exists():
                raise FileExistsError(f"incomplete final output blocks publish: {final_dir}")
            os.replace(temporary_dir, final_dir)
            successes.append(index)
            print(f"rank={args.rank} complete metadata_index={index}", flush=True)
        except Exception as error:
            traceback.print_exc()
            shutil.rmtree(temporary_dir, ignore_errors=True)
            failures.append(
                {
                    "metadata_index": index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )

    status = {
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned": assigned,
        "successes": successes,
        "skipped": skipped,
        "failures": failures,
        "checkpoint": str(args.checkpoint),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json_atomic(args.status_dir / f"worker-{args.rank}-status.json", status)
    if failures:
        raise RuntimeError(f"rank {args.rank} failed {len(failures)} metadata samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
