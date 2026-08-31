#!/usr/bin/env python3
"""Generate official WorldArena Arm rollouts from a CAP FSDP DCP checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import arm_mse_heatmap as mse_heatmap
import infer_cap_arm_sample as single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=single.VARIANTS, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--model-root", type=Path, default=single.DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=single.DEFAULT_CACHE_ROOT)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--action-downsample", type=int, default=3)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument(
        "--mse-heatmap-weights",
        type=mse_heatmap.parse_weight_selection,
        default=(),
        metavar="SELECTION",
        help=(
            "none, all, or a comma-separated subset of "
            + ",".join(mse_heatmap.WEIGHT_MODES)
        ),
    )
    parser.add_argument("--mse-heatmap-sigma", type=float, default=0.5)
    parser.add_argument("--mse-heatmap-eps", type=float, default=1e-6)
    parser.add_argument("--mse-heatmap-scale", type=int, default=8)
    parser.add_argument("--negative-prompt", default=single.DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def action_chunk(action, start: int, frames: int):
    import numpy as np

    chunk = np.asarray(action[start : start + frames], dtype=np.float32)
    if len(chunk) == 0:
        raise ValueError("empty action chunk")
    if len(chunk) < frames:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], frames - len(chunk), axis=0)])
    return chunk


def sample_to_frame(sample):
    import torch
    from PIL import Image

    frame = sample[0, :, -1].detach().float().cpu().permute(1, 2, 0).clamp(0, 1)
    return Image.fromarray((frame * 255).round().to(torch.uint8).numpy())


def heatmap_outputs_complete(record: dict[str, Any], modes: tuple[str, ...]) -> bool:
    if not modes:
        return True
    outputs = record.get("mse_heatmaps")
    if not isinstance(outputs, dict):
        return False
    for mode in modes:
        item = outputs.get(mode)
        if not isinstance(item, dict):
            return False
        for key in ("video", "array"):
            value = item.get(key)
            if not value:
                return False
            path = Path(value)
            if not path.is_file() or path.stat().st_size <= 0:
                return False
    return True


def open_ground_truth(case: dict[str, Any], frame_count: int, action_downsample: int):
    import numpy as np
    from decord import VideoReader, cpu

    reader = VideoReader(case["gt_video_path"], ctx=cpu(0), num_threads=2)
    total = len(reader)
    if total <= 0:
        raise ValueError(f"empty ground-truth video for {case['episode']}: {case['gt_video_path']}")
    strided = np.arange(0, total, action_downsample, dtype=np.int64)
    if len(strided) >= frame_count:
        indices = strided[:frame_count]
        alignment = f"source_frame_stride_{action_downsample}"
    else:
        indices = np.linspace(0, total - 1, frame_count).round().astype(np.int64)
        alignment = "uniform_resample_to_downsampled_action_length"
    return reader, indices, alignment, total


def load_ground_truth_chunk(reader, indices, start: int, frames: int, height: int, width: int):
    import numpy as np
    import torch
    from PIL import Image

    chunk_indices = np.asarray(indices[start : start + frames], dtype=np.int64)
    if len(chunk_indices) == 0:
        raise ValueError(f"empty ground-truth chunk at frame {start}")
    if len(chunk_indices) < frames:
        chunk_indices = np.concatenate(
            [chunk_indices, np.repeat(chunk_indices[-1:], frames - len(chunk_indices))]
        )
    decoded = reader.get_batch(chunk_indices).asnumpy()
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize((width, height), Image.Resampling.BILINEAR)
            )
            for frame in decoded
        ]
    )
    return (
        torch.from_numpy(resized.copy())
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .float()
        .div_(255.0)
    )


def save_case_heatmaps(
    args,
    episode: str,
    pixel_mse_chunks: list[Any],
    rho_chunks: dict[str, list[Any]],
    generated_frames: int,
):
    import numpy as np

    if not pixel_mse_chunks:
        raise RuntimeError(f"no pixel MSE heatmap chunks were captured for {episode}")
    pixel_mse = np.concatenate(pixel_mse_chunks, axis=0).astype(np.float32, copy=False)
    if len(pixel_mse) != generated_frames:
        raise RuntimeError(
            f"pixel MSE heatmap/video length mismatch for {episode}: "
            f"heatmap={len(pixel_mse)} generated={generated_frames}"
        )
    outputs = {}
    for mode in args.mse_heatmap_weights:
        if mode == "MSE":
            values = pixel_mse
            quantity = "pixel RGB MSE"
        else:
            if not rho_chunks.get(mode):
                raise RuntimeError(f"no {mode} rho chunks were captured for {episode}")
            rho = np.concatenate(rho_chunks[mode], axis=0).astype(np.float32, copy=False)
            if rho.shape != pixel_mse.shape:
                raise RuntimeError(
                    f"{mode} rho/pixel shape mismatch for {episode}: "
                    f"rho={rho.shape} pixel={pixel_mse.shape}"
                )
            values = pixel_mse * np.maximum(rho, 0.0)
            quantity = "rho * pixel RGB MSE (rho from latent Method1 diagnosis)"
        name = mse_heatmap.output_name(mode)
        mode_dir = args.output_dir / "mse_heatmaps" / name
        video_path = mse_heatmap.save_heatmap_video(
            values, mode_dir / f"{episode}.mp4", args.fps
        )
        array_path = mse_heatmap.save_heatmap_array(values, mode_dir / f"{episode}.npz")
        outputs[mode] = {
            "quantity": quantity,
            "output_name": name,
            "video": str(video_path),
            "array": str(array_path),
            "stats": mse_heatmap.heatmap_stats(values),
        }
    return outputs


def generate_case(args, case: dict[str, Any], checkpoint, cache_dir, pipeline, config, device):
    import numpy as np
    import torch
    from PIL import Image
    from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

    output_path = args.output_dir / f"{case['episode']}.mp4"
    record_path = args.output_dir / "manifests" / f"{case['episode']}.json"
    expected = {
        "episode": case["episode"],
        "checkpoint": str(checkpoint),
        "action_downsample": args.action_downsample,
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "generation_seed_base": args.generation_seed,
    }
    if args.mse_heatmap_weights:
        expected.update(
            {
                "mse_heatmap_weights": list(args.mse_heatmap_weights),
                "mse_heatmap_sigma": args.mse_heatmap_sigma,
                "mse_heatmap_eps": args.mse_heatmap_eps,
                "mse_heatmap_scale": args.mse_heatmap_scale,
                "mse_heatmap_domain": (
                    "decoded RGB pixel MSE against GT; weighted modes multiply by latent Method1 rho"
                ),
                "mse_heatmap_resolution": {
                    "width": int(args.width),
                    "height": int(args.height),
                },
            }
        )
    if output_path.is_file() and output_path.stat().st_size > 0 and record_path.is_file():
        old = json.loads(record_path.read_text(encoding="utf-8"))
        old_modes = tuple(old.get("mse_heatmap_weights") or ())
        modes_match = old_modes == args.mse_heatmap_weights
        if (
            all(old.get(key) == value for key, value in expected.items())
            and modes_match
            and heatmap_outputs_complete(old, args.mse_heatmap_weights)
        ):
            print(f"rank={args.rank} skip complete {case['episode']}", flush=True)
            return old

    action = np.load(case["action_path"]).astype(np.float32)
    action = action[:: args.action_downsample]
    if action.ndim != 2 or action.shape[1] != 14 or len(action) < 2:
        raise ValueError(f"invalid downsampled action for {case['episode']}: {action.shape}")
    current_frame = Image.open(case["first_frame_path"]).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.BILINEAR
    )
    chunks = []
    pixel_mse_chunks = []
    rho_chunks = {mode: [] for mode in args.mse_heatmap_weights if mode != "MSE"}
    needs_latent_diagnostic = any(mode != "MSE" for mode in args.mse_heatmap_weights)
    gt_reader = None
    gt_indices = None
    gt_alignment = None
    gt_source_frames = None
    if args.mse_heatmap_weights:
        gt_reader, gt_indices, gt_alignment, gt_source_frames = open_ground_truth(
            case, len(action), args.action_downsample
        )
    stride = args.frames - 1
    frame_index = 0
    chunk_index = 0
    while frame_index < len(action) - 1:
        arm = action_chunk(action, frame_index, args.frames)
        video, mask, _ = get_image_to_video_latent(
            [current_frame], None, video_length=args.frames, sample_size=[args.height, args.width]
        )
        seed = args.generation_seed + int(case["episode_id"]) * 1000 + chunk_index
        generator = torch.Generator(device=device).manual_seed(seed)
        target_video = None
        target_latents = None
        capture = None
        if args.mse_heatmap_weights:
            target_video = load_ground_truth_chunk(
                gt_reader,
                gt_indices,
                frame_index,
                args.frames,
                args.height,
                args.width,
            )
            if needs_latent_diagnostic:
                target_latents = mse_heatmap.encode_video_to_latents(
                    pipeline, target_video, device
                )
                capture = mse_heatmap.Method1HeatmapCapture(
                    pipeline,
                    target_latents,
                    args.mse_heatmap_weights,
                    sigma=args.mse_heatmap_sigma,
                    eps=args.mse_heatmap_eps,
                )
        with torch.inference_mode():
            pipeline_kwargs = {
                "negative_prompt": args.negative_prompt,
                "height": args.height,
                "width": args.width,
                "video": video,
                "mask_video": mask,
                "control_video": None,
                "arm_action": torch.from_numpy(arm).unsqueeze(0),
                "arm_action_mask": torch.ones((1,), dtype=torch.float32),
                "num_frames": args.frames,
                "num_inference_steps": args.inference_steps,
                "guidance_scale": args.guidance_scale,
                "generator": generator,
                "boundary": float(config["transformer_additional_kwargs"].get("boundary", 0.9)),
                "shift": int(config["scheduler_kwargs"].get("shift", 5)),
                "use_empty_control_latents": False,
            }
            if capture is None:
                sample = pipeline(case["instruction"], **pipeline_kwargs).videos.cpu()
            else:
                with capture:
                    sample = pipeline(case["instruction"], **pipeline_kwargs).videos.cpu()
        actual = min(args.frames, len(action) - frame_index)
        chunks.append(sample[:, :, :actual] if chunk_index == 0 else sample[:, :, 1:actual])
        if args.mse_heatmap_weights:
            pixel_mse = mse_heatmap.compute_pixel_mse_map(sample, target_video)[0, 0]
            pixel_mse = pixel_mse.detach().cpu().numpy().astype(np.float32, copy=False)
            pixel_mse_chunks.append(
                pixel_mse[:actual] if chunk_index == 0 else pixel_mse[1:actual]
            )
            if capture is not None:
                for mode in args.mse_heatmap_weights:
                    if mode == "MSE":
                        continue
                    rendered = mse_heatmap.render_map_to_video(
                        capture.rho_maps[mode],
                        args.frames,
                        scale=args.mse_heatmap_scale,
                        output_size=(args.height, args.width),
                    )
                    rho_chunks[mode].append(
                        rendered[:actual] if chunk_index == 0 else rendered[1:actual]
                    )
            del target_video, target_latents, capture
        current_frame = sample_to_frame(sample)
        frame_index += stride
        chunk_index += 1

    generated = torch.cat(chunks, dim=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(generated, str(output_path), fps=args.fps)
    heatmap_outputs = save_case_heatmaps(
        args,
        case["episode"],
        pixel_mse_chunks,
        rho_chunks,
        int(generated.shape[2]),
    ) if args.mse_heatmap_weights else {}
    record = {
        **expected,
        "episode_id": int(case["episode_id"]),
        "source_action_frames": int(case["action_frames"]),
        "downsampled_action_frames": int(len(action)),
        "generated_frames": int(generated.shape[2]),
        "chunks": chunk_index,
        "output": str(output_path),
        "gt_video_path": case["gt_video_path"],
        "first_frame_path": case["first_frame_path"],
        "instruction_path": case["instruction_path"],
        "action_preprocessing": "raw values; temporal stride only",
        "mse_heatmap_weights": list(args.mse_heatmap_weights),
        "mse_heatmap_sigma": args.mse_heatmap_sigma if args.mse_heatmap_weights else None,
        "mse_heatmap_eps": args.mse_heatmap_eps if args.mse_heatmap_weights else None,
        "mse_heatmap_scale": args.mse_heatmap_scale if args.mse_heatmap_weights else None,
        "mse_heatmap_domain": (
            "decoded RGB pixel MSE against GT; weighted modes multiply by latent Method1 rho"
            if args.mse_heatmap_weights
            else None
        ),
        "mse_heatmap_resolution": (
            {"width": int(args.width), "height": int(args.height)}
            if args.mse_heatmap_weights
            else None
        ),
        "mse_heatmaps": heatmap_outputs,
        "mse_heatmap_gt_alignment": gt_alignment,
        "mse_heatmap_gt_source_frames": gt_source_frames,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"rank={args.rank} complete {case['episode']} chunks={chunk_index} "
        f"frames={generated.shape[2]} output={output_path}",
        flush=True,
    )
    return record


def main() -> int:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must be in [0, world-size)")
    if args.frames != 17 or args.action_downsample <= 0:
        raise ValueError("--frames must be 17 and --action-downsample must be positive")
    if not 0.0 < args.mse_heatmap_sigma < 1.0:
        raise ValueError("--mse-heatmap-sigma must be in (0,1)")
    if args.mse_heatmap_eps <= 0:
        raise ValueError("--mse-heatmap-eps must be positive")
    if args.mse_heatmap_scale <= 0:
        raise ValueError("--mse-heatmap-scale must be positive")
    manifest = json.loads(args.case_manifest.read_text(encoding="utf-8"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"invalid case manifest: {args.case_manifest}")
    assigned = cases[args.rank :: args.world_size]
    args.output_dir = args.output_dir.expanduser().resolve()
    args.run_dir = args.run_dir.expanduser().resolve()
    _, checkpoint, step = single.resolve_run_and_checkpoint(args)
    if step != args.checkpoint_step:
        raise ValueError("resolved checkpoint step changed")
    cache_dir = single.cache_directory(args, args.run_dir, step)
    print(
        f"rank={args.rank}/{args.world_size} assigned={len(assigned)} checkpoint={checkpoint} "
        f"mse_heatmaps={mse_heatmap.format_weight_selection(args.mse_heatmap_weights)}",
        flush=True,
    )
    if args.dry_run:
        schema = single.checkpoint_schema(checkpoint)
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "checkpoint": str(checkpoint),
                    "checkpoint_step": step,
                    "checkpoint_schema": schema,
                    "assigned_episodes": [case["episode"] for case in assigned],
                    "output_dir": str(args.output_dir),
                    "mse_heatmap_weights": list(args.mse_heatmap_weights),
                    "mse_heatmap_sigma": args.mse_heatmap_sigma,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if not assigned:
        return 0
    pipeline, config, device = single.build_pipeline(args, checkpoint, cache_dir)
    ready_path = args.output_dir / "logs" / f"worker-{args.rank}-ready.json"
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(
        json.dumps(
            {
                "rank": args.rank,
                "world_size": args.world_size,
                "checkpoint": str(checkpoint),
                "assigned": len(assigned),
                "pid": os.getpid(),
                "ready_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"rank={args.rank} pipeline ready assigned={len(assigned)}",
        flush=True,
    )
    successes = []
    failures = []
    for case in assigned:
        try:
            successes.append(
                generate_case(args, case, checkpoint, cache_dir, pipeline, config, device)
            )
        except Exception as exc:
            failures.append({"episode": case.get("episode"), "error": repr(exc)})
            traceback.print_exc()
    status = {
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned": len(assigned),
        "successes": len(successes),
        "failures": failures,
        "mse_heatmap_weights": list(args.mse_heatmap_weights),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    status_path = args.output_dir / "logs" / f"worker-{args.rank}-status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise RuntimeError(f"rank {args.rank} failed {len(failures)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
