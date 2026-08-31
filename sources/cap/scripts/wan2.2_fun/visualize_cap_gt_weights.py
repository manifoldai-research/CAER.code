#!/usr/bin/env python3
"""Render CAP S/E diagnostic weights directly over randomly selected GT clips."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

import arm_mse_heatmap as weight_viz
import infer_cap_arm_sample as single
from visualize_cap_arm_weights import RENDERING_CONFIG, WEIGHT_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("poseanything", "libero"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path(os.environ.get("POSE_BASE_MODEL", "models/Wan2.2-TI2V-5B")))
    parser.add_argument("--config", type=Path, default=single.DEFAULT_CONFIG)
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("MODEL_CACHE_ROOT", "outputs/model-cache")))
    parser.add_argument("--transformer-cache", type=Path)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-seed", type=int, default=20260723)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--diagnostic-sigma", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_records(path: Path, count: int, seed: int) -> tuple[list[int], list[dict[str, Any]]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or count <= 0 or count > len(records):
        raise ValueError(f"invalid metadata/sample count: records={len(records)} count={count}")
    ids = random.Random(seed).sample(range(len(records)), count)
    return ids, [records[index] for index in ids]


def _read_video(path: Path, indices: np.ndarray, height: int, width: int) -> np.ndarray:
    from decord import VideoReader, cpu

    reader = VideoReader(str(path), ctx=cpu(0), num_threads=2)
    indices = np.clip(indices, 0, len(reader) - 1)
    frames = reader.get_batch(indices).asnumpy()
    return np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            )
            for frame in frames
        ]
    )


def extract_case(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    video_path = Path(record.get("file_path") or record["video_path"])
    if args.mode == "poseanything":
        aligned = np.asarray(record["sampling"]["frame_indices"], dtype=np.int64)
        positions = np.linspace(0, len(aligned) - 1, args.frames, dtype=np.int64)
        indices = aligned[positions]
        control_path = Path(
            record.get("skeleton_video_path") or record["control_file_path"]
        )
        frames = _read_video(video_path, indices, args.height, args.width)
        skeleton = _read_video(control_path, indices, args.height, args.width)
        action = None
    else:
        start = int(record.get("start_frame", 0))
        stride = int(record.get("video_sample_stride", 1))
        indices = start + np.arange(args.frames, dtype=np.int64) * stride
        frames = _read_video(video_path, indices, args.height, args.width)
        annotation_path = Path(record["ann_file"])
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        key = record.get("arm_action_key", "state")
        values = np.asarray(annotation[key], dtype=np.float32)
        action_indices = np.clip(indices, 0, len(values) - 1)
        action = values[action_indices, :7]
        skeleton = None
        control_path = annotation_path
    return {
        "frames": frames,
        "skeleton": skeleton,
        "action": action,
        "video_path": video_path,
        "control_path": control_path,
        "indices": indices.tolist(),
        "prompt": str(record.get("text", "")),
        "case_name": str(record.get("episode_id") or record.get("source", {}).get("sample_name") or video_path.stem),
    }


def _architecture_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    return single.transformer_kwargs(
        config,
        architecture_mode="poseanything" if args.mode == "poseanything" else "arm",
        arm_action_dim=7,
        arm_action_num_frames=args.frames,
    )


def _instantiate_transformer(args: argparse.Namespace):
    from accelerate import init_empty_weights
    from videox_fun.models import Wan2_2Transformer3DModel

    model_config = json.loads((args.model_root / "config.json").read_text())
    kwargs = _architecture_kwargs(args)
    mapping = kwargs.pop("dict_mapping", {})
    kwargs.pop("cap_expected_patch_embedding_source_channels", None)
    for source, target in mapping.items():
        if target not in kwargs:
            kwargs[target] = model_config[source]
    with init_empty_weights(include_buffers=True):
        model = Wan2_2Transformer3DModel.from_config(model_config, **kwargs)
    model.to_empty(device="cpu")
    return model


def load_transformer(args: argparse.Namespace):
    import torch
    from videox_fun.models import Wan2_2Transformer3DModel

    checkpoint = args.checkpoint.resolve()
    cache_dir = (
        args.transformer_cache.resolve()
        if args.transformer_cache is not None
        else args.cache_root / args.mode / checkpoint.parent.name / checkpoint.name / "transformer"
    )
    if args.mode == "poseanything":
        loader_args = SimpleNamespace(
            model_root=args.model_root.resolve(),
            cache_root=args.cache_root.resolve(),
            variant=f"poseanything-{checkpoint.parent.name}",
            force_rebuild_cache=False,
            architecture_mode="poseanything",
            arm_action_dim=7,
            arm_action_num_frames=args.frames,
        )
        from omegaconf import OmegaConf

        return single.load_or_build_transformer(
            loader_args, checkpoint, cache_dir, OmegaConf.load(args.config)
        )

    from safetensors.torch import load_file

    model = _instantiate_transformer(args)
    local_state = args.transformer_cache if args.transformer_cache is not None and args.transformer_cache.is_file() else checkpoint / "diffusion_pytorch_model.safetensors"
    state = load_file(str(local_state), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint restore mismatch: missing={missing} unexpected={unexpected}")
    # `freqs` is a plain tensor (not a registered buffer), so init-empty
    # construction leaves it on meta; rebuild it on CPU before moving the model.
    if getattr(model.freqs, "is_meta", False):
        model.freqs = torch.empty(0, device="cpu")
        model.disable_riflex()
    return model.to(dtype=torch.bfloat16).eval().requires_grad_(False)


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


def _video_tensor(frames: np.ndarray):
    import torch

    return torch.from_numpy(frames.copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0


def diagnose_case(args: argparse.Namespace, pipeline, case: dict[str, Any], sample_id: int):
    import torch

    device = torch.device(args.device)
    clean = weight_viz.encode_video_to_latents(pipeline, _video_tensor(case["frames"]), device)
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
        clean.shape[2] * clean.shape[3] * clean.shape[4]
        / (pipeline.transformer.config.patch_size[1] * pipeline.transformer.config.patch_size[2])
    )
    timestep = torch.full((1,), sigma * 1000.0, device=device, dtype=torch.float32)
    if args.mode == "poseanything":
        skeleton = weight_viz.encode_video_to_latents(
            pipeline, _video_tensor(case["skeleton"]), device
        )
        black = torch.zeros_like(_video_tensor(case["skeleton"]))
        black_latent = weight_viz.encode_video_to_latents(pipeline, black, device)
        y_cond, y_null = skeleton, black_latent
        arm_action = arm_mask = None
        pipeline.transformer._current_action_map_mask = torch.ones(1, device=device)
    else:
        mask = torch.ones((1, 4, *clean.shape[2:]), device=device, dtype=clean.dtype)
        mask[:, :, :1] = 0
        reference = torch.zeros_like(clean)
        reference[:, :, :1] = clean[:, :, :1]
        y_cond = y_null = torch.cat([mask, reference], dim=1)
        arm_action = torch.from_numpy(case["action"]).unsqueeze(0).to(device=device, dtype=clean.dtype)
        arm_mask = torch.ones(1, device=device)

    kwargs = dict(
        x=noisy,
        context=context,
        t=timestep,
        seq_len=seq_len,
        y=y_cond,
        y_camera=None,
        y_camera_mask=None,
        arm_action=arm_action,
        arm_action_mask=arm_mask,
        full_ref=None,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = pipeline.transformer(**kwargs)
        null_kwargs = dict(kwargs)
        null_kwargs["y"] = y_null
        if arm_mask is not None:
            null_kwargs["arm_action_mask"] = torch.zeros_like(arm_mask)
        if args.mode == "poseanything":
            pipeline.transformer._current_action_map_mask = torch.zeros(1, device=device)
        null_prediction = pipeline.transformer(**null_kwargs)
    effect = torch.linalg.vector_norm(
        prediction.float() - null_prediction.float(), ord=2, dim=1, keepdim=True
    )
    rho = weight_viz.compute_rho_maps(
        prediction, target, effect, WEIGHT_MODES, exclude_first_frame=True
    )
    return {mode: values.detach().cpu() for mode, values in rho.items()}


def export_case(args, case, sample_id: int, rho: dict[str, Any]) -> dict[str, Any]:
    case_dir = args.output_dir / f"case_{sample_id:06d}_{case['case_name']}"
    case_dir.mkdir(parents=True, exist_ok=True)
    selected = list(range(args.frame_stride, args.frames, args.frame_stride))
    if selected[-1] != args.frames - 1:
        selected.append(args.frames - 1)
    report_weights = {}
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
        npz_path = case_dir / f"{name}_weights.npz"
        np.savez_compressed(npz_path, weights=raw)
        mode_dir = case_dir / name
        mode_dir.mkdir(exist_ok=True)
        pngs = []
        for index in selected:
            overlay = weight_viz.overlay_weight_response(
                case["frames"][index], response[index], blur_radius=12.0
            )
            path = mode_dir / f"frame_{index:04d}.png"
            Image.fromarray(overlay).save(path, compress_level=2)
            pngs.append(str(path))
        report_weights[mode] = {
            "name": name,
            "quantity": "S / mean(S)" if mode == "CAER" else "MSE",
            "array": str(npz_path),
            "normalization": {"vmin": vmin, "vmax": vmax, "percentile": 99.0},
            "pngs": pngs,
        }
    report = {
        "sample_id": sample_id,
        "case_name": case["case_name"],
        "video_path": str(case["video_path"]),
        "condition_path": str(case["control_path"]),
        "frame_indices": case["indices"],
        "diagnostic_sigma": args.diagnostic_sigma,
        "background": "ground_truth",
        **RENDERING_CONFIG,
        "weights": report_weights,
    }
    (case_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.metadata = args.metadata.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ids, records = load_records(args.metadata, args.sample_count, args.sample_seed)
    (args.output_dir / "selection.json").write_text(
        json.dumps({"sample_seed": args.sample_seed, "sample_ids": ids}, indent=2) + "\n"
    )
    transformer = load_transformer(args)
    pipeline = build_pipeline(args, transformer)
    reports = []
    for position, (sample_id, record) in enumerate(zip(ids, records), 1):
        case = extract_case(record, args)
        rho = diagnose_case(args, pipeline, case, sample_id)
        reports.append(export_case(args, case, sample_id, rho))
        print(f"completed {position}/{len(ids)} sample_id={sample_id}", flush=True)
    root = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "metadata": str(args.metadata),
        "sample_ids": ids,
        "cases": reports,
        "rendering": RENDERING_CONFIG,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(root, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
