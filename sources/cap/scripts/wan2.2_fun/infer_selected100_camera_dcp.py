#!/usr/bin/env python3
"""Run the original Camera-100 benchmark from an FSDP DCP checkpoint."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
HOST_PROJECT_ROOT = Path(
    os.environ.get("CAP_CAMERA_HOST_PROJECT_ROOT", str(PROJECT_ROOT))
).resolve()
HOST_SCRIPT_DIR = HOST_PROJECT_ROOT / "scripts/wan2.2_fun"

# Keep the CAP model package selected even after importing the host benchmark.
sys.path.insert(0, str(PROJECT_ROOT))
import videox_fun  # noqa: F401,E402

sys.path.insert(0, str(HOST_SCRIPT_DIR))
import infer_selected100_camera_benchmark as benchmark  # noqa: E402

from infer_cap_arm_sample import load_or_build_transformer  # noqa: E402
from videox_fun.models import (  # noqa: E402
    AutoencoderKLWan,
    AutoencoderKLWan3_8,
    AutoTokenizer,
    WanT5EncoderModel,
)
from videox_fun.pipeline import Wan2_2FunControlPipeline  # noqa: E402
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler  # noqa: E402
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


def checkpoint_identity(checkpoint: Path) -> tuple[str, str, int]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"FSDP checkpoint directory is missing: {checkpoint}")
    try:
        step = int(checkpoint.name.removeprefix("checkpoint-"))
    except ValueError as error:
        raise ValueError(f"invalid checkpoint directory name: {checkpoint}") from error
    run_dir = checkpoint.parent
    variant = run_dir.parent.name
    if variant not in {"CAER", "MSE"}:
        raise ValueError(f"unsupported Camera checkpoint variant: {variant}")
    return variant, run_dir.name, step


def validate_camera_checkpoint_schema(checkpoint: Path) -> None:
    from torch.distributed.checkpoint import FileSystemReader

    metadata = FileSystemReader(
        str(checkpoint / "pytorch_model_fsdp_0")
    ).read_metadata()
    entries = metadata.state_dict_metadata
    patch = entries.get("model.patch_embedding.weight")
    if patch is None or tuple(patch.size) != (3072, 100, 1, 2, 2):
        actual = None if patch is None else tuple(patch.size)
        raise RuntimeError(f"invalid Camera patch_embedding shape: {actual}")
    if "model.control_adapter.output_conv.weight" not in entries:
        raise RuntimeError("Camera checkpoint lacks the control adapter")
    arm_keys = [key for key in entries if key.startswith("model.arm_")]
    if arm_keys:
        raise RuntimeError(f"Camera checkpoint unexpectedly contains Arm layers: {arm_keys[:3]}")


def build_pipeline(args):
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    variant, run_id, step = checkpoint_identity(checkpoint)
    validate_camera_checkpoint_schema(checkpoint)
    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.875)
    cache_root = Path(
        os.environ.get("CAP_CAMERA_INFERENCE_CACHE_ROOT", "outputs/camera-model-cache")
    ).expanduser().resolve()
    cache_dir = cache_root / variant / run_id / f"checkpoint-{step}" / "transformer"
    cache_args = SimpleNamespace(
        variant=variant,
        model_root=Path(args.model_name).expanduser().resolve(),
        architecture_mode="camera",
        arm_action_dim=14,
        arm_action_num_frames=81,
        force_rebuild_cache=os.environ.get("CAP_CAMERA_FORCE_REBUILD_CACHE", "0") == "1",
    )
    transformer = load_or_build_transformer(
        cache_args,
        checkpoint,
        cache_dir,
        config,
    )

    if args.camera_moe_root:
        raise ValueError("Camera CAP inference must run with --camera_moe_root ''")

    chosen_vae = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = chosen_vae.from_pretrained(
        os.path.join(
            args.model_name,
            config["vae_kwargs"].get("vae_subpath", "vae"),
        ),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(
            args.model_name,
            config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"),
        )
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder"),
        ),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).eval()
    scheduler_class = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }["Flow"]
    scheduler = scheduler_class(
        **benchmark.camera_batch.filter_kwargs(
            scheduler_class,
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
    pipeline.enable_model_cpu_offload(device=args.device)
    return pipeline, boundary


def main() -> None:
    benchmark.camera_batch.build_pipeline = build_pipeline
    benchmark.main()


if __name__ == "__main__":
    main()
