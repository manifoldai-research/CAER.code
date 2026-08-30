#!/usr/bin/env python3
"""Run Arm CAP inference for one metadata_index/sample_id."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import inspect
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


VARIANTS = ("uniform", "e_only", "s_only", "s_max1", "current")
DEFAULT_METADATA = Path(os.environ.get("ARM_METADATA", "data/arm/metadata.json"))
DEFAULT_MODEL = Path(os.environ.get("CAP_CONTROL_MODEL", "models/ti2v_control_init_model"))
DEFAULT_RUNS_ROOT = Path(os.environ.get("ARM_RUNS_ROOT", "outputs/arm"))
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("ARM_INFER_OUTPUT", "outputs/arm-inference"))
DEFAULT_CACHE_ROOT = Path(os.environ.get("MODEL_CACHE_ROOT", "outputs/model-cache"))
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config/wan2.2/wan_civitai_5b.yaml"
DEFAULT_NEGATIVE_PROMPT = (
    "Blurring, mutation, deformation, distortion, dark, static, text subtitles, "
    "comic, line art, low quality, worst quality, malformed robot arm."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer one Arm CAP metadata sample from a timestamped FSDP checkpoint."
    )
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument("--sample-id", type=int, dest="sample_id")
    sample_group.add_argument("--metadata-index", type=int, dest="sample_id")
    parser.add_argument("--variant", choices=VARIANTS, default="current")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="checkpoint step; default: latest complete checkpoint in the latest run",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--prompt", default=None, help="override the metadata prompt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_id < 0:
        raise SystemExit("--sample-id/--metadata-index must be >= 0")
    if args.checkpoint_step is not None and args.checkpoint_step <= 0:
        raise SystemExit("--checkpoint-step must be > 0")
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise SystemExit("--height and --width must be positive multiples of 32")
    if args.frames != 17:
        raise SystemExit(
            "--frames must be 17 because the trained Arm checkpoint uses a fixed 17x14 action MLP input"
        )
    if args.fps <= 0 or args.inference_steps <= 0:
        raise SystemExit("--fps and --inference-steps must be > 0")


def read_json_array_item(path: Path, target_index: int, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    """Stream a single object from a top-level JSON array without loading all metadata."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    index = 0
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffer += chunk
            if not started:
                buffer = buffer.lstrip()
                if not buffer:
                    continue
                if buffer[0] != "[":
                    raise ValueError(f"metadata must be a top-level JSON array: {path}")
                buffer = buffer[1:]
                started = True

            while True:
                buffer = buffer.lstrip()
                if buffer.startswith(","):
                    buffer = buffer[1:].lstrip()
                if buffer.startswith("]"):
                    raise IndexError(f"metadata_index {target_index} is out of range (items={index})")
                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                if not isinstance(item, dict):
                    raise ValueError(f"metadata entry {index} is not an object")
                if index == target_index:
                    return item
                index += 1
                buffer = buffer[end:]
    raise IndexError(f"metadata_index {target_index} is out of range (items={index})")


def first_value(record: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        value: Any = record
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if isinstance(value, dict):
            value = value.get("path") or value.get("file_path") or value.get("value")
        if value is not None and value != "":
            return value
    return default


def resolve_run_and_checkpoint(args: argparse.Namespace) -> tuple[Path, Path, int]:
    def complete_checkpoint(path: Path) -> bool:
        dcp_dir = path / "pytorch_model_fsdp_0"
        metadata = dcp_dir / ".metadata"
        shards = list(dcp_dir.glob("*.distcp"))
        return (
            metadata.is_file()
            and metadata.stat().st_size > 0
            and len(shards) == 8
            and all(shard.stat().st_size > 0 for shard in shards)
            and (path / "scheduler.bin").is_file()
            and (path / "scheduler.bin").stat().st_size > 0
        )

    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser().resolve()
        if run_dir.parent.name != args.variant:
            raise ValueError(
                f"--run-dir belongs to variant {run_dir.parent.name!r}, "
                f"but --variant is {args.variant!r}"
            )
    else:
        variant_root = args.runs_root.expanduser().resolve() / args.variant
        candidates = sorted(path for path in variant_root.glob("20*T*Z") if path.is_dir())
        candidates = [path for path in candidates if any(path.glob("checkpoint-*"))]
        if not candidates:
            raise FileNotFoundError(f"no timestamped run with checkpoints under {variant_root}")
        run_dir = candidates[-1]

    checkpoint_candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        if complete_checkpoint(path):
            checkpoint_candidates.append((step, path))
    if args.checkpoint_step is None:
        if not checkpoint_candidates:
            raise FileNotFoundError(f"no complete FSDP checkpoint under {run_dir}")
        step, checkpoint = max(checkpoint_candidates)
    else:
        step = args.checkpoint_step
        checkpoint = run_dir / f"checkpoint-{step}"
        if not complete_checkpoint(checkpoint):
            raise FileNotFoundError(f"incomplete or missing FSDP checkpoint: {checkpoint}")
    return run_dir, checkpoint.resolve(), step


def sample_paths(record: dict[str, Any]) -> tuple[Path, Path]:
    video = first_value(record, ("file_path", "video_path", "media.path", "video.path"))
    annotation = first_value(
        record,
        ("ann_file", "annotation_path", "action_annotation_path", "arm.path"),
    )
    if video is None or annotation is None:
        raise ValueError("Arm metadata requires both video file_path and ann_file")
    video_path, annotation_path = Path(video), Path(annotation)
    if not video_path.is_file():
        raise FileNotFoundError(f"sample video does not exist: {video_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"sample Arm annotation does not exist: {annotation_path}")
    return video_path, annotation_path


def checkpoint_schema(checkpoint: Path) -> dict[str, Any]:
    from torch.distributed.checkpoint import FileSystemReader

    dcp_dir = checkpoint / "pytorch_model_fsdp_0"
    metadata = FileSystemReader(str(dcp_dir)).read_metadata()
    model_entries = {
        key.removeprefix("model."): value
        for key, value in metadata.state_dict_metadata.items()
        if key.startswith("model.")
    }
    required = (
        "patch_embedding.weight",
        "arm_action_embedder.fc1.weight",
        "arm_action_embedder_proj.fc1.weight",
        "arm_condition_mask_emb.weight",
    )
    missing = [key for key in required if key not in model_entries]
    if missing:
        raise RuntimeError(f"checkpoint lacks required Arm parameters: {missing}")
    patch_shape = tuple(model_entries["patch_embedding.weight"].size)
    if len(patch_shape) != 5 or patch_shape[1] != 100:
        raise RuntimeError(f"checkpoint patch_embedding must have 100 input channels; got {patch_shape}")
    return {"parameter_entries": len(model_entries), "patch_shape": patch_shape}


def cache_directory(args: argparse.Namespace, run_dir: Path, step: int) -> Path:
    run_tag = run_dir.name
    return args.cache_root.expanduser().resolve() / args.variant / run_tag / f"checkpoint-{step}" / "transformer"


def cache_marker(cache_dir: Path) -> Path:
    return cache_dir / "cap_inference_cache.json"


def _parallel_copy_files(pairs: list[tuple[Path, Path]], workers: int = 8) -> None:
    def copy_one(pair: tuple[Path, Path]) -> None:
        source, destination = pair
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, max(1, len(pairs)))
    ) as executor:
        list(executor.map(copy_one, pairs))


def _stage_headroom_bytes() -> int:
    raw_value = os.environ.get("CAP_INFERENCE_STAGE_HEADROOM_BYTES", str(1024**3))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(
            "CAP_INFERENCE_STAGE_HEADROOM_BYTES must be a non-negative integer"
        ) from error
    if value < 0:
        raise ValueError("CAP_INFERENCE_STAGE_HEADROOM_BYTES must be non-negative")
    return value


def require_stage_space(root: Path, payload_bytes: int, label: str) -> None:
    """Fail before a cache copy can exhaust node-local storage."""
    root.mkdir(parents=True, exist_ok=True)
    required = int(payload_bytes) + _stage_headroom_bytes()
    free = shutil.disk_usage(root).free
    if free < required:
        gib = 1024**3
        raise RuntimeError(
            f"insufficient node-local space for {label}: "
            f"need at least {required / gib:.1f} GiB, have {free / gib:.1f} GiB under {root}"
        )


def _parallel_copy_tree(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    pairs = [
        (path, destination / path.relative_to(source))
        for path in source.rglob("*")
        if path.is_file()
    ]
    _parallel_copy_files(
        pairs,
        workers=int(os.environ.get("CAP_INFERENCE_COPY_WORKERS", "8")),
    )


def stage_checkpoint_dcp(checkpoint: Path) -> Path:
    source_dir = checkpoint / "pytorch_model_fsdp_0"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"checkpoint DCP directory is missing: {source_dir}")
    stage_root = Path(
        os.environ.get("CAP_INFERENCE_DCP_STAGE_ROOT", "/dev/shm/cap-inference-dcp-cache")
    ).expanduser().resolve()
    variant = checkpoint.parent.parent.name
    run_tag = checkpoint.parent.name
    target_step = stage_root / variant / run_tag / checkpoint.name
    target_dir = target_step / "pytorch_model_fsdp_0"
    marker_path = target_step / "cap_dcp_stage.json"
    sources = sorted(path for path in source_dir.iterdir() if path.is_file())
    if not sources:
        raise RuntimeError(f"checkpoint DCP directory has no files: {source_dir}")
    expected = {path.name: path.stat().st_size for path in sources}

    def ready() -> bool:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if marker.get("source_checkpoint") != str(checkpoint):
            return False
        return all(
            (target_dir / name).is_file()
            and (target_dir / name).stat().st_size == size
            for name, size in expected.items()
        )

    target_step.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_step.parent / f".{checkpoint.name}.stage.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ready():
            print(f"Reusing node-local DCP checkpoint: {target_dir}", flush=True)
            return target_dir
        require_stage_space(
            target_step.parent,
            sum(expected.values()),
            f"DCP staging for {checkpoint}",
        )
        temporary = target_step.parent / f".{checkpoint.name}.tmp-{os.getpid()}"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary_dir = temporary / "pytorch_model_fsdp_0"
        temporary_dir.mkdir(parents=True)
        print(
            f"Staging {len(sources)} DCP files in parallel from {source_dir} to {target_dir}",
            flush=True,
        )
        try:
            _parallel_copy_files(
                [(source, temporary_dir / source.name) for source in sources],
                workers=int(os.environ.get("CAP_INFERENCE_COPY_WORKERS", "8")),
            )
            (temporary / "cap_dcp_stage.json").write_text(
                json.dumps(
                    {
                        "source_checkpoint": str(checkpoint),
                        "files": expected,
                        "created_unix_time": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if target_step.exists():
                shutil.rmtree(target_step)
            os.replace(temporary, target_step)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        if not ready():
            raise RuntimeError(f"node-local DCP staging failed validation: {target_dir}")
    return target_dir


def stage_base_model_root(model_root: Path, config: Any) -> Path:
    stage_root = Path(
        os.environ.get("CAP_INFERENCE_BASE_MODEL_STAGE_ROOT", "/dev/shm/cap-inference-base-model")
    ).expanduser().resolve()
    target_root = stage_root / "wan2.2-ti2v-5b"
    marker_path = target_root / "cap_base_model_stage.json"
    vae_relative = Path(config["vae_kwargs"].get("vae_subpath", "vae"))
    text_relative = Path(
        config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")
    )
    tokenizer_relative = Path(
        config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer")
    )
    for label, relative in (
        ("VAE", vae_relative),
        ("text encoder", text_relative),
        ("tokenizer", tokenizer_relative),
    ):
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid {label} path in inference config: {relative}")
    pairs: list[tuple[Path, Path]] = []
    required_files = (Path("config.json"), vae_relative, text_relative)
    for relative in required_files:
        source = model_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"required CAP base-model file is missing: {source}")
        pairs.append((source.resolve(), target_root / relative))
    optional_configuration = model_root / "configuration.json"
    if optional_configuration.is_file():
        pairs.append((optional_configuration.resolve(), target_root / "configuration.json"))
    tokenizer_source = (model_root / tokenizer_relative).resolve()
    if not tokenizer_source.is_dir():
        raise FileNotFoundError(f"required CAP tokenizer directory is missing: {tokenizer_source}")
    for source in tokenizer_source.rglob("*"):
        if source.is_file():
            pairs.append((source, target_root / tokenizer_relative / source.relative_to(tokenizer_source)))
    if not any(destination.is_relative_to(target_root / tokenizer_relative) for _, destination in pairs):
        raise RuntimeError(f"CAP tokenizer directory has no files: {tokenizer_source}")
    expected = {
        str(destination.relative_to(target_root)): source.stat().st_size
        for source, destination in pairs
    }

    def ready() -> bool:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if marker.get("source_model_root") != str(model_root):
            return False
        return all(
            (target_root / relative).is_file()
            and (target_root / relative).stat().st_size == size
            for relative, size in expected.items()
        )

    stage_root.mkdir(parents=True, exist_ok=True)
    lock_path = stage_root / ".base-model.stage.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if ready():
            print(f"Reusing node-local VAE/T5/tokenizer: {target_root}", flush=True)
            return target_root
        require_stage_space(
            stage_root,
            sum(expected.values()),
            "CAP VAE/T5/tokenizer staging",
        )
        temporary = stage_root / f".wan2.2-ti2v-5b.tmp-{os.getpid()}"
        shutil.rmtree(temporary, ignore_errors=True)
        print(
            f"Staging VAE/T5/tokenizer in parallel from {model_root} to {target_root}",
            flush=True,
        )
        try:
            temporary_pairs = [
                (source, temporary / destination.relative_to(target_root))
                for source, destination in pairs
            ]
            _parallel_copy_files(
                temporary_pairs,
                workers=int(os.environ.get("CAP_INFERENCE_COPY_WORKERS", "8")),
            )
            (temporary / "cap_base_model_stage.json").write_text(
                json.dumps(
                    {
                        "source_model_root": str(model_root),
                        "files": expected,
                        "created_unix_time": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if target_root.exists():
                shutil.rmtree(target_root)
            os.replace(temporary, target_root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        if not ready():
            raise RuntimeError(f"node-local base-model staging failed validation: {target_root}")
    return target_root


def safetensors_header_keys(path: Path) -> set[str]:
    """Validate a safetensors header without reading tensor payloads."""
    file_size = path.stat().st_size
    if file_size < 10:
        raise ValueError(f"safetensors file is too small: {path}")
    with path.open("rb") as handle:
        header_size_bytes = handle.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"cannot read safetensors header size: {path}")
        header_size = int.from_bytes(header_size_bytes, byteorder="little", signed=False)
        if header_size <= 1 or header_size > file_size - 8:
            raise ValueError(
                f"invalid safetensors header size {header_size} for {file_size}-byte file: {path}"
            )
        header = json.loads(handle.read(header_size).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header is not an object: {path}")
    payload_size = file_size - 8 - header_size
    keys: set[str] = set()
    for key, tensor_metadata in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(tensor_metadata, dict):
            raise ValueError(f"invalid safetensors tensor metadata for {key!r}: {path}")
        offsets = tensor_metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > payload_size
        ):
            raise ValueError(f"invalid safetensors data offsets for {key!r}: {path}")
        keys.add(key)
    if not keys:
        raise ValueError(f"safetensors file contains no tensors: {path}")
    return keys


def cache_is_ready(cache_dir: Path, checkpoint: Path) -> bool:
    marker = cache_marker(cache_dir)
    config = cache_dir / "config.json"
    index = cache_dir / "diffusion_pytorch_model.safetensors.index.json"
    single = cache_dir / "diffusion_pytorch_model.safetensors"
    if not marker.is_file() or not config.is_file() or not (index.is_file() or single.is_file()):
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if value.get("source_checkpoint") != str(checkpoint):
        return False
    if single.is_file():
        try:
            safetensors_header_keys(single)
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return False
        return True
    try:
        weight_index = json.loads(index.read_text(encoding="utf-8"))
        weight_map = weight_index.get("weight_map", {})
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        expected_by_shard: dict[str, set[str]] = {}
        for tensor_name, shard_name in weight_map.items():
            if not isinstance(tensor_name, str) or not isinstance(shard_name, str):
                return False
            expected_by_shard.setdefault(shard_name, set()).add(tensor_name)
        for shard_name, expected_keys in expected_by_shard.items():
            actual_keys = safetensors_header_keys(cache_dir / shard_name)
            if not expected_keys.issubset(actual_keys):
                return False
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return False
    return True


def prune_old_runtime_caches(runtime_cache_root: Path, checkpoint: Path) -> None:
    """Keep one active transformer cache per isolated Volcano job."""
    if os.environ.get("CAP_INFERENCE_KEEP_RUNTIME_CACHE", "0") == "1":
        return
    if not runtime_cache_root.is_dir():
        return
    current_source = str(checkpoint)
    for marker in runtime_cache_root.glob("*/*/*/transformer/cap_inference_cache.json"):
        try:
            value = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if value.get("source_checkpoint") == current_source:
            continue
        cache_dir = marker.parent
        print(f"Removing stale node-local transformer cache: {cache_dir}", flush=True)
        shutil.rmtree(cache_dir, ignore_errors=True)


def cleanup_staged_checkpoint_dcp(staged_dcp_dir: Path, checkpoint: Path) -> None:
    """Release the 20+ GiB DCP staging copy after the validated cache exists."""
    if os.environ.get("CAP_INFERENCE_KEEP_DCP_STAGE", "0") == "1":
        return
    target_step = staged_dcp_dir.parent
    marker_path = target_step / "cap_dcp_stage.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if marker.get("source_checkpoint") != str(checkpoint):
        return
    print(f"Releasing node-local DCP staging: {target_step}", flush=True)
    shutil.rmtree(target_step, ignore_errors=True)


def transformer_kwargs(
    config: Any,
    *,
    architecture_mode: str = "arm",
    arm_action_dim: int = 14,
    arm_action_num_frames: int = 17,
) -> dict[str, Any]:
    from omegaconf import OmegaConf

    kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    if architecture_mode == "poseanything":
        kwargs.update(
            {
                "in_dim": 96,
                "in_channels": 96,
                "add_control_adapter": False,
                "add_arm_action_embedder": False,
                "cap_expected_patch_embedding_source_channels": [48, 96],
            }
        )
    elif architecture_mode == "arm":
        kwargs.update(
            {
                "in_dim": 100,
                "in_channels": 100,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "downscale_factor_control_adapter": 16,
                "add_arm_action_embedder": True,
                "arm_action_dim": int(arm_action_dim),
                "arm_action_num_frames": int(arm_action_num_frames),
                "zero_init_arm_action_output": True,
                "cap_expected_patch_embedding_source_channels": [48, 100],
            }
        )
    elif architecture_mode == "camera":
        kwargs.update(
            {
                "in_dim": 100,
                "in_channels": 100,
                "add_control_adapter": True,
                "in_dim_control_adapter": 24,
                "downscale_factor_control_adapter": 16,
                "zero_init_control_adapter_output": True,
                "add_arm_action_embedder": False,
                "cap_expected_patch_embedding_source_channels": [48, 100],
            }
        )
    else:
        raise ValueError(f"unknown architecture_mode: {architecture_mode}")
    return kwargs


def instantiate_arm_transformer_from_config(args: argparse.Namespace, config: Any):
    import json as json_module
    from videox_fun.models import Wan2_2Transformer3DModel

    with (args.model_root / "config.json").open("r", encoding="utf-8") as handle:
        model_config = json_module.load(handle)
    kwargs = transformer_kwargs(
        config,
        architecture_mode=getattr(args, "architecture_mode", "arm"),
        arm_action_dim=getattr(args, "arm_action_dim", 14),
        arm_action_num_frames=getattr(args, "arm_action_num_frames", 17),
    )
    mapping = kwargs.pop("dict_mapping", {})
    kwargs.pop("cap_expected_patch_embedding_source_channels", None)
    for source_name, target_name in mapping.items():
        if target_name not in kwargs:
            kwargs[target_name] = model_config[source_name]
    return Wan2_2Transformer3DModel.from_config(model_config, **kwargs)


def load_dcp_state(dcp: Any, state: dict[str, Any], checkpoint_id: str) -> bool:
    """Load a DCP state dict across the PyTorch APIs used by CAP runtimes."""
    dcp_load_kwargs: dict[str, Any] = {"checkpoint_id": checkpoint_id}
    try:
        supports_no_dist = "no_dist" in inspect.signature(dcp.load).parameters
    except (TypeError, ValueError):
        # Some extension-backed callables do not expose an inspectable
        # signature. The modern API works without this legacy argument.
        supports_no_dist = False
    if supports_no_dist:
        dcp_load_kwargs["no_dist"] = True
    try:
        dcp.load(state, **dcp_load_kwargs)
    except TypeError as error:
        if not supports_no_dist or "no_dist" not in str(error):
            raise
        # Be defensive if a runtime reports a stale signature but rejects the
        # old keyword at the actual call site.
        dcp_load_kwargs.pop("no_dist")
        dcp.load(state, **dcp_load_kwargs)
        supports_no_dist = False
    return supports_no_dist


def load_or_build_transformer(
    args: argparse.Namespace,
    checkpoint: Path,
    cache_dir: Path,
    config: Any,
):
    import torch
    import torch.distributed.checkpoint as dcp
    from accelerate import init_empty_weights
    from videox_fun.models import Wan2_2Transformer3DModel

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    runtime_cache_root = Path(
        os.environ.get(
            "CAP_INFERENCE_RUNTIME_CACHE_ROOT",
            "/dev/shm/cap-inference-model-cache",
        )
    ).expanduser().resolve()
    runtime_cache_dir = (
        runtime_cache_root
        / args.variant
        / cache_dir.parent.parent.name
        / cache_dir.parent.name
        / cache_dir.name
    )
    runtime_cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir.parent / ".build.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prune_old_runtime_caches(runtime_cache_root, checkpoint)
        if args.force_rebuild_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
            shutil.rmtree(runtime_cache_dir, ignore_errors=True)
        local_build_dir: Path | None = None
        staged_dcp_dir: Path | None = None
        if not cache_is_ready(cache_dir, checkpoint):
            print(f"Building inference transformer cache from {checkpoint}", flush=True)
            print("This first build reads only pytorch_model_fsdp_0; optimizer shards are not read.", flush=True)
            staged_dcp_dir = stage_checkpoint_dcp(checkpoint)
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            shared_tmp_dir = cache_dir.parent / f".{cache_dir.name}.tmp-{os.getpid()}"
            local_tmp_root = Path(
                os.environ.get("CAP_INFERENCE_CACHE_TMPDIR", str(runtime_cache_root))
            ).expanduser().resolve()
            local_tmp_root.mkdir(parents=True, exist_ok=True)
            local_tmp_dir = local_tmp_root / (
                f"cap-inference-cache-{args.variant}-{cache_dir.parent.parent.name}-"
                f"{cache_dir.parent.name}-{os.getpid()}"
            )
            shutil.rmtree(shared_tmp_dir, ignore_errors=True)
            shutil.rmtree(local_tmp_dir, ignore_errors=True)
            # The FSDP checkpoint is a complete transformer. Instantiate only
            # its architecture so cache creation never reads base-model weights.
            try:
                with init_empty_weights(include_buffers=True):
                    model = instantiate_arm_transformer_from_config(args, config)
                model.to_empty(device="cpu")
                state = {"model": model.state_dict()}
                # `no_dist` was removed from the newer DCP load API.  When it
                # is absent, `dcp.load` uses its built-in no-process-group
                # path, so keep this cache builder compatible with both APIs.
                load_dcp_state(dcp, state, str(staged_dcp_dir))
                missing, unexpected = model.load_state_dict(state["model"], strict=True)
                if missing or unexpected:
                    raise RuntimeError(
                        f"FSDP restore was not strict: missing={missing}, unexpected={unexpected}"
                    )
                model.to(dtype=torch.bfloat16)
                model.save_pretrained(
                    str(local_tmp_dir), safe_serialization=True, max_shard_size="5GB"
                )
                marker_value = {
                    "source_checkpoint": str(checkpoint),
                    "created_unix_time": time.time(),
                    "dtype": "bfloat16",
                }
                (local_tmp_dir / "cap_inference_cache.json").write_text(
                    json.dumps(marker_value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if not cache_is_ready(local_tmp_dir, checkpoint):
                    raise RuntimeError(
                        f"locally staged inference cache failed validation: {local_tmp_dir}"
                    )
                print(
                    f"Copying validated inference cache from {local_tmp_dir} to {cache_dir}",
                    flush=True,
                )
                _parallel_copy_tree(local_tmp_dir, shared_tmp_dir)
                if not cache_is_ready(shared_tmp_dir, checkpoint):
                    raise RuntimeError(
                        f"copied inference cache failed validation: {shared_tmp_dir}"
                    )
                os.replace(shared_tmp_dir, cache_dir)
                del state, model
                local_build_dir = local_tmp_dir
            except BaseException:
                shutil.rmtree(shared_tmp_dir, ignore_errors=True)
                shutil.rmtree(local_tmp_dir, ignore_errors=True)
                raise
        else:
            print(f"Reusing persistent inference transformer cache: {cache_dir}", flush=True)

        if not cache_is_ready(runtime_cache_dir, checkpoint):
            runtime_tmp_dir = runtime_cache_dir.parent / (
                f".{runtime_cache_dir.name}.tmp-{os.getpid()}"
            )
            shutil.rmtree(runtime_cache_dir, ignore_errors=True)
            shutil.rmtree(runtime_tmp_dir, ignore_errors=True)
            try:
                if local_build_dir is not None and local_build_dir.exists():
                    print(
                        f"Publishing node-local inference cache: {runtime_cache_dir}",
                        flush=True,
                    )
                    os.replace(local_build_dir, runtime_tmp_dir)
                    local_build_dir = None
                else:
                    print(
                        f"Staging node-local inference cache from {cache_dir} "
                        f"to {runtime_cache_dir}",
                        flush=True,
                    )
                    _parallel_copy_tree(cache_dir, runtime_tmp_dir)
                if not cache_is_ready(runtime_tmp_dir, checkpoint):
                    raise RuntimeError(
                        f"node-local inference cache failed validation: {runtime_tmp_dir}"
                    )
                os.replace(runtime_tmp_dir, runtime_cache_dir)
            except BaseException:
                shutil.rmtree(runtime_tmp_dir, ignore_errors=True)
                raise
        else:
            print(f"Reusing node-local inference cache: {runtime_cache_dir}", flush=True)

        if local_build_dir is not None:
            shutil.rmtree(local_build_dir, ignore_errors=True)
        if staged_dcp_dir is not None:
            cleanup_staged_checkpoint_dcp(staged_dcp_dir, checkpoint)

    model = Wan2_2Transformer3DModel.from_pretrained(
        str(runtime_cache_dir),
        transformer_additional_kwargs=transformer_kwargs(
            config,
            architecture_mode=getattr(args, "architecture_mode", "arm"),
            arm_action_dim=getattr(args, "arm_action_dim", 14),
            arm_action_num_frames=getattr(args, "arm_action_num_frames", 17),
        ),
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    )
    model.eval().requires_grad_(False)
    return model


def extract_sample(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from PIL import Image

    video_path, annotation_path = sample_paths(record)
    reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
    total_frames = len(reader)
    explicit = first_value(record, ("sampling.frame_indices", "frame_indices"))
    if explicit is not None:
        indices = np.asarray(explicit, dtype=np.int64).reshape(-1)
        indices = np.clip(indices, 0, max(total_frames - 1, 0))
        if len(indices) > args.frames:
            selection = np.linspace(0, len(indices) - 1, args.frames, dtype=int)
            indices = indices[selection]
    else:
        start = int(first_value(record, ("sampling.start_frame", "start_frame"), 0))
        stride = max(
            int(first_value(record, ("sampling.stride", "video_sample_stride"), 1)),
            1,
        )
        declared_frames = max(
            int(
                first_value(
                    record,
                    ("sampling.num_frames", "window_size", "video_sample_n_frames"),
                    args.frames,
                )
            ),
            1,
        )
        requested_frames = min(declared_frames, args.frames)
        max_possible = (total_frames - 1) // stride + 1
        actual_frames = min(requested_frames, max_possible)
        clip_length = min(total_frames, (actual_frames - 1) * stride + 1)
        max_start = max(total_frames - clip_length, 0)
        start = int(np.clip(start, 0, max_start))
        indices = np.linspace(start, start + clip_length - 1, actual_frames, dtype=int)
    if len(indices) != args.frames:
        raise ValueError(
            f"sample provides {len(indices)} frames, but inference requires {args.frames}: {video_path}"
        )

    frames = reader.get_batch(indices).asnumpy()
    resized = np.stack(
        [
            np.asarray(
                Image.fromarray(frame).resize(
                    (args.width, args.height), Image.Resampling.BILINEAR
                )
            )
            for frame in frames
        ]
    )

    with annotation_path.open("r", encoding="utf-8") as handle:
        annotation = json.load(handle)
    action_key = str(
        first_value(record, ("arm.key", "arm_action_key", "action_key"), "state")
    )
    if action_key not in annotation:
        raise KeyError(f"Arm annotation has no key {action_key!r}: {annotation_path}")
    action = np.asarray(annotation[action_key], dtype=np.float32)
    if action.ndim == 1:
        action = action[None, :]
    elif action.ndim > 2:
        action = action.reshape(action.shape[0], -1)
    clipped_indices = np.clip(indices, 0, max(len(action) - 1, 0))
    action = action[clipped_indices]
    if action.shape[1] > 14:
        action = action[:, :14]
    elif action.shape[1] < 14:
        action = np.pad(action, ((0, 0), (0, 14 - action.shape[1])))

    return {
        "video_path": video_path,
        "annotation_path": annotation_path,
        "action_key": action_key,
        "frame_indices": indices.tolist(),
        "frames_uint8": resized,
        "first_frame": Image.fromarray(resized[0]),
        "arm_action": torch.from_numpy(action).unsqueeze(0),
        "prompt": args.prompt
        or str(
            first_value(
                record,
                ("text", "prompt.text", "prompt", "caption"),
                "",
            )
        ),
    }


def build_pipeline(
    args: argparse.Namespace,
    checkpoint: Path,
    cache_dir: Path,
):
    from videox_fun.runtime_compat import configure_attention_runtime

    configure_attention_runtime()

    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    from videox_fun.models import AutoTokenizer, AutoencoderKLWan3_8, WanT5EncoderModel
    from videox_fun.pipeline import Wan2_2FunControlPipeline
    from videox_fun.utils.utils import filter_kwargs

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan 2.2 Arm inference")
    config = OmegaConf.load(args.config)
    args.model_root = stage_base_model_root(args.model_root.expanduser().resolve(), config)
    transformer = load_or_build_transformer(args, checkpoint, cache_dir, config)
    vae = AutoencoderKLWan3_8.from_pretrained(
        str(args.model_root / config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_root / config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(args.model_root / config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
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
    device = torch.device(args.device)
    pipeline.enable_model_cpu_offload(device=device)
    return pipeline, config, device


def run_inference_with_pipeline(
    args: argparse.Namespace,
    sample: dict[str, Any],
    checkpoint: Path,
    cache_dir: Path,
    output_dir: Path,
    pipeline: Any,
    config: Any,
    device: Any,
) -> Path:
    import numpy as np
    import torch
    from videox_fun.utils.utils import get_image_to_video_latent, save_videos_grid

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
            f"invalid first-frame video latent input shape: "
            f"expected={expected_video_shape} actual={tuple(inpaint_video.shape)}"
        )
    if tuple(inpaint_mask.shape) != expected_mask_shape:
        raise RuntimeError(
            f"invalid first-frame mask shape: "
            f"expected={expected_mask_shape} actual={tuple(inpaint_mask.shape)}"
        )
    if torch.count_nonzero(inpaint_mask[:, :, :1]).item() != 0:
        raise RuntimeError("first-frame mask must preserve frame 0")
    if torch.count_nonzero(inpaint_mask[:, :, 1:] != 255).item() != 0:
        raise RuntimeError("first-frame mask must mark frames 1..N as generated")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        generated = pipeline(
            sample["prompt"],
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            video=inpaint_video,
            mask_video=inpaint_mask,
            control_video=None,
            arm_action=sample["arm_action"],
            arm_action_mask=torch.ones((1,), dtype=torch.float32),
            num_frames=args.frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            boundary=float(config["transformer_additional_kwargs"].get("boundary", 0.9)),
            shift=int(config["scheduler_kwargs"].get("shift", 5)),
            use_empty_control_latents=False,
        ).videos.cpu()

    output_dir.mkdir(parents=True, exist_ok=False)
    generated_path = output_dir / "generated.mp4"
    target_path = output_dir / "target_clip.mp4"
    first_frame_path = output_dir / "first_frame.png"
    save_videos_grid(generated, str(generated_path), fps=args.fps)
    target = torch.from_numpy(np.asarray(sample["frames_uint8"]).copy()).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
    save_videos_grid(target, str(target_path), fps=args.fps)
    sample["first_frame"].save(first_frame_path)

    manifest = {
        "sample_id": args.sample_id,
        "metadata_index": args.sample_id,
        "variant": args.variant,
        "checkpoint": str(checkpoint),
        "cache": str(cache_dir),
        "video_path": str(sample["video_path"]),
        "annotation_path": str(sample["annotation_path"]),
        "action_key": sample["action_key"],
        "frame_indices": sample["frame_indices"],
        "prompt": sample["prompt"],
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "frames": args.frames,
        "fps": args.fps,
        "seed": args.seed,
        "random_selection_seed": getattr(args, "random_seed", None),
        "generation_seed_base": getattr(args, "generation_seed", None),
        "inference_steps": args.inference_steps,
        "guidance_scale": args.guidance_scale,
        "generated": str(generated_path),
        "target_clip": str(target_path),
        "first_frame": str(first_frame_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return generated_path


def run_inference(
    args: argparse.Namespace,
    sample: dict[str, Any],
    checkpoint: Path,
    cache_dir: Path,
    output_dir: Path,
) -> Path:
    pipeline, config, device = build_pipeline(args, checkpoint, cache_dir)
    return run_inference_with_pipeline(
        args,
        sample,
        checkpoint,
        cache_dir,
        output_dir,
        pipeline,
        config,
        device,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    for path, label in ((args.metadata, "metadata"), (args.model_root, "model root"), (args.config, "config")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    run_dir, checkpoint, checkpoint_step = resolve_run_and_checkpoint(args)
    record = read_json_array_item(args.metadata, args.sample_id)
    video_path, annotation_path = sample_paths(record)
    sample = extract_sample(record, args)
    schema = checkpoint_schema(checkpoint)
    cache_dir = cache_directory(args, run_dir, checkpoint_step)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = (
        args.output_root.expanduser().resolve()
        / args.variant
        / f"checkpoint-{checkpoint_step}"
        / f"sample-{args.sample_id}-{timestamp}"
    )
    summary = {
        "sample_id": args.sample_id,
        "metadata_index": args.sample_id,
        "variant": args.variant,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_step": checkpoint_step,
        "checkpoint_parameter_entries": schema["parameter_entries"],
        "checkpoint_patch_shape": schema["patch_shape"],
        "video_path": str(video_path),
        "annotation_path": str(annotation_path),
        "start_frame": first_value(record, ("sampling.start_frame", "start_frame"), 0),
        "window_size": first_value(record, ("sampling.num_frames", "window_size"), args.frames),
        "frame_indices": sample["frame_indices"],
        "decoded_frame_shape": list(sample["frames_uint8"].shape),
        "arm_action_shape": list(sample["arm_action"].shape),
        "action_key": sample["action_key"],
        "prompt_present": bool(first_value(record, ("text", "prompt.text", "prompt", "caption"), "")),
        "cache_dir": str(cache_dir),
        "cache_ready": cache_is_ready(cache_dir, checkpoint),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
    }
    print("CAP Arm sample inference preflight:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("DRY_RUN passed: no model shards were restored and no GPU inference was started.")
        return 0

    generated_path = run_inference(
        args, sample, checkpoint, cache_dir, output_dir
    )
    print(f"CAP Arm inference complete: {generated_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
