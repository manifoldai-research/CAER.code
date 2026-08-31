"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import csv
import gc
import json
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
import warnings
from datetime import timedelta


def _run_lightweight_metadata_preflight_if_requested():
    if "--metadata_preflight_only" not in sys.argv:
        return
    import importlib.util
    from pathlib import Path

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--metadata_preflight_only", action="store_true")
    parser.add_argument(
        "--dataset_name",
        choices=["custom", "current_arm", "action_map", "camera", "poseanything"],
        default="custom",
    )
    parser.add_argument("--action_injection", default="auto")
    parser.add_argument("--train_data_meta", default=None)
    parser.add_argument("--train_data_dir", default=None)
    parser.add_argument("--skip_dataset_preflight", action="store_true")
    args, _ = parser.parse_known_args()
    if args.skip_dataset_preflight:
        print("CAP dataset preflight skipped by request")
        raise SystemExit(0)

    project_root = Path(__file__).resolve().parents[2]
    module_path = project_root / "videox_fun/data/cap_dataset_presets.py"
    spec = importlib.util.spec_from_file_location("cap_dataset_presets_light", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preset = module.CAP_DATASET_PRESETS.get(args.dataset_name)
    if preset is None:
        if args.train_data_meta is None or args.action_injection == "auto":
            parser.error(
                "custom preflight requires --train_data_meta and an explicit --action_injection"
            )
        metadata = args.train_data_meta
        data_root = args.train_data_dir
        action_injection = args.action_injection
    else:
        metadata = args.train_data_meta or preset["train_data_meta"]
        data_root = args.train_data_dir if args.train_data_dir is not None else preset["train_data_dir"]
        action_injection = (
            preset["action_injection"]
            if args.action_injection == "auto"
            else args.action_injection
        )
    summary = module.preflight_cap_metadata(
        metadata, data_root, action_injection, check_paths=True
    )
    print("CAP dataset preflight: " + json.dumps(summary, ensure_ascii=True, sort_keys=True))
    raise SystemExit(0)


_run_lightweight_metadata_preflight_if_requested()


current_file_path = os.path.abspath(__file__)
project_roots = [
    os.path.dirname(current_file_path),
    os.path.dirname(os.path.dirname(current_file_path)),
    os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))),
]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from videox_fun.runtime_compat import configure_attention_runtime
from videox_fun.training.realtime_metrics import (
    append_jsonl,
    prepare_step_metrics_jsonl,
    write_json_atomic,
)
from videox_fun.training.sample_loss_recorder import (
    Method1SampleLossRecorder as DualLossSampleRecorder,
    padded_epoch_sample_count,
)


CAP_ATTENTION_RUNTIME = configure_attention_runtime()


class _LegacySingleLossSampleRecorder:
    """Deprecated and unused; runtime recording uses DualLossSampleRecorder."""
    CSV_FIELDS = (
        "loss_rank_desc",
        "epoch",
        "metadata_index",
        "sample_id",
        "episode_id",
        "task",
        "start_frame",
        "file_path",
        "mean_loss",
        "min_loss",
        "max_loss",
        "observations",
        "action_conditioned_observations",
        "first_optimizer_step_before",
        "last_optimizer_step_before",
    )

    def __init__(self, output_dir, metadata, accelerator):
        self.output_dir = os.path.abspath(output_dir)
        self.metadata = metadata
        self.accelerator = accelerator
        self.current_epoch = None
        self.aggregates = {}
        self.raw_file = None
        self.previous_epoch = None
        self.previous_losses = None
        if accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

    def _metadata_row(self, metadata_index):
        if 0 <= metadata_index < len(self.metadata):
            item = self.metadata[metadata_index]
        else:
            item = {}
        episode_id = item.get("episode_id", item.get("episode", ""))
        start_frame = item.get("start_frame", "")
        return {
            "metadata_index": metadata_index,
            "sample_id": str(metadata_index),
            "episode_id": episode_id,
            "task": item.get("task", ""),
            "start_frame": start_frame,
            "file_path": item.get("file_path", ""),
        }

    def start_epoch(self, epoch):
        if not self.accelerator.is_main_process:
            return
        if self.raw_file is not None:
            raise RuntimeError("Method1 sample-loss recorder started a new epoch before finalizing the previous one.")
        self.current_epoch = int(epoch)
        self.aggregates = {}
        raw_path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}_visits.jsonl")
        self.raw_file = open(raw_path, "w", encoding="utf-8", buffering=1024 * 1024)

    def record_gathered(self, epoch, dataloader_step, optimizer_step_before, gathered):
        if not self.accelerator.is_main_process:
            return
        if self.raw_file is None or self.current_epoch != int(epoch):
            raise RuntimeError("Method1 sample-loss recorder received a sample outside an active epoch.")
        for metadata_index_f, loss, action_conditioned_f in gathered.detach().cpu().tolist():
            metadata_index = int(metadata_index_f)
            loss = float(loss)
            action_conditioned = int(action_conditioned_f >= 0.5)
            if not math.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite Method1 sample loss for metadata index {metadata_index}: {loss}"
                )
            self.raw_file.write(
                json.dumps(
                    {
                        "epoch": int(epoch) + 1,
                        "metadata_index": metadata_index,
                        "sample_id": str(metadata_index),
                        "loss": loss,
                        "action_conditioned": action_conditioned,
                        "dataloader_step": int(dataloader_step),
                        "optimizer_step_before": int(optimizer_step_before),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            aggregate = self.aggregates.get(metadata_index)
            if aggregate is None:
                self.aggregates[metadata_index] = {
                    "sum": loss,
                    "min": loss,
                    "max": loss,
                    "count": 1,
                    "action_count": action_conditioned,
                    "first_step": int(optimizer_step_before),
                    "last_step": int(optimizer_step_before),
                }
            else:
                aggregate["sum"] += loss
                aggregate["min"] = min(aggregate["min"], loss)
                aggregate["max"] = max(aggregate["max"], loss)
                aggregate["count"] += 1
                aggregate["action_count"] += action_conditioned
                aggregate["last_step"] = int(optimizer_step_before)

    @staticmethod
    def _atomic_csv(path, fieldnames, rows):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)

    @staticmethod
    def _display(value):
        return "N/A" if value is None else value

    def finalize_epoch(self, epoch, complete, optimizer_step_after):
        if not self.accelerator.is_main_process:
            return None
        if self.raw_file is None or self.current_epoch != int(epoch):
            raise RuntimeError("Method1 sample-loss recorder cannot finalize an inactive epoch.")
        self.raw_file.close()
        self.raw_file = None

        losses = {
            metadata_index: values["sum"] / values["count"]
            for metadata_index, values in self.aggregates.items()
        }
        sorted_items = sorted(losses.items(), key=lambda item: (-item[1], item[0]))
        missing_items = [
            metadata_index
            for metadata_index in range(len(self.metadata))
            if metadata_index not in losses
        ]
        epoch_path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}_by_loss_desc.csv")
        epoch_rows = []
        for rank, (metadata_index, mean_loss) in enumerate(sorted_items, 1):
            aggregate = self.aggregates[metadata_index]
            metadata_row = self._metadata_row(metadata_index)
            epoch_rows.append(
                {
                    "loss_rank_desc": rank,
                    "epoch": int(epoch) + 1,
                    **metadata_row,
                    "mean_loss": f"{mean_loss:.17g}",
                    "min_loss": f"{aggregate['min']:.17g}",
                    "max_loss": f"{aggregate['max']:.17g}",
                    "observations": aggregate["count"],
                    "action_conditioned_observations": aggregate["action_count"],
                    "first_optimizer_step_before": aggregate["first_step"],
                    "last_optimizer_step_before": aggregate["last_step"],
                }
            )
        for metadata_index in missing_items:
            epoch_rows.append(
                {
                    "loss_rank_desc": "N/A",
                    "epoch": int(epoch) + 1,
                    **self._metadata_row(metadata_index),
                    "mean_loss": "N/A",
                    "min_loss": "N/A",
                    "max_loss": "N/A",
                    "observations": 0,
                    "action_conditioned_observations": 0,
                    "first_optimizer_step_before": "N/A",
                    "last_optimizer_step_before": "N/A",
                }
            )
        self._atomic_csv(epoch_path, self.CSV_FIELDS, epoch_rows)

        comparison_path = None
        if self.previous_losses is not None:
            previous_epoch = int(self.previous_epoch)
            comparison_path = os.path.join(
                self.output_dir,
                f"epoch_{previous_epoch + 1:03d}_to_{epoch + 1:03d}_by_loss_drop_desc.csv",
            )
            comparison_rows = []
            for metadata_index in range(len(self.metadata)):
                previous_loss = self.previous_losses.get(metadata_index)
                current_loss = losses.get(metadata_index)
                loss_drop = (
                    None if previous_loss is None or current_loss is None else previous_loss - current_loss
                )
                loss_drop_pct = (
                    None
                    if loss_drop is None or previous_loss == 0
                    else loss_drop / previous_loss * 100.0
                )
                comparison_rows.append(
                    {
                        "metadata_index": metadata_index,
                        "sample_id": str(metadata_index),
                        **{
                            key: value
                            for key, value in self._metadata_row(metadata_index).items()
                            if key not in ("metadata_index", "sample_id")
                        },
                        "previous_epoch": previous_epoch + 1,
                        "previous_loss": self._display(
                            None if previous_loss is None else f"{previous_loss:.17g}"
                        ),
                        "current_epoch": int(epoch) + 1,
                        "current_loss": self._display(
                            None if current_loss is None else f"{current_loss:.17g}"
                        ),
                        "loss_drop_previous_minus_current": self._display(
                            None if loss_drop is None else f"{loss_drop:.17g}"
                        ),
                        "loss_drop_pct": self._display(
                            None if loss_drop_pct is None else f"{loss_drop_pct:.17g}"
                        ),
                    }
                )
            comparison_rows.sort(
                key=lambda row: (
                    row["loss_drop_previous_minus_current"] == "N/A",
                    -float(row["loss_drop_previous_minus_current"])
                    if row["loss_drop_previous_minus_current"] != "N/A"
                    else 0.0,
                    int(row["metadata_index"]),
                )
            )
            for rank, row in enumerate(comparison_rows, 1):
                row["loss_drop_rank_desc"] = rank
            comparison_fields = (
                "loss_drop_rank_desc",
                "metadata_index",
                "sample_id",
                "episode_id",
                "task",
                "start_frame",
                "file_path",
                "previous_epoch",
                "previous_loss",
                "current_epoch",
                "current_loss",
                "loss_drop_previous_minus_current",
                "loss_drop_pct",
            )
            self._atomic_csv(comparison_path, comparison_fields, comparison_rows)

        summary = {
            "epoch": int(epoch) + 1,
            "complete": bool(complete),
            "optimizer_step_after": int(optimizer_step_after),
            "observations": sum(values["count"] for values in self.aggregates.values()),
            "unique_samples": len(losses),
            "missing_metadata_candidates": max(len(self.metadata) - len(losses), 0),
            "loss_sorted_csv": epoch_path,
            "comparison_csv": comparison_path,
            "loss_definition": "per-sample Method1 focused loss used for backward",
        }
        summary_path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}_summary.json")
        tmp_summary_path = summary_path + ".tmp"
        with open(tmp_summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_summary_path, summary_path)

        self.previous_epoch = int(epoch)
        self.previous_losses = losses
        self.current_epoch = None
        self.aggregates = {}
        return summary

def _safe_cache_component(value):
    value = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _isolate_method1_runtime_cache():
    if os.environ.get("VIDEOX_METHOD1_ISOLATE_RUNTIME_CACHE", "0") != "1":
        return

    cache_root = os.environ.get("VIDEOX_RUNTIME_CACHE_ROOT", "/tmp/videox_fun_method1_cache")
    run_tag = _safe_cache_component(os.environ.get("RUN_TAG", "method1"))
    host = _safe_cache_component(os.environ.get("HOSTNAME") or os.uname().nodename)
    rank = _safe_cache_component(os.environ.get("RANK", "0"))
    local_rank = _safe_cache_component(os.environ.get("LOCAL_RANK", "0"))
    process_cache_root = os.path.join(cache_root, run_tag, f"{host}_rank{rank}_local{local_rank}")

    triton_cache_dir = os.path.join(process_cache_root, "triton")
    torch_extensions_dir = os.path.join(process_cache_root, "torch_extensions")
    os.makedirs(triton_cache_dir, exist_ok=True)
    os.makedirs(torch_extensions_dir, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    os.environ["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir


_isolate_method1_runtime_cache()


def _method1_colorize(values, vmax):
    x = np.clip(values / max(float(vmax), 1e-8), 0.0, 1.0)
    stops = np.asarray(
        [[49, 54, 149], [69, 117, 180], [116, 173, 209], [253, 174, 97], [215, 48, 39]],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int64)
    high = np.clip(low + 1, 0, len(stops) - 1)
    frac = scaled[..., None] - low[..., None]
    return np.uint8(np.clip(stops[low] * (1.0 - frac) + stops[high] * frac, 0, 255))


def _method1_colorize_centered(values, vmin, center, vmax):
    low_color = np.asarray([49, 54, 149], dtype=np.float32)
    mid_color = np.asarray([220, 220, 220], dtype=np.float32)
    high_color = np.asarray([215, 48, 39], dtype=np.float32)

    values = np.asarray(values, dtype=np.float32)
    image = np.empty(values.shape + (3,), dtype=np.float32)
    lower = values <= center
    lower_scale = np.clip((values - float(vmin)) / max(float(center - vmin), 1e-8), 0.0, 1.0)
    upper_scale = np.clip((values - float(center)) / max(float(vmax - center), 1e-8), 0.0, 1.0)
    image[lower] = low_color * (1.0 - lower_scale[lower, None]) + mid_color * lower_scale[lower, None]
    image[~lower] = mid_color * (1.0 - upper_scale[~lower, None]) + high_color * upper_scale[~lower, None]
    return np.uint8(np.clip(image, 0, 255))


def _method1_centered_limits(values, center=1.0):
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, float(center), 2.0
    vmin = float(np.percentile(flat, 1.0))
    vmax = float(np.percentile(flat, 99.0))
    if vmin >= center:
        vmin = float(np.min(flat))
    if vmax <= center:
        vmax = float(np.max(flat))
    if vmin >= center:
        vmin = float(center) - 1e-3
    if vmax <= center:
        vmax = float(center) + 1e-3
    return vmin, float(center), vmax


def _save_method1_map_gif(tensor, output_dir, name, global_step, max_frames, centered_at_one=False):
    if tensor is None:
        return None

    values = tensor.detach().float().cpu()
    if values.ndim != 5 or values.shape[0] < 1 or values.shape[1] < 1:
        return None

    frames = values[0, 0]
    finite = torch.isfinite(frames)
    if not bool(finite.all().item()):
        frames = torch.where(finite, frames, torch.zeros_like(frames))

    frame_np = frames.numpy()
    if centered_at_one:
        vmin, center, vmax = _method1_centered_limits(frame_np, center=1.0)
    else:
        positive = frame_np[frame_np > 0]
        vmax = float(np.percentile(positive, 99.0)) if positive.size else float(np.max(np.abs(frame_np)) + 1e-8)
        vmin = 0.0
        center = None
    frame_count = min(int(max_frames), frame_np.shape[0])
    latent_scale = max(1, min(32, 512 // max(int(frame_np.shape[-1]), 1)))
    images = []
    png_dir = os.path.join(output_dir, "png", name)
    os.makedirs(png_dir, exist_ok=True)

    for frame_idx in range(frame_count):
        if centered_at_one:
            heat = _method1_colorize_centered(frame_np[frame_idx], vmin, center, vmax)
        else:
            heat = _method1_colorize(frame_np[frame_idx], vmax)
        image = Image.fromarray(heat).resize(
            (heat.shape[1] * latent_scale, heat.shape[0] * latent_scale),
            Image.Resampling.NEAREST,
        )
        image.save(os.path.join(png_dir, f"{name}_step{global_step}_t{frame_idx:02d}.png"))
        images.append(image)

    if images:
        gif_path = os.path.join(output_dir, f"{name}_step{global_step}.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=120, loop=0)

    return {
        "shape": list(values.shape),
        "finite": bool(torch.isfinite(values).all().item()),
        "min": float(values.min().item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "vmin_p01": vmin,
        "center": center,
        "vmax_p99_positive": vmax,
    }


def _method1_pixel_values_to_uint8(pixel_values):
    video = pixel_values.detach().float().cpu()
    if video.ndim != 4:
        return None

    if video.shape[1] in (1, 3):
        video = video.permute(0, 2, 3, 1).contiguous()
    elif video.shape[0] in (1, 3):
        video = video.permute(1, 2, 3, 0).contiguous()
    elif video.shape[-1] not in (1, 3):
        return None

    video = video.clamp(-1.0, 1.0)
    video = (video * 0.5 + 0.5) * 255.0
    if video.shape[-1] == 1:
        video = video.expand(-1, -1, -1, 3)
    return np.uint8(video.clamp(0, 255).numpy())


def _method1_upsample_map_to_video(tensor, video_shape):
    if tensor is None:
        return None
    values = tensor.detach().float().cpu()
    if values.ndim != 5 or values.shape[0] < 1 or values.shape[1] < 1:
        return None

    target_frames, target_height, target_width = video_shape
    if target_frames <= 0 or target_height <= 0 or target_width <= 0:
        return None

    values = values[:, :1]
    finite = torch.isfinite(values)
    if not bool(finite.all().item()):
        values = torch.where(finite, values, torch.zeros_like(values))
    values = F.interpolate(
        values,
        size=(target_frames, target_height, target_width),
        mode="trilinear",
        align_corners=False,
    )
    return values[0, 0].numpy()


def _save_method1_pixel_overlay_gif(
    tensor,
    video_uint8,
    output_dir,
    name,
    global_step,
    sample_idx,
    max_frames,
    alpha=0.45,
    centered_at_one=False,
):
    if tensor is None or video_uint8 is None:
        return None

    frame_count = int(video_uint8.shape[0])
    height = int(video_uint8.shape[1])
    width = int(video_uint8.shape[2])
    map_np = _method1_upsample_map_to_video(tensor[sample_idx : sample_idx + 1], (frame_count, height, width))
    if map_np is None:
        return None

    if centered_at_one:
        vmin, center, vmax = _method1_centered_limits(map_np, center=1.0)
    else:
        positive = map_np[map_np > 0]
        vmax = float(np.percentile(positive, 99.0)) if positive.size else float(np.max(np.abs(map_np)) + 1e-8)
        vmin = 0.0
        center = None
    save_count = min(max(int(max_frames), 1), frame_count)
    if save_count >= frame_count:
        frame_indices = list(range(frame_count))
    else:
        frame_indices = np.linspace(0, frame_count - 1, save_count).round().astype(np.int64).tolist()

    case_dir = os.path.join(output_dir, "pixel_overlay", f"case-{sample_idx:02d}", name)
    os.makedirs(case_dir, exist_ok=True)

    overlay_images = []
    heat_images = []
    for out_idx, frame_idx in enumerate(frame_indices):
        base = video_uint8[frame_idx].astype(np.float32)
        if centered_at_one:
            heat = _method1_colorize_centered(map_np[frame_idx], vmin, center, vmax).astype(np.float32)
        else:
            heat = _method1_colorize(map_np[frame_idx], vmax).astype(np.float32)
        overlay = np.uint8(np.clip(base * (1.0 - alpha) + heat * alpha, 0, 255))
        overlay_image = Image.fromarray(overlay)
        heat_image = Image.fromarray(np.uint8(heat))
        overlay_image.save(
            os.path.join(case_dir, f"{name}_overlay_step{global_step}_case{sample_idx:02d}_f{frame_idx:03d}.png")
        )
        heat_image.save(
            os.path.join(case_dir, f"{name}_heat_step{global_step}_case{sample_idx:02d}_f{frame_idx:03d}.png")
        )
        overlay_images.append(overlay_image)
        heat_images.append(heat_image)

    if overlay_images:
        overlay_gif_path = os.path.join(case_dir, f"{name}_overlay_step{global_step}_case{sample_idx:02d}.gif")
        overlay_images[0].save(overlay_gif_path, save_all=True, append_images=overlay_images[1:], duration=120, loop=0)
        heat_gif_path = os.path.join(case_dir, f"{name}_heat_step{global_step}_case{sample_idx:02d}.gif")
        heat_images[0].save(heat_gif_path, save_all=True, append_images=heat_images[1:], duration=120, loop=0)

    return {
        "shape": list(tensor.detach().shape),
        "video_shape": list(video_uint8.shape),
        "saved_frame_indices": [int(idx) for idx in frame_indices],
        "finite": bool(torch.isfinite(tensor.detach()).all().item()),
        "min": float(np.min(map_np)),
        "mean": float(np.mean(map_np)),
        "max": float(np.max(map_np)),
        "vmin_p01": vmin,
        "center": center,
        "vmax_p99_positive": vmax,
    }


def _save_method1_source_video_gif(video_uint8, output_dir, global_step, sample_idx, max_frames):
    save_count = min(max(int(max_frames), 1), int(video_uint8.shape[0]))
    if save_count >= int(video_uint8.shape[0]):
        frame_indices = list(range(int(video_uint8.shape[0])))
    else:
        frame_indices = np.linspace(0, int(video_uint8.shape[0]) - 1, save_count).round().astype(np.int64).tolist()

    case_dir = os.path.join(output_dir, "pixel_overlay", f"case-{sample_idx:02d}", "source")
    os.makedirs(case_dir, exist_ok=True)
    images = []
    for frame_idx in frame_indices:
        image = Image.fromarray(video_uint8[frame_idx])
        image.save(os.path.join(case_dir, f"source_step{global_step}_case{sample_idx:02d}_f{frame_idx:03d}.png"))
        images.append(image)
    if images:
        gif_path = os.path.join(case_dir, f"source_step{global_step}_case{sample_idx:02d}.gif")
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=120, loop=0)
    return {
        "video_shape": list(video_uint8.shape),
        "saved_frame_indices": [int(idx) for idx in frame_indices],
    }


def _method1_compute_s_hat(effect_map, eps):
    if effect_map is None:
        return None
    values = effect_map.detach().float()
    if values.ndim != 5:
        return None
    future_values = values[:, :, 1:, :, :] if values.size(2) > 1 else values
    value_mean = future_values.mean(dim=(1, 2, 3, 4), keepdim=True).clamp_min(float(eps))
    return values / value_mean


def _save_method1_pixel_overlays(args, output_dir, global_step, maps, pixel_values, max_frames):
    if pixel_values is None:
        return None
    if pixel_values.ndim != 5:
        return None

    sample_count = min(
        int(pixel_values.shape[0]),
        max(int(args.method1_heatmap_max_cases), 1),
    )
    result = {}
    for sample_idx in range(sample_count):
        video_uint8 = _method1_pixel_values_to_uint8(pixel_values[sample_idx])
        if video_uint8 is None:
            continue
        sample_key = f"case-{sample_idx:02d}"
        result[sample_key] = {
            "source": _save_method1_source_video_gif(video_uint8, output_dir, global_step, sample_idx, max_frames),
            "maps": {},
        }
        for name, tensor in maps.items():
            centered_at_one = name == "S_action_effect"
            result[sample_key]["maps"][name] = _save_method1_pixel_overlay_gif(
                tensor,
                video_uint8,
                output_dir,
                name,
                global_step,
                sample_idx,
                max_frames,
                alpha=float(args.method1_heatmap_overlay_alpha),
                centered_at_one=centered_at_one,
            )
    return result


def save_method1_checkpoint_heatmaps(args, global_step, effect_map, noise_pred, target, rho, pixel_values=None):
    if not args.method1_heatmap_on_checkpoint:
        return

    heatmap_dir = args.method1_heatmap_dir or os.path.join(args.output_dir, "method1_heatmaps")
    output_dir = os.path.join(heatmap_dir, f"step-{global_step}")
    os.makedirs(output_dir, exist_ok=True)

    residual_map = None
    if noise_pred is not None and target is not None:
        residual_map = torch.linalg.vector_norm(
            noise_pred.detach().float() - target.detach().float(),
            ord=2,
            dim=1,
            keepdim=True,
        )

    max_frames = max(int(args.method1_heatmap_max_frames), 1)
    manifest = {
        "global_step": int(global_step),
        "maps": {},
    }
    s_hat_map = _method1_compute_s_hat(effect_map, args.method1_eps)
    maps = {
        "S_action_effect": s_hat_map,
        "E_residual": residual_map,
        "rho_weight": rho,
    }
    manifest["maps"]["S_action_effect"] = _save_method1_map_gif(
        s_hat_map, output_dir, "S_action_effect", global_step, max_frames, centered_at_one=True
    )
    if manifest["maps"]["S_action_effect"] is not None:
        manifest["maps"]["S_action_effect"]["quantity"] = "S_hat"
        manifest["maps"]["S_action_effect"]["normalization"] = (
            "per-sample mean over future latent tokens, S / (mu_b(S) + eps)"
        )
        manifest["maps"]["S_action_effect"]["color_scale"] = "centered diverging scale with center=1.0"
    manifest["maps"]["E_residual"] = _save_method1_map_gif(
        residual_map, output_dir, "E_residual", global_step, max_frames
    )
    manifest["maps"]["rho_weight"] = _save_method1_map_gif(
        rho, output_dir, "rho_weight", global_step, max_frames
    )
    if args.method1_heatmap_pixel_overlay:
        manifest["pixel_overlay"] = _save_method1_pixel_overlays(
            args,
            output_dir,
            global_step,
            maps,
            pixel_values,
            max_frames,
        )
    with open(os.path.join(output_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

warnings.filterwarnings("ignore", category=FutureWarning)

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

import datasets

from videox_fun.data.bucket_sampler import (ASPECT_RATIO_512,
                                            ASPECT_RATIO_RANDOM_CROP_512,
                                            ASPECT_RATIO_RANDOM_CROP_PROB,
                                            AspectRatioBatchImageVideoSampler,
                                            RandomSampler, get_closest_ratio)
from videox_fun.data.dataset_image_video import (ImageVideoDataset,
                                                 ImageVideoSampler,
                                                 get_random_mask,
                                                 process_pose_file,
                                                 process_pose_params)
from videox_fun.data.dataset_image_video_actionmap import ImageVideoControlDataset
from videox_fun.data.cap_dataset_presets import (
    CAP_DATASET_PRESETS,
    apply_cap_dataset_preset,
    preflight_cap_metadata,
)
from videox_fun.models import (AutoencoderKLWan, AutoencoderKLWan3_8,
                               CLIPModel, Wan2_2Transformer3DModel,
                               WanT5EncoderModel)
from videox_fun.training.method1_focused_loss import method1_focused_flow_loss
from videox_fun.training.cap_conditioning import (
    build_action_map_control_latents,
    build_poseanything_condition_latents,
    pack_camera_condition,
)
from videox_fun.training.cap_gradient_audit import local_shard_max_abs
from videox_fun.pipeline import Wan2_2FunControlPipeline
from videox_fun.utils.discrete_sampler import DiscreteSampling
from videox_fun.utils.utils import (calculate_dimensions, get_image_latent,
                                    get_image_to_video_latent,
                                    get_video_to_video_latent,
                                    save_videos_grid)

if is_wandb_available():
    import wandb


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def log_validation(vae, text_encoder, tokenizer, transformer3d, args, config, accelerator, weight_dtype, global_step):
    try:
        is_deepspeed = type(transformer3d).__name__ == 'DeepSpeedEngine'
        if is_deepspeed:
            origin_config = transformer3d.config
            transformer3d.config = accelerator.unwrap_model(transformer3d).config
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
            logger.info("Running validation... ")
            scheduler = FlowMatchEulerDiscreteScheduler(
                **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
            )
            if args.boundary_type == "full":
                transformer3d_1 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d
                transformer3d_2 = None
            else:
                if args.boundary_type == "low":
                    transformer3d_1 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d
                    
                    sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
                    transformer3d_2 = Wan2_2Transformer3DModel.from_pretrained(
                        os.path.join(args.pretrained_model_name_or_path, sub_path),
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                    ).to(weight_dtype)
                    
                else:
                    sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
                    transformer3d_1 = Wan2_2Transformer3DModel.from_pretrained(
                        os.path.join(args.pretrained_model_name_or_path, sub_path),
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
                    ).to(weight_dtype)

                    transformer3d_2 = accelerator.unwrap_model(transformer3d) if type(transformer3d).__name__ == 'DistributedDataParallel' else transformer3d

            pipeline = Wan2_2FunControlPipeline(
                vae=vae, 
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                transformer=transformer3d_1,
                transformer_2=transformer3d_2,
                scheduler=scheduler,
            )
            pipeline = pipeline.to(accelerator.device)

            if args.seed is None:
                generator = None
            else:
                rank_seed = args.seed + accelerator.process_index
                generator = torch.Generator(device=accelerator.device).manual_seed(rank_seed)
                logger.info(f"Rank {accelerator.process_index} using seed: {rank_seed}")

            for i in range(len(args.validation_prompts)):
                import cv2
                cap = cv2.VideoCapture(args.validation_paths[i])
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                width, height = calculate_dimensions(args.image_sample_size * args.image_sample_size,  width / height)
                video_length = int((args.video_sample_n_frames - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if args.video_sample_n_frames != 1 else 1
                
                inpaint_video, inpaint_video_mask, clip_image = get_image_to_video_latent(None, None, video_length=video_length, sample_size=[height, width])
                input_video, input_video_mask, ref_image, clip_image = get_video_to_video_latent(args.validation_paths[i], video_length=video_length, sample_size=[height, width])
                sample = pipeline(
                    args.validation_prompts[i], 
                    num_frames = video_length,
                    negative_prompt = "bad detailed",
                    height      = height,
                    width       = width,
                    generator   = generator,

                    control_video   = input_video,
                    video           = inpaint_video,
                    mask_video      = inpaint_video_mask,
                    num_inference_steps = 25,
                    guidance_scale      = 4.5,
                    boundary            = config['transformer_additional_kwargs'].get('boundary', 0.900)
                ).videos
                os.makedirs(os.path.join(args.logging_dir, "sample"), exist_ok=True)
                save_videos_grid(
                    sample, 
                    os.path.join(
                        args.logging_dir, 
                        f"sample/sample-{global_step}-rank{accelerator.process_index}-image-{i}.gif"
                    )
                )

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
            if not args.enable_text_encoder_in_dataloader:
                text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if is_deepspeed:
            transformer3d.config = origin_config
    except Exception as e:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"Eval error on rank {accelerator.process_index} with info {e}")
        vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
        if not args.enable_text_encoder_in_dataloader:
            text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--input_perturbation", type=float, default=0, help="The scale of input perturbation. Recommended 0.1."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. "
        ),
    )
    parser.add_argument(
        "--train_data_meta",
        type=str,
        default=None,
        help=(
            "A csv containing the training data. "
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="custom",
        choices=["custom", *CAP_DATASET_PRESETS.keys()],
        help="Select a registered CAP dataset; explicit train_data paths override the preset.",
    )
    parser.add_argument(
        "--action_injection",
        type=str,
        default="auto",
        choices=["auto", "none", "arm", "action_map", "camera", "poseanything"],
        help="Select exactly one action/control injection path.",
    )
    parser.add_argument(
        "--poseanything_resume_checkpoint",
        action="store_true",
        help="Allow a trained 96-channel PoseAnything checkpoint instead of expanding a 48-channel base model.",
    )
    parser.add_argument(
        "--skip_dataset_preflight",
        action="store_true",
        help="Skip the first-record metadata/path audit (not recommended).",
    )
    parser.add_argument(
        "--metadata_preflight_only",
        action="store_true",
        help="Audit dataset selection and exit before model/GPU initialization.",
    )
    parser.add_argument(
        "--dataset_max_retries",
        type=int,
        default=20,
        help="Fail a dataset item after this many replacement attempts instead of retrying forever.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "Truncate the metadata prefix to this many examples if set. The CAP Volcano launcher "
            "rounds its public prefix value up to the fixed effective batch multiple before passing it here."
        ),
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--validation_paths",
        type=str,
        default=None,
        nargs="+",
        help=("A set of control videos evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--use_came",
        action="store_true",
        help="whether to use came",
    )
    parser.add_argument(
        "--multi_stream",
        action="store_true",
        help="whether to use cuda multi-stream",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--vae_mini_batch", type=int, default=32, help="mini batch size for vae."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_model_info", action="store_true", help="Whether or not to report more info about model (such as norm, grad)."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_with_new_dataset",
        action="store_true",
        help=(
            "Resume model/optimizer state while starting a fresh sampler at the beginning of a "
            "deliberately changed dataset, then train num_train_epochs additional dataset epochs."
        ),
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=2000,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--snr_loss", action="store_true", help="Whether or not to use snr_loss."
    )
    parser.add_argument(
        "--uniform_sampling", action="store_true", help="Whether or not to use uniform_sampling."
    )
    parser.add_argument(
        "--enable_text_encoder_in_dataloader", action="store_true", help="Whether or not to use text encoder in dataloader."
    )
    parser.add_argument(
        "--enable_bucket", action="store_true", help="Whether enable bucket sample in datasets."
    )
    parser.add_argument(
        "--random_ratio_crop", action="store_true", help="Whether enable random ratio crop sample in datasets."
    )
    parser.add_argument(
        "--random_frame_crop", action="store_true", help="Whether enable random frame crop sample in datasets."
    )
    parser.add_argument(
        "--random_hw_adapt", action="store_true", help="Whether enable random adapt height and width in datasets."
    )
    parser.add_argument(
        "--training_with_video_token_length", action="store_true", help="The training stage of the model in training.",
    )
    parser.add_argument(
        "--auto_tile_batch_size", action="store_true", help="Whether to auto tile batch size.",
    )
    parser.add_argument(
        "--motion_sub_loss", action="store_true", help="Whether enable motion sub loss."
    )
    parser.add_argument(
        "--motion_sub_loss_ratio", type=float, default=0.25, help="The ratio of motion sub loss."
    )
    parser.add_argument(
        "--train_sampling_steps",
        type=int,
        default=1000,
        help="Run train_sampling_steps.",
    )
    parser.add_argument(
        "--keep_all_node_same_token_length",
        action="store_true", 
        help="Reference of the length token.",
    )
    parser.add_argument(
        "--token_sample_size",
        type=int,
        default=512,
        help="Sample size of the token.",
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Sample size of the video.",
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Sample size of the image.",
    )
    parser.add_argument(
        "--fix_sample_size", 
        nargs=2, type=int, default=None,
        help="Fix Sample size [height, width] when using bucket and collate_fn."
    )
    parser.add_argument(
        "--require_input_resolution",
        nargs=2,
        type=int,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Fail on the first batch unless pixel_values exactly match HEIGHT x WIDTH.",
    )
    parser.add_argument(
        "--video_sample_stride",
        type=int,
        default=4,
        help="Sample stride of the video.",
    )
    parser.add_argument(
        "--video_sample_n_frames",
        type=int,
        default=17,
        help="Num frame of video.",
    )
    parser.add_argument(
        "--video_repeat",
        type=int,
        default=0,
        help="Num of repeat video.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "The config of the model in training."
        ),
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other transformers, input its path."),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("If you want to load the weight from other vaes, input its path."),
    )

    parser.add_argument(
        '--trainable_modules', 
        nargs='+', 
        help='Enter a list of trainable modules'
    )
    parser.add_argument(
        '--trainable_modules_low_learning_rate', 
        nargs='+', 
        default=[],
        help='Enter a list of trainable modules with lower learning rate'
    )
    parser.add_argument(
        "--require_all_transformer_trainable",
        action="store_true",
        help="Fail before optimizer creation unless every Transformer parameter is trainable.",
    )
    parser.add_argument(
        "--require_selective_ffn_arm_trainable",
        action="store_true",
        help=(
            "Fail before optimizer creation unless the historical selective configuration is exact: "
            "Dense FFN, arm action modules, arm condition embeddings, and control adapter only."
        ),
    )
    parser.add_argument(
        "--moe_mode",
        type=str,
        default="camera_kinematic",
        choices=["camera_kinematic", "control_expert"],
        help="MoE injection mode for transformer FFNs.",
    )
    parser.add_argument(
        "--disable_moe",
        action="store_true",
        help="Disable external MoE injection while keeping action map conditioning logic enabled.",
    )
    parser.add_argument(
        "--moe_all_blocks",
        action="store_true",
        help="Inject MoE into all transformer blocks instead of the default subset for the selected mode.",
    )
    parser.add_argument(
        "--moe_route_temperature",
        type=float,
        default=1.0,
        help="Softmax temperature used by the explicit control expert MoE routing weights.",
    )
    parser.add_argument(
        "--camera_moe_root",
        type=str,
        default=os.environ.get("CAMERA_MOE_ROOT", ""),
        help="Directory containing camera_moe_core.py for external MoE injection.",
    )
    parser.add_argument(
        '--tokenizer_max_length', 
        type=int,
        default=512,
        help='Max length of tokenizer'
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", help="Whether or not to use deepspeed."
    )
    parser.add_argument(
        "--use_fsdp", action="store_true", help="Whether or not to use fsdp."
    )
    parser.add_argument(
        "--low_vram", action="store_true", help="Whether enable low_vram mode."
    )
    parser.add_argument(
        "--freeze_control_adapter", action="store_true",
        help="Freeze the control_adapter (camera) module so it is not trained."
    )
    parser.add_argument(
        "--zero_init_camera_adapter_output",
        action="store_true",
        help="Add a zero-initialized camera adapter output projection.",
    )
    parser.add_argument(
        "--require_camera_adapter_zero_init",
        action="store_true",
        help="Fail unless the camera adapter output projection is exactly zero initialized.",
    )
    parser.add_argument(
        "--enable_arm_info", action="store_true",
        help="Enable robotic arm action conditioning from dataset metadata."
    )
    parser.add_argument(
        "--zero_init_arm_action_output",
        action="store_true",
        help="Zero-initialize both arm ActionMLP output layers so step-0 action injection is exactly zero."
    )
    parser.add_argument(
        "--require_arm_action_zero_init",
        action="store_true",
        help="Fail before optimizer creation unless arm ActionMLP outputs are zero and their input layers are nonzero."
    )
    parser.add_argument(
        "--enable_action_map_info", action="store_true",
        help="Enable action map / pose-video conditioning from dataset metadata."
    )
    parser.add_argument(
        "--arm_action_stat_path",
        type=str,
        default=None,
        help="Path to robotic arm action normalization statistics."
    )
    parser.add_argument(
        "--arm_action_key",
        type=str,
        default="state",
        help="Key used to read robotic arm action values from JSON annotations."
    )
    parser.add_argument(
        "--arm_action_dim",
        type=int,
        default=14,
        help="Dimension of each robotic arm action vector."
    )
    parser.add_argument(
        "--arm_action_num_frames",
        type=int,
        default=None,
        help="Fixed number of frames used by the robotic arm action embedder."
    )
    parser.add_argument(
        "--enable_method1_focused_loss",
        action="store_true",
        help="Enable method1 action-effect focused flow-matching loss."
    )
    parser.add_argument(
        "--method1_loss_variant",
        type=str,
        default="CAER",
        choices=["MSE", "CAER"],
        help=(
            "MSE uses rho=1; CAER uses the validated action-effect weighting."
        ),
    )
    parser.add_argument(
        "--method1_action_dropout_prob",
        type=float,
        default=0.10,
        help="Probability of dropping action/control conditioning for method1 null-condition branch training."
    )
    parser.add_argument(
        "--method1_tau_s",
        type=float,
        default=0.50,
        help="Target sigma/noise mixing ratio used as the fixed medium-noise step for method1 action-effect maps."
    )
    parser.add_argument(
        "--method1_eps",
        type=float,
        default=1e-6,
        help="Numerical epsilon for method1 per-sample normalization."
    )
    parser.add_argument(
        "--method1_mse_threshold",
        type=float,
        default=0.0,
        help="Optional absolute residual threshold for method1 squared error. <=0 disables thresholding."
    )
    parser.add_argument(
        "--method1_log_stats",
        action="store_true",
        help="Log lightweight method1 weight/dropout statistics in the progress logs."
    )
    parser.add_argument(
        "--method1_skip_nonfinite_updates",
        action="store_true",
        help=(
            "Discard a whole accumulated optimizer update when its clipped "
            "gradient norm is non-finite. Finite updates are unchanged."
        ),
    )
    parser.add_argument(
        "--method1_max_nonfinite_update_skips",
        type=int,
        default=10,
        help=(
            "Maximum non-finite optimizer updates to discard when "
            "--method1_skip_nonfinite_updates is enabled."
        ),
    )
    parser.add_argument(
        "--boundary_type",
        type=str,
        default="low",
        help=(
            'The format of training data. Support `"low"` and `"high"`'
        ),
    )
    parser.add_argument(
        "--abnormal_norm_clip_start",
        type=int,
        default=1000,
        help=(
            'When do we start doing additional processing on abnormal gradients. '
        ),
    )
    parser.add_argument(
        "--initial_grad_norm_ratio",
        type=int,
        default=5,
        help=(
            'The initial gradient is relative to the multiple of the max_grad_norm. '
        ),
    )
    parser.add_argument(
        "--train_mode",
        type=str,
        default="control",
        help=(
            'The format of training data. Support `"control"`'
            ' (default), `"control_ref"`, `"control_camera_ref"`.'
        ),
    )
    parser.add_argument(
        "--control_ref_image",
        type=str,
        default="first_frame",
        help=(
            'The format of training data. Support `"first_frame"`'
            ' (default), `"random"`.'
        ),
    )
    parser.add_argument(
        "--add_full_ref_image_in_self_attention",
        action="store_true",
        help=(
            'Whether enable add full ref image in self attention.'
        ),
    )
    parser.add_argument(
        "--add_inpaint_info",
        action="store_true",
        help=(
            'Whether enable add inpaint info in self attention.'
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--benchmark_timing_path",
        type=str,
        default=None,
        help=(
            "Optional JSONL path for closed-file per-update metrics: globally reduced weighted/uniform "
            "losses, learning rate, timing, and main-rank peak CUDA memory."
        ),
    )
    parser.add_argument(
        "--method1_sample_loss_dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for per-sample Method1 loss visits, per-epoch loss rankings, "
            "and adjacent-epoch loss-drop rankings."
        ),
    )
    parser.add_argument(
        "--require_method1_sample_loss_recording",
        action="store_true",
        help="Fail unless per-sample Method1 loss recording is configured and active.",
    )
    parser.add_argument(
        "--skip_sanity_check",
        action="store_true",
        help="Skip first-batch GIF/PNG sanity check generation.",
    )
    parser.add_argument(
        "--skip_final_checkpoint",
        action="store_true",
        help="Skip the final checkpoint save at training end.",
    )
    parser.add_argument(
        "--method1_force_exit_after_training",
        action="store_true",
        help="Exit immediately after reaching max_train_steps. Intended only for short method1 smoke tests.",
    )
    parser.add_argument(
        "--require_cap_condition_gradient",
        action="store_true",
        help=(
            "Require a finite, nonzero first-backward gradient on the parameters that consume "
            "the selected CAP condition. Intended as a smoke-test gate."
        ),
    )
    parser.add_argument(
        "--method1_heatmap_on_checkpoint",
        action="store_true",
        help="Save latent-grid Method1 S/E/rho heatmaps whenever a checkpoint is saved.",
    )
    parser.add_argument(
        "--method1_heatmap_dir",
        type=str,
        default=None,
        help="Directory for Method1 checkpoint heatmaps. Defaults to output_dir/method1_heatmaps.",
    )
    parser.add_argument(
        "--method1_heatmap_max_frames",
        type=int,
        default=17,
        help="Maximum latent time frames to save for each Method1 checkpoint heatmap.",
    )
    parser.add_argument(
        "--method1_heatmap_pixel_overlay",
        action="store_true",
        help="Save Method1 heatmaps upsampled to source video pixel space and overlaid on RGB frames.",
    )
    parser.add_argument(
        "--method1_heatmap_max_cases",
        type=int,
        default=1,
        help="Maximum batch samples to save for each Method1 heatmap export.",
    )
    parser.add_argument(
        "--method1_heatmap_overlay_alpha",
        type=float,
        default=0.45,
        help="Alpha used when blending Method1 heatmaps over source RGB frames.",
    )
    parser.add_argument(
        "--method1_heatmap_every_steps",
        type=int,
        default=0,
        help="If positive, save Method1 heatmaps every N optimizer steps without requiring a checkpoint save.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def adapt_action_map_moe_state_dict(state_dict, model_state_dict):
    adapted_state_dict = dict(state_dict)

    for key, value in list(adapted_state_dict.items()):
        if not key.endswith(".ffn.control_moe.router.weight") or key not in model_state_dict:
            continue
        target_value = model_state_dict[key]
        if value.shape == target_value.shape:
            continue
        if value.ndim == 2 and value.shape[0] == 3 and target_value.ndim == 2 and target_value.shape[0] == 4 and value.shape[1] == target_value.shape[1]:
            new_value = torch.zeros_like(target_value, device=value.device, dtype=value.dtype)
            new_value[:3] = value
            adapted_state_dict[key] = new_value

    for key, target_value in model_state_dict.items():
        if ".ffn.control_moe.action_map_expert." not in key or key in adapted_state_dict:
            continue
        source_key = key.replace(".ffn.control_moe.action_map_expert.", ".ffn.control_moe.shared_expert.")
        source_value = adapted_state_dict.get(source_key, None)
        if source_value is not None and source_value.shape == target_value.shape:
            adapted_state_dict[key] = source_value

    return adapted_state_dict


def main():
    args = apply_cap_dataset_preset(parse_args())

    if args.resume_with_new_dataset and not args.resume_from_checkpoint:
        raise ValueError("--resume_with_new_dataset requires --resume_from_checkpoint")
    if args.resume_with_new_dataset and args.max_train_steps is None:
        raise ValueError(
            "--resume_with_new_dataset requires an absolute --max_train_steps target"
        )
    resume_new_dataset_epochs = (
        args.num_train_epochs if args.resume_with_new_dataset else None
    )

    if not args.skip_dataset_preflight:
        preflight_summary = preflight_cap_metadata(
            args.train_data_meta,
            args.train_data_dir,
            args.action_injection,
            check_paths=True,
        )
        print("CAP dataset preflight: " + json.dumps(preflight_summary, ensure_ascii=True, sort_keys=True))
    if args.metadata_preflight_only:
        return

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    args.logging_dir = logging_dir

    config = OmegaConf.load(args.config_path)
    if args.arm_action_num_frames is None:
        args.arm_action_num_frames = args.video_sample_n_frames

    if "transformer_additional_kwargs" not in config or config["transformer_additional_kwargs"] is None:
        config["transformer_additional_kwargs"] = OmegaConf.create()

    if args.enable_arm_info:
        config["transformer_additional_kwargs"]["add_arm_action_embedder"] = True
        config["transformer_additional_kwargs"]["arm_action_dim"] = args.arm_action_dim
        config["transformer_additional_kwargs"]["arm_action_num_frames"] = args.arm_action_num_frames
        config["transformer_additional_kwargs"]["zero_init_arm_action_output"] = bool(
            args.zero_init_arm_action_output
        )
    elif args.require_arm_action_zero_init:
        raise ValueError("--require_arm_action_zero_init requires --enable_arm_info")

    if args.action_injection in {"arm", "action_map", "camera"}:
        config["transformer_additional_kwargs"]["in_dim"] = 100
        config["transformer_additional_kwargs"]["in_channels"] = 100
        config["transformer_additional_kwargs"]["add_control_adapter"] = True
        config["transformer_additional_kwargs"]["in_dim_control_adapter"] = 24
        config["transformer_additional_kwargs"]["downscale_factor_control_adapter"] = 16
        config["transformer_additional_kwargs"]["cap_expected_patch_embedding_source_channels"] = [48, 100]
    elif args.action_injection == "poseanything":
        config["transformer_additional_kwargs"]["in_dim"] = 96
        config["transformer_additional_kwargs"]["in_channels"] = 96
        config["transformer_additional_kwargs"]["add_control_adapter"] = False
        # The base model is always 48-channel; a trained 96-channel checkpoint is loaded later.
        config["transformer_additional_kwargs"]["cap_expected_patch_embedding_source_channels"] = [48]

    if args.action_injection == "camera":
        config["transformer_additional_kwargs"][
            "zero_init_control_adapter_output"
        ] = bool(args.zero_init_camera_adapter_output)
        if (
            args.require_camera_adapter_zero_init
            and not args.zero_init_camera_adapter_output
        ):
            raise ValueError(
                "--require_camera_adapter_zero_init requires "
                "--zero_init_camera_adapter_output"
            )
    elif args.require_camera_adapter_zero_init:
        raise ValueError(
            "--require_camera_adapter_zero_init requires --action_injection=camera"
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    fsdp_plugin = None
    if args.use_fsdp:
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy="FULL_SHARD",
            backward_prefetch="BACKWARD_PRE",
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=["WanAttentionBlock"],
            state_dict_type="SHARDED_STATE_DICT",
            cpu_ram_efficient_loading=False,
            use_orig_params=True,
        )

    accelerator_kwargs_handlers = []
    process_group_timeout_seconds = os.environ.get("VIDEOX_PROCESS_GROUP_TIMEOUT_SECONDS")
    if process_group_timeout_seconds:
        accelerator_kwargs_handlers.append(
            InitProcessGroupKwargs(timeout=timedelta(seconds=int(process_group_timeout_seconds)))
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=accelerator_kwargs_handlers,
    )

    deepspeed_plugin = accelerator.state.deepspeed_plugin if hasattr(accelerator.state, "deepspeed_plugin") else None
    fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
    if deepspeed_plugin is not None:
        zero_stage = int(deepspeed_plugin.zero_stage)
        fsdp_stage = 0
        print(f"Using DeepSpeed Zero stage: {zero_stage}")

        args.use_deepspeed = True
        if zero_stage == 3:
            print(f"Auto set save_state to True because zero_stage == 3")
            args.save_state = True
    elif fsdp_plugin is not None:
        from torch.distributed.fsdp import ShardingStrategy
        zero_stage = 0
        if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
            fsdp_stage = 3
        elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
            fsdp_stage = 2
        else:
            fsdp_stage = 0
        print(f"Using FSDP stage: {fsdp_stage}")

        args.use_fsdp = True
        if fsdp_stage == 3:
            print(f"Auto set save_state to True because fsdp_stage == 3")
            args.save_state = True
    else:
        zero_stage = 0
        fsdp_stage = 0
        print("DeepSpeed is not enabled.")

    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=logging_dir)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        rng = np.random.default_rng(np.random.PCG64(args.seed + accelerator.process_index))
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
    else:
        rng = None
        torch_rng = None
    method1_rng_seed = (
        (args.seed if args.seed is not None else torch.initial_seed())
        + accelerator.process_index
        + 1_000_003
    )
    method1_torch_rng = torch.Generator(accelerator.device).manual_seed(
        method1_rng_seed
    )
    index_rng = np.random.default_rng(np.random.PCG64(43))
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Load scheduler, tokenizer and models.
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        # Get Text encoder
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
        text_encoder = text_encoder.eval()
        # Get Vae
        Chosen_AutoencoderKL = {
            "AutoencoderKLWan": AutoencoderKLWan,
            "AutoencoderKLWan3_8": AutoencoderKLWan3_8
        }[config['vae_kwargs'].get('vae_type', 'AutoencoderKLWan')]
        vae = Chosen_AutoencoderKL.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )
        vae.eval()
            
    # Get Transformer
    if args.boundary_type == "low" or args.boundary_type == "full":
        sub_path = config['transformer_additional_kwargs'].get('transformer_low_noise_model_subpath', 'transformer')
    else:
        sub_path = config['transformer_additional_kwargs'].get('transformer_high_noise_model_subpath', 'transformer')
    transformer3d = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(args.pretrained_model_name_or_path, sub_path),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    ).to(weight_dtype)

    patch_channels = int(transformer3d.patch_embedding.in_channels)
    expected_patch_channels = 96 if args.action_injection == "poseanything" else 100
    if args.action_injection != "none" and patch_channels != expected_patch_channels:
        raise RuntimeError(
            f"{args.action_injection} requires patch_embedding.in_channels="
            f"{expected_patch_channels}, got {patch_channels}."
        )
    if args.action_injection == "poseanything" and not args.poseanything_resume_checkpoint:
        video_weight_max = transformer3d.patch_embedding.weight[:, :48].detach().abs().max().item()
        skeleton_weight_max = transformer3d.patch_embedding.weight[:, 48:].detach().abs().max().item()
        pose_init_ok = video_weight_max > 0.0 and skeleton_weight_max == 0.0
        accelerator.print(
            "PoseAnything patch initialization audit: "
            f"video_weight_max={video_weight_max:.9g} "
            f"skeleton_weight_max={skeleton_weight_max:.9g} passed={int(pose_init_ok)}"
        )
        if not pose_init_ok:
            raise RuntimeError("PoseAnything 48->96 patch expansion audit failed.")

    if not args.disable_moe:
        # --- INJECT MOE HERE ---
        import sys
        if args.camera_moe_root and args.camera_moe_root not in sys.path:
            sys.path.append(args.camera_moe_root)
        from camera_moe_core import inject_moe_into_wan_model
        moe_target_block_indices = list(range(len(transformer3d.blocks))) if args.moe_all_blocks else None
        transformer3d = inject_moe_into_wan_model(
            transformer3d,
            target_block_indices=moe_target_block_indices,
            moe_mode=args.moe_mode,
            route_temperature=args.moe_route_temperature,
        )
        # -----------------------

    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer3d.requires_grad_(False)

    if args.transformer_path is not None:
        print(f"From checkpoint: {args.transformer_path}")
        if args.transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.transformer_path)
        else:
            state_dict = torch.load(args.transformer_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict
        if args.action_injection == "poseanything":
            patch_weight = state_dict.get("patch_embedding.weight")
            if patch_weight is None or patch_weight.ndim != 5 or patch_weight.size(1) != 96:
                raise RuntimeError(
                    "--poseanything_resume_checkpoint requires a checkpoint containing "
                    "patch_embedding.weight with 96 input channels."
                )
        state_dict = adapt_action_map_moe_state_dict(state_dict, transformer3d.state_dict())

        m, u = transformer3d.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.vae_path is not None:
        print(f"From checkpoint: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    if args.enable_arm_info:
        action_mlps = (
            transformer3d.arm_action_embedder,
            transformer3d.arm_action_embedder_proj,
        )
        fc2_weight_max_abs = max(
            module.fc2.weight.detach().abs().max().item() for module in action_mlps
        )
        fc2_bias_max_abs = max(
            module.fc2.bias.detach().abs().max().item() for module in action_mlps
        )
        fc1_min_of_max_abs = min(
            module.fc1.weight.detach().abs().max().item() for module in action_mlps
        )
        mask_max_abs = max(
            transformer3d.arm_condition_mask_emb.weight.detach().abs().max().item(),
            transformer3d.arm_condition_mask_emb_proj.weight.detach().abs().max().item(),
        )
        config_zero_init = bool(
            getattr(transformer3d.config, "zero_init_arm_action_output", False)
        )
        zero_init_ok = (
            config_zero_init
            and fc2_weight_max_abs == 0.0
            and fc2_bias_max_abs == 0.0
            and mask_max_abs == 0.0
            and fc1_min_of_max_abs > 0.0
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Arm action zero-init audit: "
                f"requested={int(args.zero_init_arm_action_output)} "
                f"config={int(config_zero_init)} "
                f"fc2_weight_max_abs={fc2_weight_max_abs:.9g} "
                f"fc2_bias_max_abs={fc2_bias_max_abs:.9g} "
                f"mask_max_abs={mask_max_abs:.9g} "
                f"fc1_min_of_max_abs={fc1_min_of_max_abs:.9g} "
                f"passed={int(zero_init_ok)}"
            )
        if args.require_arm_action_zero_init and not zero_init_ok:
            raise RuntimeError("Required arm action zero-output initialization audit failed.")

    if args.action_injection == "camera":
        output_conv = getattr(transformer3d.control_adapter, "output_conv", None)
        output_weight_max_abs = (
            output_conv.weight.detach().abs().max().item()
            if output_conv is not None
            else float("inf")
        )
        output_bias_max_abs = (
            output_conv.bias.detach().abs().max().item()
            if output_conv is not None and output_conv.bias is not None
            else float("inf")
        )
        feature_weight_max_abs = max(
            transformer3d.control_adapter.conv.weight.detach().abs().max().item(),
            max(
                parameter.detach().abs().max().item()
                for name, parameter in transformer3d.control_adapter.named_parameters()
                if name.startswith("residual_blocks") and name.endswith("weight")
            ),
        )
        config_zero_init = bool(
            getattr(
                transformer3d.config,
                "zero_init_control_adapter_output",
                False,
            )
        )
        camera_zero_init_ok = (
            config_zero_init
            and output_conv is not None
            and output_weight_max_abs == 0.0
            and output_bias_max_abs == 0.0
            and feature_weight_max_abs > 0.0
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Camera adapter zero-init audit: "
                f"requested={int(args.zero_init_camera_adapter_output)} "
                f"config={int(config_zero_init)} "
                f"output_weight_max_abs={output_weight_max_abs:.9g} "
                f"output_bias_max_abs={output_bias_max_abs:.9g} "
                f"feature_weight_max_abs={feature_weight_max_abs:.9g} "
                f"passed={int(camera_zero_init_ok)}"
            )
        if args.require_camera_adapter_zero_init and not camera_zero_init_ok:
            raise RuntimeError(
                "Required camera adapter zero-output initialization audit failed."
            )
    
    # A good trainable modules is showed below now.
    # For 3D Patch: trainable_modules = ['ff.net', 'pos_embed', 'attn2', 'proj_out', 'timepositionalencoding', 'h_position', 'w_position']
    # For 2D Patch: trainable_modules = ['ff.net', 'attn2', 'timepositionalencoding', 'h_position', 'w_position']
    transformer3d.train()
    if accelerator.is_main_process:
        accelerator.print(
            f"Trainable modules '{args.trainable_modules}'."
        )
    for name, param in transformer3d.named_parameters():
        for trainable_module_name in args.trainable_modules + args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                param.requires_grad = True
                break

    if args.freeze_control_adapter:
        frozen_count = 0
        for name, param in transformer3d.named_parameters():
            if "control_adapter" in name:
                param.requires_grad = False
                frozen_count += 1
        if accelerator.is_main_process:
            accelerator.print(f"Froze {frozen_count} control_adapter parameters.")

    transformer_total_params = sum(param.numel() for param in transformer3d.parameters())
    transformer_trainable_params = sum(
        param.numel() for param in transformer3d.parameters() if param.requires_grad
    )
    transformer_frozen_names = [
        name for name, param in transformer3d.named_parameters() if not param.requires_grad
    ]
    if accelerator.is_main_process:
        accelerator.print(
            "Transformer parameter audit: "
            f"trainable={transformer_trainable_params} "
            f"total={transformer_total_params} "
            f"frozen_tensors={len(transformer_frozen_names)}"
        )
    if args.require_all_transformer_trainable and transformer_frozen_names:
        frozen_preview = ", ".join(transformer_frozen_names[:20])
        raise RuntimeError(
            "Full-parameter training was required, but "
            f"{len(transformer_frozen_names)} Transformer tensors remain frozen. "
            f"First frozen tensors: {frozen_preview}"
        )
    if args.require_selective_ffn_arm_trainable:
        expected_main_selectors = [
            "ffn",
            "arm_action_embedder",
            "arm_condition_mask_emb",
        ]
        expected_low_lr_selectors = [
            "ffn.control_moe.shared_expert",
            "control_adapter",
        ]
        all_selectors = expected_main_selectors + expected_low_lr_selectors
        selector_match = (
            args.trainable_modules == expected_main_selectors
            and args.trainable_modules_low_learning_rate == expected_low_lr_selectors
            and not args.freeze_control_adapter
        )
        selected_names = {
            name
            for name, _ in transformer3d.named_parameters()
            if any(selector in name for selector in all_selectors)
        }
        trainable_names = {
            name for name, param in transformer3d.named_parameters() if param.requires_grad
        }
        unexpected_trainable = sorted(trainable_names - selected_names)
        missing_selected = sorted(selected_names - trainable_names)
        required_groups = {
            "dense_ffn": any(".ffn." in name for name in trainable_names),
            "arm_action": all(
                any(name.startswith(prefix) for name in trainable_names)
                for prefix in ("arm_action_embedder.", "arm_action_embedder_proj.")
            ),
            "arm_mask": all(
                any(name.startswith(prefix) for name in trainable_names)
                for prefix in ("arm_condition_mask_emb.", "arm_condition_mask_emb_proj.")
            ),
            "control_adapter": any(name.startswith("control_adapter.") for name in trainable_names),
        }
        selective_audit_ok = (
            selector_match
            and transformer_trainable_params < transformer_total_params
            and len(transformer_frozen_names) > 0
            and not unexpected_trainable
            and not missing_selected
            and all(required_groups.values())
        )
        if accelerator.is_main_process:
            accelerator.print(
                "Selective FFN parameter audit: "
                f"requested=1 selector_match={int(selector_match)} "
                f"trainable={transformer_trainable_params} "
                f"total={transformer_total_params} "
                f"frozen_tensors={len(transformer_frozen_names)} "
                f"unexpected_trainable={len(unexpected_trainable)} "
                f"missing_selected={len(missing_selected)} "
                f"required_groups={int(all(required_groups.values()))} "
                f"passed={int(selective_audit_ok)}"
            )
        if not selective_audit_ok:
            raise RuntimeError(
                "Required selective FFN/arm/control-adapter parameter audit failed. "
                f"unexpected trainable: {unexpected_trainable[:20]}; "
                f"missing selected: {missing_selected[:20]}; "
                f"required groups: {required_groups}"
            )

    cap_condition_gradient_audit = {
        "values": [],
        "checked": False,
        "handles": [],
        "parameter_names": [],
        "parameters": [],
    }
    if args.require_cap_condition_gradient:
        if args.action_injection == "none":
            raise ValueError(
                "--require_cap_condition_gradient requires a non-none --action_injection."
            )

        def _record_condition_gradient(grad, channel_slice=None):
            selected = grad
            if channel_slice is not None:
                if grad.ndim != 5:
                    cap_condition_gradient_audit["values"].append(
                        grad.new_tensor(float("nan"), dtype=torch.float32)
                    )
                    return
                selected = grad[:, channel_slice]
            cap_condition_gradient_audit["values"].append(
                selected.detach().abs().amax().to(dtype=torch.float32)
            )

        for parameter_name, parameter in transformer3d.named_parameters():
            channel_slice = None
            selected = False
            if args.action_injection == "action_map" and parameter_name == "patch_embedding.weight":
                channel_slice = slice(52, 100)
                selected = True
            elif args.action_injection == "poseanything" and parameter_name == "patch_embedding.weight":
                channel_slice = slice(48, 96)
                selected = True
            elif args.action_injection == "camera" and parameter_name.startswith("control_adapter."):
                selected = True
            elif (
                args.action_injection == "arm"
                and parameter_name.startswith(
                    ("arm_action_embedder.", "arm_action_embedder_proj.")
                )
                and ".fc2." in parameter_name
            ):
                selected = True

            if selected:
                if not parameter.requires_grad:
                    raise RuntimeError(
                        "CAP condition-gradient audit selected a frozen parameter: "
                        f"{parameter_name}"
                    )
                cap_condition_gradient_audit["parameter_names"].append(parameter_name)
                cap_condition_gradient_audit["parameters"].append(
                    {
                        "parameter": parameter,
                        "full_shape": tuple(parameter.shape),
                        "channel_slice": channel_slice,
                    }
                )
                if not args.use_fsdp:
                    cap_condition_gradient_audit["handles"].append(
                        parameter.register_hook(
                            lambda grad, local_slice=channel_slice: _record_condition_gradient(
                                grad, local_slice
                            )
                        )
                    )

        if not cap_condition_gradient_audit["parameter_names"]:
            raise RuntimeError(
                "CAP condition-gradient audit found no parameters for "
                f"action_injection={args.action_injection}."
            )
        if accelerator.is_main_process:
            accelerator.print(
                "CAP condition gradient audit armed: "
                f"mode={args.action_injection} "
                f"parameters={len(cap_condition_gradient_audit['parameter_names'])}"
            )

    # Create EMA for the transformer3d.
    if args.use_ema:
        if zero_stage == 3:
            raise NotImplementedError("FSDP does not support EMA.")

        ema_transformer3d = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(args.pretrained_model_name_or_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        ).to(weight_dtype)

        ema_transformer3d = EMAModel(ema_transformer3d.parameters(), model_cls=Wan2_2Transformer3DModel, model_config=ema_transformer3d.config)

    checkpoint_epoch = 0

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        if fsdp_stage != 0 or zero_stage == 3:
            def save_model_hook(models, weights, output_dir):
                save_extra_safetensors = os.environ.get("VIDEOX_SAVE_FSDP_EXTRA_SAFETENSORS", "0") == "1"
                if save_extra_safetensors:
                    accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
                if accelerator.is_main_process:
                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, checkpoint_epoch], file)

                    if save_extra_safetensors:
                        from safetensors.torch import save_file

                        safetensor_save_path = os.path.join(output_dir, f"diffusion_pytorch_model.safetensors")
                        try:
                            accelerate_state_dict = {k: v.to(dtype=weight_dtype) for k, v in accelerate_state_dict.items()}
                            save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})
                        except Exception as exc:
                            if os.path.exists(safetensor_save_path):
                                try:
                                    os.remove(safetensor_save_path)
                                except OSError:
                                    pass
                            logger.warning(
                                "Skipping optional FSDP safetensors export to %s after save error: %s",
                                safetensor_save_path,
                                exc,
                            )

            def load_model_hook(models, input_dir):
                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        if args.resume_with_new_dataset:
                            batch_sampler.sampler._pos_start = 0
                        else:
                            batch_sampler.sampler._pos_start = max(
                                loaded_number
                                - args.dataloader_num_workers
                                * accelerator.num_processes
                                * 2
                                * args.train_batch_size,
                                0,
                            )
                    print(
                        f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}. "
                        f"sampler_pos_start={batch_sampler.sampler._pos_start} "
                        f"new_dataset={int(args.resume_with_new_dataset)}."
                    )
        else:
            # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
            def save_model_hook(models, weights, output_dir):
                if accelerator.is_main_process:
                    if args.use_ema:
                        ema_transformer3d.save_pretrained(os.path.join(output_dir, "transformer_ema"))

                    models[0].save_pretrained(os.path.join(output_dir, "transformer"))
                    if not args.use_deepspeed:
                        weights.pop()

                    with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                        pickle.dump([batch_sampler.sampler._pos_start, checkpoint_epoch], file)

            def load_model_hook(models, input_dir):
                if args.use_ema:
                    ema_path = os.path.join(input_dir, "transformer_ema")
                    _, ema_kwargs = Wan2_2Transformer3DModel.load_config(ema_path, return_unused_kwargs=True)
                    load_model = Wan2_2Transformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer_ema",
                        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs'])
                    )
                    load_model = EMAModel(load_model.parameters(), model_cls=Wan2_2Transformer3DModel, model_config=load_model.config)
                    load_model.load_state_dict(ema_kwargs)

                    ema_transformer3d.load_state_dict(load_model.state_dict())
                    ema_transformer3d.to(accelerator.device)
                    del load_model

                for i in range(len(models)):
                    # pop models so that they are not loaded again
                    model = models.pop()

                    # load diffusers style into model
                    load_model = Wan2_2Transformer3DModel.from_pretrained(
                        input_dir, subfolder="transformer"
                    )
                    model.register_to_config(**load_model.config)

                    model.load_state_dict(load_model.state_dict())
                    del load_model

                pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as file:
                        loaded_number, _ = pickle.load(file)
                        if args.resume_with_new_dataset:
                            batch_sampler.sampler._pos_start = 0
                        else:
                            batch_sampler.sampler._pos_start = max(
                                loaded_number
                                - args.dataloader_num_workers
                                * accelerator.num_processes
                                * 2
                                * args.train_batch_size,
                                0,
                            )
                    print(
                        f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}. "
                        f"sampler_pos_start={batch_sampler.sampler._pos_start} "
                        f"new_dataset={int(args.resume_with_new_dataset)}."
                    )

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except Exception:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_params = list(filter(lambda p: p.requires_grad, transformer3d.parameters()))
    trainable_params_optim = [
        {'params': [], 'lr': args.learning_rate},
        {'params': [], 'lr': args.learning_rate / 2},
    ]
    debug_lr_assignment = os.environ.get("WAN22_DEBUG_LR_ASSIGNMENT", "0") == "1"
    in_already = []
    for name, param in transformer3d.named_parameters():
        if not param.requires_grad:
            continue
        if name in in_already:
            continue
        low_lr_flag = False
        for trainable_module_name in args.trainable_modules_low_learning_rate:
            if trainable_module_name in name:
                in_already.append(name)
                low_lr_flag = True
                trainable_params_optim[1]['params'].append(param)
                if debug_lr_assignment and accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate / 2}")
                break
        if low_lr_flag:
            continue
        for trainable_module_name in args.trainable_modules:
            if trainable_module_name in name:
                in_already.append(name)
                trainable_params_optim[0]['params'].append(param)
                if debug_lr_assignment and accelerator.is_main_process:
                    print(f"Set {name} to lr : {args.learning_rate}")
                break

    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            # weight_decay=args.adam_weight_decay,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params_optim,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Get the training dataset
    sample_n_frames_bucket_interval = vae.config.temporal_compression_ratio
    spatial_compression_ratio = vae.config.spatial_compression_ratio
    
    if args.fix_sample_size is not None and args.enable_bucket:
        args.video_sample_size = max(max(args.fix_sample_size), args.video_sample_size)
        args.image_sample_size = max(max(args.fix_sample_size), args.image_sample_size)
        args.training_with_video_token_length = False
        args.random_hw_adapt = False

    skip_unused_control_pixel_values = args.train_mode == "control_camera_ref"

    # Get the dataset
    train_dataset = ImageVideoControlDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        video_length_drop_start=0.0 if args.action_injection != "none" else 0.1,
        video_length_drop_end=1.0 if args.action_injection != "none" else 0.9,
        enable_bucket=args.enable_bucket, 
        enable_camera_info=args.train_mode == "control_camera_ref",
        enable_arm_info=args.enable_arm_info,
        enable_action_map_info=args.enable_action_map_info,
        skip_control_pixel_values=skip_unused_control_pixel_values,
        arm_action_stat_path=args.arm_action_stat_path,
        arm_action_key=args.arm_action_key,
        arm_action_dim=args.arm_action_dim,
        arm_action_num_frames=args.arm_action_num_frames,
        max_samples=args.max_train_samples,
        max_retries=args.dataset_max_retries,
    )

    def _get_control_type(example):
        return example.get("control_type", "control")

    def _sample_clip_index(num_frames):
        if args.control_ref_image == "first_frame":
            return 0

        def _create_special_list(length):
            if length == 1:
                return [1.0]
            if length >= 2:
                first_element = 0.40
                remaining_sum = 1.0 - first_element
                other_elements_value = remaining_sum / (length - 1)
                special_list = [first_element] + [other_elements_value] * (length - 1)
                return special_list

        number_list_prob = np.array(_create_special_list(num_frames))
        return int(np.random.choice(list(range(num_frames)), p=number_list_prob))

    def _prepare_arm_action(example):
        local_arm_action = torch.zeros((args.arm_action_num_frames, args.arm_action_dim), dtype=torch.float32)
        local_arm_mask = 0.0
        arm_action_values = example.get("arm_action_values", None)
        control_type = _get_control_type(example)

        if not args.enable_arm_info or control_type != "arm" or arm_action_values is None:
            return local_arm_action, local_arm_mask

        local_arm_action = torch.as_tensor(arm_action_values, dtype=torch.float32)
        if local_arm_action.ndim == 1:
            local_arm_action = local_arm_action.unsqueeze(0)
        elif local_arm_action.ndim > 2:
            local_arm_action = local_arm_action.reshape(local_arm_action.shape[0], -1)

        if local_arm_action.size(0) == 0:
            return torch.zeros((args.arm_action_num_frames, args.arm_action_dim), dtype=torch.float32), 0.0

        if local_arm_action.size(0) > args.arm_action_num_frames:
            frame_index = torch.linspace(0, local_arm_action.size(0) - 1, args.arm_action_num_frames).long()
            local_arm_action = local_arm_action[frame_index]
        elif local_arm_action.size(0) < args.arm_action_num_frames:
            pad_size = args.arm_action_num_frames - local_arm_action.size(0)
            local_arm_action = torch.cat([local_arm_action, local_arm_action[-1:].repeat(pad_size, 1)], dim=0)

        if local_arm_action.size(1) > args.arm_action_dim:
            local_arm_action = local_arm_action[:, :args.arm_action_dim]
        elif local_arm_action.size(1) < args.arm_action_dim:
            local_arm_action = torch.cat(
                [
                    local_arm_action,
                    local_arm_action.new_zeros(local_arm_action.size(0), args.arm_action_dim - local_arm_action.size(1)),
                ],
                dim=1,
            )

        return local_arm_action, 1.0

    def collate_fn_no_bucket(examples):
        new_examples = {}
        new_examples["pixel_values"] = torch.stack([example["pixel_values"] for example in examples])
        if not skip_unused_control_pixel_values:
            new_examples["control_pixel_values"] = torch.stack([example["control_pixel_values"] for example in examples])
        if args.enable_action_map_info:
            action_map_flags = [
                _get_control_type(example) in {"action_map", "poseanything"}
                and example.get("action_map_pixel_values", None) is not None
                for example in examples
            ]
            new_examples["action_map_mask"] = torch.tensor(action_map_flags, dtype=torch.float32)
            new_examples["action_map_pixel_values"] = torch.stack([
                example["action_map_pixel_values"] if example.get("action_map_pixel_values", None) is not None else torch.zeros_like(example["pixel_values"])
                for example in examples
            ])
        new_examples["text"] = [example["text"] for example in examples]
        new_examples["idx"] = torch.tensor([example["idx"] for example in examples])
        new_examples["data_type"] = [example["data_type"] for example in examples]
        new_examples["control_type"] = [_get_control_type(example) for example in examples]
    
        if args.train_mode == "control_camera_ref":
            use_camera_flags = [
                _get_control_type(example) == "camera" and example.get("control_camera_values", None) is not None
                for example in examples
            ]
            new_examples["control_camera_mask"] = torch.tensor(use_camera_flags, dtype=torch.float32)
            if any(use_camera_flags):
                control_camera_values = []
                for example, use_camera in zip(examples, use_camera_flags):
                    if use_camera:
                        control_camera_values.append(example["control_camera_values"])
                    else:
                        example_pixel_values = example["pixel_values"]
                        control_camera_values.append(
                            torch.zeros(
                                (example_pixel_values.size(0), 6, example_pixel_values.size(2), example_pixel_values.size(3)),
                                dtype=example_pixel_values.dtype,
                            )
                        )
                new_examples["control_camera_values"] = torch.stack(control_camera_values)
            else:
                new_examples["control_camera_values"] = None

        if args.enable_arm_info:
            arm_action_values = []
            arm_action_mask = []
            for example in examples:
                local_arm_action, local_arm_mask = _prepare_arm_action(example)
                arm_action_values.append(local_arm_action)
                arm_action_mask.append(local_arm_mask)
            new_examples["arm_action_values"] = torch.stack(arm_action_values)
            new_examples["arm_action_mask"] = torch.tensor(arm_action_mask, dtype=torch.float32)

        if args.train_mode != "control":
            new_examples["ref_pixel_values"] = []
            new_examples["clip_pixel_values"] = []
            new_examples["clip_idx"] = []
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []

            for pixel_values in new_examples["pixel_values"]:
                clip_index = _sample_clip_index(len(pixel_values))
                new_examples["clip_idx"].append(clip_index)

                ref_pixel_values = pixel_values[clip_index].unsqueeze(0)
                new_examples["ref_pixel_values"].append(ref_pixel_values)

                clip_pixel_values = pixel_values[clip_index].permute(1, 2, 0).contiguous()
                clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                new_examples["clip_pixel_values"].append(clip_pixel_values)

                if args.add_inpaint_info:
                    mask = get_random_mask(pixel_values.size())
                    mask_pixel_values = pixel_values * (1 - mask) 
                    # Wan 2.1 use 0 for masked pixels
                    # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                    new_examples["mask_pixel_values"].append(mask_pixel_values)
                    new_examples["mask"].append(mask)

        if args.enable_text_encoder_in_dataloader:
            prompt_ids = tokenizer(
                new_examples['text'], 
                max_length=args.tokenizer_max_length, 
                padding="max_length", 
                add_special_tokens=True, 
                truncation=True, 
                return_tensors="pt"
            )
            encoder_hidden_states = text_encoder(
                prompt_ids.input_ids
            )[0]
            new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
            new_examples['encoder_hidden_states'] = encoder_hidden_states

        return new_examples

    def worker_init_fn(_seed):
        _seed = _seed * 256
        def _worker_init_fn(worker_id):
            if os.environ.get("WAN22_DEBUG_WORKER_INIT", "0") == "1":
                print(f"worker_init_fn with {_seed + worker_id}")
            np.random.seed(_seed + worker_id)
            random.seed(_seed + worker_id)
        return _worker_init_fn
    
    if args.enable_bucket:
        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        sampler_num_samples = None
        use_complete_fixed_size_sampler = False
        if args.method1_sample_loss_dir:
            sampler_num_samples = padded_epoch_sample_count(
                len(train_dataset),
                accelerator.num_processes,
                args.gradient_accumulation_steps,
                args.train_batch_size,
            )
            accelerator.print(
                "CAP epoch sample coverage audit: "
                f"metadata_samples={len(train_dataset)} "
                f"scheduled_samples={sampler_num_samples} "
                f"padding_samples={sampler_num_samples - len(train_dataset)} "
                f"effective_batch={accelerator.num_processes * args.gradient_accumulation_steps * args.train_batch_size} "
                "drop_metadata_samples=0"
            )
            if args.train_batch_size > 1:
                content_types = {
                    item.get("type", "image") if isinstance(item, dict) else "unknown"
                    for item in train_dataset.dataset
                }
                if args.fix_sample_size is None or len(content_types) != 1:
                    raise RuntimeError(
                        "Complete per-metadata Method1 loss recording with "
                        "--train_batch_size>1 requires fixed resolution and one content type; "
                        f"fix_sample_size={args.fix_sample_size} content_types={sorted(content_types)}."
                    )
                use_complete_fixed_size_sampler = True

        base_sampler = RandomSampler(
            train_dataset,
            replacement=False,
            num_samples=sampler_num_samples,
            generator=batch_sampler_generator,
            pad_with_first=(
                sampler_num_samples is not None
                and sampler_num_samples > len(train_dataset)
            ),
        )
        if sampler_num_samples is not None and sampler_num_samples > len(train_dataset):
            accelerator.print(
                "CAP prefix padding audit: "
                f"mode=front_metadata_prefix count={sampler_num_samples - len(train_dataset)} "
                f"metadata_indices=0..{sampler_num_samples - len(train_dataset) - 1}"
            )
        if use_complete_fixed_size_sampler:
            batch_sampler = torch.utils.data.BatchSampler(
                base_sampler,
                args.train_batch_size,
                drop_last=False,
            )
            accelerator.print(
                "CAP complete-batch sampler audit: "
                f"mode=fixed_size_single_type batch_size={args.train_batch_size} "
                "full_batches=1 passed=1"
            )
        else:
            batch_sampler = AspectRatioBatchImageVideoSampler(
                sampler=base_sampler,
                dataset=train_dataset.dataset,
                batch_size=args.train_batch_size,
                train_folder=args.train_data_dir,
                drop_last=False,
                aspect_ratios=aspect_ratio_sample_size,
            )

        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]
                
                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }

                return length_to_frame_num

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
                        remaining_sum = 1.0 - first_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [first_element] + [other_elements_value] * (length - 1)
                        return special_list
                        
                if sample_size >= 1536:
                    number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
                elif sample_size >= 1024:
                    number_list = [1, 1.25, 1.5, 2] + image_ratio
                elif sample_size >= 768:
                    number_list = [1, 1.25, 1.5] + image_ratio
                elif sample_size >= 512:
                    number_list = [1] + image_ratio
                else:
                    number_list = [1]

                if all_choices:
                    return number_list

                number_list_prob = np.array(_create_special_list(len(number_list)))
                if rng is None:
                    return np.random.choice(number_list, p = number_list_prob)
                else:
                    return rng.choice(number_list, p = number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            new_examples["idx"]          = []
            new_examples["control_type"] = []
            # Used in Control Mode
            if not skip_unused_control_pixel_values:
                new_examples["control_pixel_values"] = []
            # Used in Control Ref Mode
            if args.train_mode != "control":
                new_examples["ref_pixel_values"] = []
                new_examples["clip_pixel_values"] = []
                new_examples["clip_idx"] = []
            # Used in Control Camera Ref Mode
            if args.train_mode == "control_camera_ref":
                new_examples["control_camera_values"] = []
                new_examples["control_camera_mask"] = []
            if args.enable_arm_info:
                new_examples["arm_action_values"] = []
                new_examples["arm_action_mask"] = []
            if args.enable_action_map_info:
                new_examples["action_map_pixel_values"] = []
                new_examples["action_map_mask"] = []
                
            # Used in Inpaint mode 
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
                new_examples["clip_pixel_values"] = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

                aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))

                        def get_random_downsample_probability(choice_list, token_sample_size):
                            length = len(choice_list)
                            if length == 1:
                                return [1.0]  # If there's only one element, it gets all the probability
                            
                            # Find the index of the closest value to token_sample_size
                            closest_index = min(range(length), key=lambda i: abs(choice_list[i] - token_sample_size))
                            
                            # Assign 50% to the closest index
                            first_element = 0.50
                            remaining_sum = 1.0 - first_element
                            
                            # Distribute the remaining 50% evenly among the other elements
                            other_elements_value = remaining_sum / (length - 1) if length > 1 else 0.0
                            
                            # Construct the probability distribution
                            probability_list = [other_elements_value] * length
                            probability_list[closest_index] = first_element
                            
                            return probability_list

                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        probabilities = get_random_downsample_probability(choice_list, args.token_sample_size)
                        local_video_sample_size = np.random.choice(choice_list, p=probabilities)

                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            if args.fix_sample_size is not None:
                fix_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in args.fix_sample_size]
            elif args.random_ratio_crop:
                if rng is None:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                else:
                    random_sample_size = aspect_ratio_random_crop_sample_size[
                        rng.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                    ]
                random_sample_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in random_sample_size]
            else:
                closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
                closest_size = [int(x / spatial_compression_ratio / 2) * spatial_compression_ratio * 2 for x in closest_size]

            min_example_length = min(
                [example["pixel_values"].shape[0] for example in examples]
            )
            batch_video_length = int(min(batch_video_length, min_example_length))

            # Magvae needs the number of frames to be 4n + 1.
            batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

            if batch_video_length <= 0:
                batch_video_length = 1

            use_camera_flags = None
            batch_has_camera = False
            if args.train_mode == "control_camera_ref":
                use_camera_flags = [
                    _get_control_type(example) == "camera" and example.get("control_camera_values", None) is not None
                    for example in examples
                ]
                batch_has_camera = any(use_camera_flags)
                
            for example_idx, example in enumerate(examples):
                # To 0~1
                pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                pixel_values = pixel_values / 255.

                if not skip_unused_control_pixel_values:
                    control_pixel_values = torch.from_numpy(example["control_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    control_pixel_values = control_pixel_values / 255.
                if args.enable_action_map_info:
                    if example.get("action_map_pixel_values", None) is not None:
                        action_map_pixel_values = torch.from_numpy(example["action_map_pixel_values"]).permute(0, 3, 1, 2).contiguous()
                        action_map_pixel_values = action_map_pixel_values / 255.
                    else:
                        action_map_pixel_values = torch.zeros_like(pixel_values)

                if args.fix_sample_size is not None:
                    # Get adapt hw for resize
                    fix_sample_size = list(map(lambda x: int(x), fix_sample_size))
                    pose_resize_size = fix_sample_size
                    transform = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])

                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(fix_sample_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(fix_sample_size),
                    ])
                elif args.random_ratio_crop:
                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    pose_resize_size = [nh, nw]
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
    
                    transform_no_normalize = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                    ])
                else:
                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    pose_resize_size = resize_size
                    
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
    
                    transform_no_normalize = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                    ])

                new_examples["pixel_values"].append(transform(pixel_values)[:batch_video_length])
                if not skip_unused_control_pixel_values:
                    new_examples["control_pixel_values"].append(transform(control_pixel_values))
                control_type = _get_control_type(example)
                new_examples["control_type"].append(control_type)
                if args.enable_action_map_info:
                    new_examples["action_map_pixel_values"].append(transform(action_map_pixel_values)[:batch_video_length])
                    new_examples["action_map_mask"].append(
                        float(
                            control_type in {"action_map", "poseanything"}
                            and example.get("action_map_pixel_values", None) is not None
                        )
                    )
            
                if args.train_mode == "control_camera_ref":
                    use_camera = use_camera_flags[example_idx]
                    new_examples["control_camera_mask"].append(float(use_camera))
                    if batch_has_camera:
                        if not use_camera:
                            example_pixel_values = new_examples["pixel_values"][-1]
                            control_camera_values_size = (
                                example_pixel_values.size()[0], 
                                6, 
                                example_pixel_values.size()[2], 
                                example_pixel_values.size()[3]
                            )
                            local_control_camera_values = torch.zeros(control_camera_values_size, dtype=example_pixel_values.dtype)
                            new_examples["control_camera_values"].append(local_control_camera_values)
                        else:
                            local_control_camera_values = process_pose_params(example["control_camera_values"], height=pose_resize_size[0], width=pose_resize_size[1]).permute(0, 3, 1, 2).contiguous()
                            new_examples["control_camera_values"].append(transform_no_normalize(local_control_camera_values))

                if args.enable_arm_info:
                    local_arm_action, local_arm_mask = _prepare_arm_action(example)
                    new_examples["arm_action_values"].append(local_arm_action)
                    new_examples["arm_action_mask"].append(local_arm_mask)
                
                new_examples["text"].append(example["text"])
                new_examples["idx"].append(int(example["idx"]))

                if args.train_mode != "control":
                    clip_index = _sample_clip_index(len(new_examples["pixel_values"][-1]))
                    new_examples["clip_idx"].append(clip_index)

                    ref_pixel_values = new_examples["pixel_values"][-1][clip_index].unsqueeze(0)
                    new_examples["ref_pixel_values"].append(ref_pixel_values)

                    clip_pixel_values = new_examples["pixel_values"][-1][clip_index].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    new_examples["clip_pixel_values"].append(clip_pixel_values)

                    if args.add_inpaint_info:
                        mask = get_random_mask(new_examples["pixel_values"][-1].size())
                        mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) 
                        # Wan 2.1 use 0 for masked pixels
                        # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                        new_examples["mask_pixel_values"].append(mask_pixel_values)
                        new_examples["mask"].append(mask)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example for example in new_examples["pixel_values"]])
            new_examples["idx"] = torch.tensor(new_examples["idx"], dtype=torch.long)
            if not skip_unused_control_pixel_values:
                new_examples["control_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_pixel_values"]])
            if args.train_mode != "control":
                new_examples["ref_pixel_values"] = torch.stack([example for example in new_examples["ref_pixel_values"]])
                new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])
                new_examples["clip_idx"] = torch.tensor(new_examples["clip_idx"])
            if args.train_mode == "control_camera_ref":
                if batch_has_camera:
                    new_examples["control_camera_values"] = torch.stack([example[:batch_video_length] for example in new_examples["control_camera_values"]])
                else:
                    new_examples["control_camera_values"] = None
                new_examples["control_camera_mask"] = torch.tensor(new_examples["control_camera_mask"], dtype=torch.float32)
            if args.enable_arm_info:
                new_examples["arm_action_values"] = torch.stack([example for example in new_examples["arm_action_values"]])
                new_examples["arm_action_mask"] = torch.tensor(new_examples["arm_action_mask"], dtype=torch.float32)
            if args.enable_action_map_info:
                new_examples["action_map_pixel_values"] = torch.stack([example for example in new_examples["action_map_pixel_values"]])
                new_examples["action_map_mask"] = torch.tensor(new_examples["action_map_mask"], dtype=torch.float32)
            if args.add_inpaint_info:
                new_examples["mask_pixel_values"] = torch.stack([example for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example for example in new_examples["mask"]])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )
    else:
        # DataLoaders creation:
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            collate_fn=collate_fn_no_bucket,
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    transformer3d, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer3d, optimizer, train_dataloader, lr_scheduler
    )

    if fsdp_stage != 0 or zero_stage != 0:
        from functools import partial

        from videox_fun.dist import set_multi_gpus_devices, shard_model
        shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=weight_dtype)
        text_encoder = shard_fn(text_encoder)

    if args.use_ema:
        ema_transformer3d.to(accelerator.device)

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)
    if not args.enable_text_encoder_in_dataloader:
        text_encoder.to(accelerator.device if not args.low_vram else "cpu", dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs. A deliberately
    # changed resume dataset uses the requested value as an additional-epoch
    # count; its absolute epoch range is assigned after the checkpoint loads.
    if args.resume_with_new_dataset:
        args.num_train_epochs = resume_new_dataset_epochs
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        keys_to_pop = [k for k, v in tracker_config.items() if isinstance(v, list)]
        for k in keys_to_pop:
            tracker_config.pop(k)
            print(f"Removed tracker_config['{k}']")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Function for unwrapping if model was compiled with `torch.compile`.
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        resume_checkpoint_path = None
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
            if os.path.isdir(args.resume_from_checkpoint):
                resume_checkpoint_path = args.resume_from_checkpoint
            else:
                resume_checkpoint_path = os.path.join(args.output_dir, path)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            valid_dirs = []
            for checkpoint_name in dirs:
                checkpoint_dir = os.path.join(args.output_dir, checkpoint_name)
                has_model_state = (
                    os.path.isfile(os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors"))
                    or os.path.isfile(os.path.join(checkpoint_dir, "pytorch_model.bin"))
                    or os.path.isdir(os.path.join(checkpoint_dir, "transformer"))
                    or os.path.isfile(os.path.join(checkpoint_dir, "pytorch_model_fsdp_0", ".metadata"))
                )
                has_training_state = (
                    os.path.isfile(os.path.join(checkpoint_dir, "scheduler.bin"))
                    and (
                        os.path.isfile(os.path.join(checkpoint_dir, "optimizer_0", ".metadata"))
                        or os.path.isdir(os.path.join(checkpoint_dir, "optimizer"))
                    )
                )
                if has_model_state and has_training_state:
                    valid_dirs.append(checkpoint_name)
                elif accelerator.is_main_process:
                    logger.warning(f"Skipping incomplete checkpoint {checkpoint_dir}")
            path = valid_dirs[-1] if len(valid_dirs) > 0 else None
            if path is not None:
                resume_checkpoint_path = os.path.join(args.output_dir, path)

        if path is None or resume_checkpoint_path is None or not os.path.isdir(resume_checkpoint_path):
            if args.resume_with_new_dataset:
                raise RuntimeError(
                    "--resume_with_new_dataset requires a valid checkpoint; "
                    f"could not resolve {args.resume_from_checkpoint!r}."
                )
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            global_step = int(path.split("-")[1])

            initial_global_step = global_step

            pkl_path = os.path.join(resume_checkpoint_path, "sampler_pos_start.pkl")
            saved_checkpoint_epoch = None
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    _, saved_checkpoint_epoch = pickle.load(file)
            elif args.resume_with_new_dataset:
                raise RuntimeError(
                    "--resume_with_new_dataset requires sampler_pos_start.pkl in the checkpoint: "
                    f"{resume_checkpoint_path}"
                )
            if args.resume_with_new_dataset:
                first_epoch = (saved_checkpoint_epoch + 1) if saved_checkpoint_epoch is not None else 0
                expected_target_step = (
                    global_step
                    + resume_new_dataset_epochs * num_update_steps_per_epoch
                )
                if args.max_train_steps != expected_target_step:
                    raise RuntimeError(
                        "New-dataset resume target mismatch: "
                        f"configured={args.max_train_steps} expected={expected_target_step} "
                        f"checkpoint_step={global_step} "
                        f"additional_epochs={resume_new_dataset_epochs} "
                        f"steps_per_epoch={num_update_steps_per_epoch}."
                    )
                args.num_train_epochs = first_epoch + resume_new_dataset_epochs
                accelerator.print(
                    "CAP new-dataset resume audit: "
                    f"checkpoint_step={global_step} first_epoch={first_epoch} "
                    f"additional_epochs={resume_new_dataset_epochs} "
                    f"steps_per_epoch={num_update_steps_per_epoch} "
                    f"target_step={args.max_train_steps} sampler_pos_start=0"
                )
            else:
                first_epoch = global_step // num_update_steps_per_epoch
            print(
                f"Load pkl from {pkl_path}. "
                f"Get saved_checkpoint_epoch = {saved_checkpoint_epoch}, "
                f"derived first_epoch = {first_epoch}."
            )

            accelerator.print(f"Resuming from checkpoint {resume_checkpoint_path}")
            if os.environ.get("VIDEOX_RESUME_RESET_OPTIMIZER", "0") == "1":
                accelerator.print(
                    "VIDEOX_RESUME_RESET_OPTIMIZER=1: loading model/sampler state only and "
                    "reinitializing optimizer/scheduler for the current trainable parameter set."
                )
                saved_optimizers = accelerator._optimizers
                saved_schedulers = accelerator._schedulers
                try:
                    accelerator._optimizers = []
                    accelerator._schedulers = []
                    accelerator.load_state(resume_checkpoint_path)
                finally:
                    accelerator._optimizers = saved_optimizers
                    accelerator._schedulers = saved_schedulers
            else:
                accelerator.load_state(resume_checkpoint_path)
    else:
        initial_global_step = 0

    if args.require_method1_sample_loss_recording and not args.method1_sample_loss_dir:
        raise RuntimeError(
            "--require_method1_sample_loss_recording requires --method1_sample_loss_dir."
        )
    if args.method1_sample_loss_dir and not args.enable_method1_focused_loss:
        raise RuntimeError(
            "Per-sample Method1 loss recording requires --enable_method1_focused_loss."
        )
    method1_sample_loss_recorder = None
    if args.method1_sample_loss_dir:
        if not isinstance(train_dataset.dataset, list):
            raise RuntimeError(
                "Per-sample Method1 loss recording requires list-backed metadata with stable indices."
            )
        method1_sample_loss_recorder = DualLossSampleRecorder(
            args.method1_sample_loss_dir,
            train_dataset.dataset,
            accelerator,
        )
        if args.resume_from_checkpoint and first_epoch > 0 and not args.resume_with_new_dataset:
            previous_sample_loss_path = method1_sample_loss_recorder.load_previous_epoch(
                first_epoch - 1
            )
            if accelerator.is_main_process:
                accelerator.print(
                    "Method1 previous-epoch resume audit: "
                    f"epoch={first_epoch} "
                    f"sample_losses_csv={previous_sample_loss_path or 'N/A'}"
                )
        if accelerator.is_main_process:
            accelerator.print(
                "Method1 sample-loss recording audit: "
                f"requested={int(args.require_method1_sample_loss_recording)} "
                f"enabled=1 metadata_items={len(train_dataset.dataset)} "
                f"id_field=metadata_index losses=weighted_and_uniform "
                f"output_dir={os.path.abspath(args.method1_sample_loss_dir)} passed=1"
            )

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    if args.multi_stream:
        # create extra cuda streams to speedup inpaint vae computation
        vae_stream_1 = torch.cuda.Stream()
        vae_stream_2 = torch.cuda.Stream()
    else:
        vae_stream_1 = None
        vae_stream_2 = None

    # Calculate the index we need
    boundary        = config['transformer_additional_kwargs'].get('boundary', 0.900)
    split_timesteps = args.train_sampling_steps * boundary
    differences     = torch.abs(noise_scheduler.timesteps - split_timesteps)
    closest_index   = torch.argmin(differences).item()
    print(f"The boundary is {boundary} and the boundary_type is {args.boundary_type}. The closest_index we calculate is {closest_index}")
    if args.boundary_type == "high":
        start_num_idx = 0
        train_sampling_steps = closest_index
    elif args.boundary_type == "low":
        start_num_idx = closest_index
        train_sampling_steps = args.train_sampling_steps - closest_index
    else:
        start_num_idx = 0
        train_sampling_steps = args.train_sampling_steps
    idx_sampling = DiscreteSampling(train_sampling_steps, start_num_idx=start_num_idx, uniform_sampling=args.uniform_sampling)

    if args.benchmark_timing_path and accelerator.is_main_process:
        metrics_path = os.path.abspath(args.benchmark_timing_path)
        metrics_parent = os.path.dirname(metrics_path)
        if metrics_parent:
            os.makedirs(metrics_parent, exist_ok=True)
        retained_latest_metrics = prepare_step_metrics_jsonl(
            metrics_path,
            initial_global_step if args.resume_from_checkpoint else None,
        )
        latest_metrics_path = os.path.join(
            os.path.dirname(metrics_path),
            "latest_metrics.json",
        )
        if args.resume_from_checkpoint:
            if retained_latest_metrics is None:
                if os.path.isfile(latest_metrics_path):
                    os.remove(latest_metrics_path)
            else:
                write_json_atomic(latest_metrics_path, retained_latest_metrics)
            accelerator.print(
                "CAP train metrics resume audit: "
                f"checkpoint_step={initial_global_step} "
                f"retained_through_step="
                f"{retained_latest_metrics['global_step'] if retained_latest_metrics else 0} "
                f"path={metrics_path}"
            )

    def synchronize_for_timing():
        if args.benchmark_timing_path and torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    debug_heartbeat_enabled = os.environ.get("WAN22_DEBUG_HEARTBEAT", "0") == "1"
    debug_heartbeat_steps = int(os.environ.get("WAN22_DEBUG_HEARTBEAT_STEPS", "20"))
    debug_heartbeat_every = max(1, int(os.environ.get("WAN22_DEBUG_HEARTBEAT_EVERY", "1")))
    debug_progress_every_steps = int(os.environ.get("WAN22_DEBUG_PROGRESS_EVERY_STEPS", "10"))
    debug_heartbeat_sync = os.environ.get("WAN22_DEBUG_HEARTBEAT_SYNC", "0") == "1"
    debug_heartbeat_stop_step = initial_global_step + debug_heartbeat_steps
    debug_heartbeat_file = None
    debug_heartbeat_dir = os.environ.get("WAN22_DEBUG_HEARTBEAT_DIR", "")
    if debug_heartbeat_enabled and debug_heartbeat_dir:
        os.makedirs(debug_heartbeat_dir, exist_ok=True)
        debug_heartbeat_file = os.path.join(
            debug_heartbeat_dir,
            f"heartbeat_rank{accelerator.process_index}_local{accelerator.local_process_index}.log",
        )

    def _debug_value_summary(value):
        if value is None:
            return "None"
        if torch.is_tensor(value):
            shape = tuple(value.shape)
            return f"Tensor(shape={shape}, dtype={value.dtype}, device={value.device})"
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, (list, tuple)):
            preview = ", ".join(_debug_value_summary(item) for item in list(value)[:3])
            suffix = ", ..." if len(value) > 3 else ""
            return f"{type(value).__name__}(len={len(value)}, items=[{preview}{suffix}])"
        if isinstance(value, dict):
            return f"dict(keys={list(value.keys())[:8]})"
        return repr(value)

    def debug_heartbeat(phase, *, epoch=None, dataloader_step=None, force=False, **items):
        if not debug_heartbeat_enabled:
            return
        if not force:
            if global_step >= debug_heartbeat_stop_step:
                return
            if dataloader_step is not None and dataloader_step % debug_heartbeat_every != 0:
                return
        if debug_heartbeat_sync and torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)
        memory_text = ""
        if torch.cuda.is_available():
            try:
                allocated_gb = torch.cuda.memory_allocated(accelerator.device) / (1024 ** 3)
                reserved_gb = torch.cuda.memory_reserved(accelerator.device) / (1024 ** 3)
                memory_text = f" cuda_allocated_gb={allocated_gb:.2f} cuda_reserved_gb={reserved_gb:.2f}"
            except Exception as exc:
                memory_text = f" cuda_memory_error={type(exc).__name__}:{exc}"
        item_text = " ".join(f"{key}={_debug_value_summary(value)}" for key, value in items.items())
        message = (
            f"[HEARTBEAT {time.strftime('%Y-%m-%d %H:%M:%S')}] "
            f"rank={accelerator.process_index}/{accelerator.num_processes} "
            f"local_rank={accelerator.local_process_index} "
            f"device={accelerator.device} "
            f"epoch={epoch} dataloader_step={dataloader_step} global_step={global_step} "
            f"sync_gradients={accelerator.sync_gradients} phase={phase}"
            f"{memory_text} {item_text}"
        )
        print(message, flush=True)
        if debug_heartbeat_file is not None:
            with open(debug_heartbeat_file, "a", buffering=1) as f:
                f.write(message + "\n")

    last_iter_end = time.perf_counter()
    benchmark_update_data_time = 0.0
    benchmark_update_compute_time = 0.0
    benchmark_update_vae_time = 0.0
    benchmark_update_micro_steps = 0
    nonfinite_update_skip_count = 0
    if args.method1_skip_nonfinite_updates and accelerator.is_main_process:
        accelerator.print(
            "CAP nonfinite update guard enabled: "
            f"max_skips={args.method1_max_nonfinite_update_skips}"
        )

    def finalize_method1_sample_loss_epoch(epoch, epoch_complete):
        summary = method1_sample_loss_recorder.finalize_epoch(
            epoch,
            epoch_complete,
            global_step,
        )
        local_missing = (
            int(summary["missing_metadata_candidates"])
            if accelerator.is_main_process
            else 0
        )
        gathered_missing = accelerator.gather(
            torch.tensor([local_missing], device=accelerator.device, dtype=torch.int64)
        )
        missing_metadata = int(gathered_missing.max().item())
        accelerator.wait_for_everyone()
        if (
            epoch_complete
            and args.require_method1_sample_loss_recording
            and missing_metadata != 0
        ):
            raise RuntimeError(
                "Complete Method1 sample-loss epoch missed metadata samples: "
                f"epoch={epoch + 1} missing={missing_metadata}."
            )
        if accelerator.is_main_process:
            accelerator.print(
                "Method1 sample-loss epoch finalized: "
                f"epoch={epoch + 1} complete={int(epoch_complete)} "
                f"observations={summary['observations']} "
                f"unique_samples={summary['unique_samples']} "
                f"missing_metadata={missing_metadata} "
                f"sample_losses_csv={summary['sample_losses_csv']} "
                f"comparison_csv={summary['comparison_csv'] or 'N/A'}"
            )
        return summary

    for epoch in range(first_epoch, args.num_train_epochs):
        checkpoint_epoch = epoch
        train_loss = 0.0
        train_uniform_loss = 0.0
        if method1_sample_loss_recorder is not None:
            resume_sample_loss_step = (
                initial_global_step
                if args.resume_from_checkpoint
                and not args.resume_with_new_dataset
                and epoch == first_epoch
                else None
            )
            retained_sample_loss_visits = method1_sample_loss_recorder.start_epoch(
                epoch,
                resume_optimizer_step=resume_sample_loss_step,
            )
            if accelerator.is_main_process and resume_sample_loss_step is not None:
                accelerator.print(
                    "Method1 sample-loss resume audit: "
                    f"epoch={epoch + 1} checkpoint_step={resume_sample_loss_step} "
                    f"retained_visits={retained_sample_loss_visits}"
                )
        epoch_last_dataloader_step = -1
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            epoch_last_dataloader_step = step
            skip_optimizer_update = False
            control_latents = None
            control_camera_latents = None
            control_camera_mask = None
            action_map_mask = None
            arm_action_values = None
            arm_action_mask = None
            mask_conditions = None
            full_ref = None
            debug_heartbeat(
                "batch_received",
                epoch=epoch,
                dataloader_step=step,
                batch_idx=batch.get("idx"),
                batch_data_type=batch.get("data_type"),
                batch_control_type=batch.get("control_type"),
                pixel_values=batch.get("pixel_values"),
            )
            if args.require_input_resolution is not None and epoch == first_epoch and step == 0:
                expected_height, expected_width = args.require_input_resolution
                actual_height, actual_width = map(int, batch["pixel_values"].shape[-2:])
                resolution_ok = (
                    actual_height == expected_height and actual_width == expected_width
                )
                if accelerator.is_main_process:
                    accelerator.print(
                        "Input resolution audit: "
                        f"requested={expected_width}x{expected_height} "
                        f"actual={actual_width}x{actual_height} "
                        f"passed={int(resolution_ok)}"
                    )
                if not resolution_ok:
                    raise RuntimeError(
                        "Required input resolution audit failed: "
                        f"expected HxW={expected_height}x{expected_width}, "
                        f"got {actual_height}x{actual_width}."
                    )
            synchronize_for_timing()
            iter_start = time.perf_counter()
            data_time = iter_start - last_iter_end
            compute_start = iter_start
            vae_encode_time = 0.0
            # Data batch sanity check
            if not args.skip_sanity_check and epoch == first_epoch and step == 0 and accelerator.is_local_main_process:
                pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                control_pixel_values = batch.get("control_pixel_values", None)
                pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                if control_pixel_values is not None:
                    control_pixel_values = control_pixel_values.cpu()
                    control_pixel_values = rearrange(control_pixel_values, "b f c h w -> b c f h w")
                os.makedirs(os.path.join(args.logging_dir, "sanity_check"), exist_ok=True)
                for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                    pixel_value = pixel_value[None, ...]
                    gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                    save_videos_grid(pixel_value, f"{args.logging_dir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                    if control_pixel_values is not None:
                        control_pixel_value = control_pixel_values[idx][None, ...]
                        save_videos_grid(control_pixel_value, f"{args.logging_dir}/sanity_check/{gif_name[:10]}_control.gif", rescale=True)
                
                if args.train_mode != "control":
                    ref_pixel_values = batch["ref_pixel_values"].cpu()
                    ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> b c f h w")
                    for idx, (ref_pixel_value, text) in enumerate(zip(ref_pixel_values, texts)):
                        ref_pixel_value = ref_pixel_value[None, ...]
                        gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{global_step}-{idx}'
                        save_videos_grid(ref_pixel_value, f"{args.logging_dir}/sanity_check/{gif_name[:10]}_ref.gif", rescale=True)

                if args.add_inpaint_info:
                    clip_pixel_values, mask_pixel_values, texts = batch['clip_pixel_values'].cpu(), batch['mask_pixel_values'].cpu(), batch['text']
                    mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                    for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        Image.fromarray(np.uint8(clip_pixel_value)).save(f"{args.logging_dir}/sanity_check/clip_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.png")
                        save_videos_grid(pixel_value, f"{args.logging_dir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{global_step}-{idx}'}.gif", rescale=True)

            with accelerator.accumulate(transformer3d):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                control_pixel_values = None
                if batch.get("control_pixel_values", None) is not None:
                    control_pixel_values = batch["control_pixel_values"].to(weight_dtype)
                control_camera_values = None
                control_camera_mask = None
                arm_action_values = None
                arm_action_mask = None
                action_map_pixel_values = None
                action_map_mask = None
                if args.train_mode == "control_camera_ref":
                    control_camera_mask = batch.get("control_camera_mask", None)
                    if control_camera_mask is not None:
                        control_camera_mask = control_camera_mask.to(device=pixel_values.device, dtype=torch.float32)
                    batch_control_camera_values = batch.get("control_camera_values", None)
                    if batch_control_camera_values is not None and (
                        control_camera_mask is None or bool(torch.any(control_camera_mask > 0).item())
                    ):
                        control_camera_values = batch_control_camera_values.to(weight_dtype)
                if args.enable_arm_info and "arm_action_values" in batch:
                    arm_action_values = batch["arm_action_values"].to(weight_dtype)
                    arm_action_mask = batch.get("arm_action_mask", None)
                    if arm_action_mask is not None:
                        arm_action_mask = arm_action_mask.to(device=arm_action_values.device, dtype=torch.float32)
                if args.enable_action_map_info and batch.get("action_map_pixel_values", None) is not None:
                    action_map_pixel_values = batch["action_map_pixel_values"].to(weight_dtype)
                    action_map_mask = batch.get("action_map_mask", None)
                    if action_map_mask is not None:
                        action_map_mask = action_map_mask.to(device=action_map_pixel_values.device, dtype=torch.float32)

                # Increase the batch size when the length of the latent sequence of the current sample is small
                if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                    if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (4, 1, 1, 1, 1))
                        if control_pixel_values is not None:
                            control_pixel_values = torch.tile(control_pixel_values, (4, 1, 1, 1, 1))
                        if action_map_pixel_values is not None:
                            action_map_pixel_values = torch.tile(action_map_pixel_values, (4, 1, 1, 1, 1))
                            if action_map_mask is not None:
                                action_map_mask = torch.tile(action_map_mask, (4,))
                        if args.train_mode == "control_camera_ref":
                            if control_camera_values is not None:
                                control_camera_values = torch.tile(control_camera_values, (4, 1, 1, 1, 1))
                            if control_camera_mask is not None:
                                control_camera_mask = torch.tile(control_camera_mask, (4,))
                        if arm_action_values is not None:
                            arm_action_values = torch.tile(arm_action_values, (4, 1, 1))
                            if arm_action_mask is not None:
                                arm_action_mask = torch.tile(arm_action_mask, (4,))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (4, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (4, 1))
                        else:
                            batch['text'] = batch['text'] * 4
                    elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                        pixel_values = torch.tile(pixel_values, (2, 1, 1, 1, 1))
                        if control_pixel_values is not None:
                            control_pixel_values = torch.tile(control_pixel_values, (2, 1, 1, 1, 1))
                        if action_map_pixel_values is not None:
                            action_map_pixel_values = torch.tile(action_map_pixel_values, (2, 1, 1, 1, 1))
                            if action_map_mask is not None:
                                action_map_mask = torch.tile(action_map_mask, (2,))
                        if args.train_mode == "control_camera_ref":
                            if control_camera_values is not None:
                                control_camera_values = torch.tile(control_camera_values, (2, 1, 1, 1, 1))
                            if control_camera_mask is not None:
                                control_camera_mask = torch.tile(control_camera_mask, (2,))
                        if arm_action_values is not None:
                            arm_action_values = torch.tile(arm_action_values, (2, 1, 1))
                            if arm_action_mask is not None:
                                arm_action_mask = torch.tile(arm_action_mask, (2,))
                        if args.enable_text_encoder_in_dataloader:
                            batch['encoder_hidden_states'] = torch.tile(batch['encoder_hidden_states'], (2, 1, 1))
                            batch['encoder_attention_mask'] = torch.tile(batch['encoder_attention_mask'], (2, 1))
                        else:
                            batch['text'] = batch['text'] * 2
                
                if args.train_mode != "control":
                    ref_pixel_values = batch["ref_pixel_values"].to(weight_dtype)
                    clip_idx = batch["clip_idx"]
                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and zero_stage != 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            ref_pixel_values = torch.tile(ref_pixel_values, (4, 1, 1, 1, 1))
                            clip_idx = torch.tile(clip_idx, (4,))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            ref_pixel_values = torch.tile(ref_pixel_values, (2, 1, 1, 1, 1))
                            clip_idx = torch.tile(clip_idx, (2,))

                if args.add_inpaint_info:
                    mask_pixel_values = batch["mask_pixel_values"].to(weight_dtype)
                    mask = batch["mask"].to(weight_dtype)
                    # Increase the batch size when the length of the latent sequence of the current sample is small
                    if args.auto_tile_batch_size and args.training_with_video_token_length and not zero_stage == 3:
                        if args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 16 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            mask_pixel_values = torch.tile(mask_pixel_values, (4, 1, 1, 1, 1))
                            mask = torch.tile(mask, (4, 1, 1, 1, 1))
                        elif args.video_sample_n_frames * args.token_sample_size * args.token_sample_size // 4 >= pixel_values.size()[1] * pixel_values.size()[3] * pixel_values.size()[4]:
                            mask_pixel_values = torch.tile(mask_pixel_values, (2, 1, 1, 1, 1))
                            mask = torch.tile(mask, (2, 1, 1, 1, 1))

                if args.random_frame_crop:
                    def _create_special_list(length):
                        if length == 1:
                            return [1.0]
                        if length >= 2:
                            last_element = 0.90
                            remaining_sum = 1.0 - last_element
                            other_elements_value = remaining_sum / (length - 1)
                            special_list = [other_elements_value] * (length - 1) + [last_element]
                            return special_list
                    select_frames = [_tmp for _tmp in list(range(sample_n_frames_bucket_interval + 1, args.video_sample_n_frames + sample_n_frames_bucket_interval, sample_n_frames_bucket_interval))]
                    select_frames_prob = np.array(_create_special_list(len(select_frames)))
                    
                    if len(select_frames) != 0:
                        if rng is None:
                            temp_n_frames = np.random.choice(select_frames, p = select_frames_prob)
                        else:
                            temp_n_frames = rng.choice(select_frames, p = select_frames_prob)
                    else:
                        temp_n_frames = 1

                    # Magvae needs the number of frames to be 4n + 1.
                    temp_n_frames = (temp_n_frames - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :temp_n_frames, :, :]
                    if control_pixel_values is not None:
                        control_pixel_values = control_pixel_values[:, :temp_n_frames, :, :]
                    if action_map_pixel_values is not None:
                        action_map_pixel_values = action_map_pixel_values[:, :temp_n_frames, :, :]
                    if args.train_mode == "control_camera_ref" and control_camera_values is not None:
                        control_camera_values = control_camera_values[:, :temp_n_frames, :, :, :]
                    
                # Keep all node same token length to accelerate the traning when resolution grows.
                if args.keep_all_node_same_token_length:
                    if args.token_sample_size > 256:
                        numbers_list = list(range(256, args.token_sample_size + 1, 128))

                        if numbers_list[-1] != args.token_sample_size:
                            numbers_list.append(args.token_sample_size)
                    else:
                        numbers_list = [256]
                    numbers_list = [_number * _number * args.video_sample_n_frames for _number in  numbers_list]
            
                    actual_token_length = index_rng.choice(numbers_list)
                    actual_video_length = (min(
                            actual_token_length / pixel_values.size()[-1] / pixel_values.size()[-2], args.video_sample_n_frames
                    ) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1
                    actual_video_length = int(max(actual_video_length, 1))

                    # Magvae needs the number of frames to be 4n + 1.
                    actual_video_length = (actual_video_length - 1) // sample_n_frames_bucket_interval + 1

                    pixel_values = pixel_values[:, :actual_video_length, :, :]
                    if control_pixel_values is not None:
                        control_pixel_values = control_pixel_values[:, :actual_video_length, :, :]
                    if action_map_pixel_values is not None:
                        action_map_pixel_values = action_map_pixel_values[:, :actual_video_length, :, :]
                    if args.train_mode == "control_camera_ref" and control_camera_values is not None:
                        control_camera_values = control_camera_values[:, :actual_video_length, :, :, :]

                if args.low_vram:
                    torch.cuda.empty_cache()
                    vae.to(accelerator.device)
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to("cpu")

                synchronize_for_timing()
                vae_encode_start = time.perf_counter()
                with torch.no_grad():
                    # This way is quicker when batch grows up
                    def _batch_encode_vae(pixel_values, sample_posterior=True):
                        pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                        bs = args.vae_mini_batch
                        new_pixel_values = []
                        for i in range(0, pixel_values.shape[0], bs):
                            pixel_values_bs = pixel_values[i : i + bs]
                            posterior = vae.encode(pixel_values_bs)[0]
                            pixel_values_bs = posterior.sample() if sample_posterior else posterior.mode()
                            new_pixel_values.append(pixel_values_bs)
                        return torch.cat(new_pixel_values, dim = 0)
                    if vae_stream_1 is not None:
                        vae_stream_1.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(vae_stream_1):
                            latents = _batch_encode_vae(pixel_values)
                    else:
                        latents = _batch_encode_vae(pixel_values)

                    action_map_latents = None
                    poseanything_null_latents = None
                    if action_map_pixel_values is not None and action_map_mask is not None and bool(torch.any(action_map_mask > 0).item()):
                        if args.action_injection == "poseanything":
                            # PoseAnything uses deterministic VAE mu for both real and black skeleton clips.
                            action_map_latents = _batch_encode_vae(
                                action_map_pixel_values, sample_posterior=False
                            )
                            black_skeleton_pixels = torch.full_like(action_map_pixel_values, -1.0)
                            poseanything_null_latents = _batch_encode_vae(
                                black_skeleton_pixels, sample_posterior=False
                            )
                        else:
                            action_map_latents = _batch_encode_vae(action_map_pixel_values)

                    if args.train_mode != "control_camera_ref":
                        control_latents = _batch_encode_vae(control_pixel_values)
                        # Make control latents to zero
                        for bs_index in range(control_latents.size()[0]):
                            if rng is None:
                                zero_init_control_latents_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                            else:
                                zero_init_control_latents_conv_in = rng.choice([0, 1], p = [0.90, 0.10])

                            if zero_init_control_latents_conv_in:
                                control_latents[bs_index] = control_latents[bs_index] * 0
                        control_camera_latents = None
                    else:
                        control_latents = None
                        if control_camera_values is None:
                            control_camera_latents = None
                        else:
                            control_camera_latents = pack_camera_condition(control_camera_values)
                            
                    if args.train_mode != "control":
                        ref_latents = _batch_encode_vae(ref_pixel_values)
                        if args.add_full_ref_image_in_self_attention:
                            full_ref = ref_latents[:, :, 0].clone()

                        ref_latents_conv_in = torch.zeros_like(latents).to(ref_latents.device, ref_latents.dtype)
                        ref_latents_conv_in[:, :, :1] = ref_latents
                        for bs_index in range(ref_latents.size()[0]):
                            if rng is None:
                                zero_init_ref_latents_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                            else:
                                zero_init_ref_latents_conv_in = rng.choice([0, 1], p = [0.90, 0.10])

                            if clip_idx[bs_index] != 0 or (zero_init_ref_latents_conv_in and latents.size()[1] != 1):
                                ref_latents_conv_in[bs_index, :, :1] = ref_latents_conv_in[bs_index, :, :1] * 0

                            if args.add_full_ref_image_in_self_attention:
                                if rng is None:
                                    zero_init_full_ref_conv_in = np.random.choice([0, 1], p = [0.90, 0.10])
                                else:
                                    zero_init_full_ref_conv_in = rng.choice([0, 1], p = [0.90, 0.10])
                                if clip_idx[bs_index] == 0 or zero_init_full_ref_conv_in:
                                    full_ref[bs_index] = full_ref[bs_index] * 0

                    if args.add_inpaint_info:
                        t2v_flag = [(_mask == 1).all() for _mask in mask]
                        new_t2v_flag = []
                        for _mask in t2v_flag:
                            if _mask and np.random.rand() < 0.90:
                                new_t2v_flag.append(0)
                            else:
                                new_t2v_flag.append(1)
                        t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(accelerator.device, dtype=weight_dtype)
                        
                        mask = rearrange(mask, "b f c h w -> b c f h w")
                        mask = torch.concat(
                            [
                                torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                                mask[:, :, 1:]
                            ], dim=2
                        )
                        mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
                        mask = mask.transpose(1, 2)
                        mask_conditions = F.interpolate(mask[:, :1], size=latents.size()[-3:], mode='trilinear', align_corners=True).to(accelerator.device, weight_dtype)
                        mask = resize_mask(1 - mask, latents)

                        # Encode inpaint latents.
                        mask_latents = _batch_encode_vae(mask_pixel_values)

                        inpaint_latents = torch.concat([mask, mask_latents], dim=1)
                        inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents
                    else:
                        inpaint_latents = None

                    if control_latents is None:
                        if inpaint_latents is None:
                            control_latents = ref_latents_conv_in
                        else:
                            control_latents = inpaint_latents
                    else:
                        if inpaint_latents is None:
                            control_latents = torch.cat([control_latents, ref_latents_conv_in], dim = 1)
                        else:
                            control_latents = torch.cat([control_latents, inpaint_latents], dim = 1)

                    reference_control_latents = control_latents
                    conditioned_control_latents = control_latents
                    null_control_latents = control_latents
                    if args.action_injection == "action_map":
                        if action_map_latents is None:
                            raise RuntimeError("Action-map mode produced no action-map VAE latents.")
                        conditioned_control_latents, null_control_latents = (
                            build_action_map_control_latents(
                                control_latents,
                                action_map_latents,
                                latent_channels=vae.latent_channels,
                                action_map_mask=action_map_mask,
                            )
                        )
                    elif args.action_injection == "poseanything":
                        if action_map_latents is None or poseanything_null_latents is None:
                            raise RuntimeError("PoseAnything mode produced no skeleton/null VAE latents.")
                        conditioned_control_latents, null_control_latents = (
                            build_poseanything_condition_latents(
                                latents,
                                action_map_latents,
                                poseanything_null_latents,
                                skeleton_mask=action_map_mask,
                            )
                        )
                    control_latents = conditioned_control_latents
                                
                # wait for latents = vae.encode(pixel_values) to complete
                if vae_stream_1 is not None:
                    torch.cuda.current_stream().wait_stream(vae_stream_1)
                synchronize_for_timing()
                vae_encode_time = time.perf_counter() - vae_encode_start

                if args.low_vram:
                    vae.to('cpu')
                    torch.cuda.empty_cache()
                    if not args.enable_text_encoder_in_dataloader:
                        text_encoder.to(accelerator.device)

                if args.enable_text_encoder_in_dataloader:
                    prompt_embeds = batch['encoder_hidden_states'].to(device=latents.device)
                else:
                    with torch.no_grad():
                        prompt_ids = tokenizer(
                            batch['text'], 
                            padding="max_length", 
                            max_length=args.tokenizer_max_length, 
                            truncation=True, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )
                        text_input_ids = prompt_ids.input_ids
                        prompt_attention_mask = prompt_ids.attention_mask

                        seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
                        prompt_embeds = text_encoder(text_input_ids.to(latents.device), attention_mask=prompt_attention_mask.to(latents.device))[0]
                        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]

                if args.low_vram and not args.enable_text_encoder_in_dataloader:
                    text_encoder.to('cpu')
                    torch.cuda.empty_cache()

                bsz, channel, num_frames, height, width = latents.size()
                debug_heartbeat(
                    "latents_and_text_ready",
                    epoch=epoch,
                    dataloader_step=step,
                    latents=latents,
                    control_latents=control_latents,
                    control_camera_latents=control_camera_latents,
                    mask_conditions=mask_conditions,
                    prompt_embeds=prompt_embeds,
                    arm_action_values=arm_action_values,
                    arm_action_mask=arm_action_mask,
                )
                noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)

                if not args.uniform_sampling:
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler.config.num_train_timesteps).long()
                else:
                    # Sample a random timestep for each image
                    # timesteps = generate_timestep_with_lognorm(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    # timesteps = torch.randint(0, args.train_sampling_steps, (bsz,), device=latents.device, generator=torch_rng)
                    indices = idx_sampling(bsz, generator=torch_rng, device=latents.device)
                    indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
                    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
                    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
                    timesteps = timesteps.to(accelerator.device)
                    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

                    sigma = sigmas[step_indices].flatten()
                    while len(sigma.shape) < n_dim:
                        sigma = sigma.unsqueeze(-1)
                    return sigma

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

                # Add noise
                target = noise - latents
                
                target_shape = (vae.latent_channels, num_frames, width, height)
                seq_len = math.ceil(
                    (target_shape[2] * target_shape[3]) /
                    (accelerator.unwrap_model(transformer3d).config.patch_size[1] * accelerator.unwrap_model(transformer3d).config.patch_size[2]) *
                    target_shape[1]
                )

                if spatial_compression_ratio >= 16:
                    mask_conditions_bs = mask_conditions.size()[0]
                    mask_conditions[:, :, 1:, :, :] = 1
                    if not mask_conditions[:, :, 0, :, :].any():
                        noisy_latents = (
                            (1 - mask_conditions)
                            * reference_control_latents[:, -vae.latent_channels:]
                            + mask_conditions * noisy_latents
                        )
                        
                        temp_ts = (mask_conditions[:, 0, :, ::2, ::2] * timesteps[:, None, None, None]).flatten(1)
                        timesteps = torch.cat([temp_ts, temp_ts.new_ones(mask_conditions_bs, seq_len - temp_ts.size(1)) * timesteps[:, None,]], dim = 1)
                    else:
                        timesteps = mask_conditions.new_ones(mask_conditions_bs, seq_len) * timesteps[:, None,]

                # Predict the noise residual
                transformer_for_action_mask = accelerator.unwrap_model(transformer3d)

                def set_current_action_map_mask(mask_value=None):
                    if mask_value is None:
                        transformer_for_action_mask._current_action_map_mask = torch.zeros(bsz, device=latents.device, dtype=torch.float32)
                    else:
                        transformer_for_action_mask._current_action_map_mask = mask_value.to(device=latents.device, dtype=torch.float32).view(-1)

                def build_token_timesteps(base_timesteps):
                    if spatial_compression_ratio >= 16:
                        mask_conditions_bs = mask_conditions.size()[0]
                        if not mask_conditions[:, :, 0, :, :].any():
                            temp_ts = (mask_conditions[:, 0, :, ::2, ::2] * base_timesteps[:, None, None, None]).flatten(1)
                            return torch.cat(
                                [
                                    temp_ts,
                                    temp_ts.new_ones(mask_conditions_bs, seq_len - temp_ts.size(1)) * base_timesteps[:, None,],
                                ],
                                dim=1,
                            )
                        return mask_conditions.new_ones(mask_conditions_bs, seq_len) * base_timesteps[:, None,]
                    return base_timesteps

                def apply_inpaint_to_noisy_latents(noisy_value):
                    if spatial_compression_ratio >= 16 and not mask_conditions[:, :, 0, :, :].any():
                        return (
                            (1 - mask_conditions)
                            * reference_control_latents[:, -vae.latent_channels:]
                            + mask_conditions * noisy_value
                        )
                    return noisy_value

                method1_has_arm_condition = args.enable_arm_info and arm_action_values is not None
                method1_has_camera_condition = (
                    args.train_mode == "control_camera_ref"
                    and control_camera_latents is not None
                )
                method1_has_latent_condition = (
                    args.action_injection in {"action_map", "poseanything"}
                    and action_map_mask is not None
                    and conditioned_control_latents is not None
                    and null_control_latents is not None
                )
                method1_enabled = args.enable_method1_focused_loss and (
                    method1_has_arm_condition
                    or method1_has_camera_condition
                    or method1_has_latent_condition
                )
                method1_requires_effect_map = args.method1_loss_variant == "CAER"
                method1_effect_map = None
                method1_main_arm_action_mask = arm_action_mask
                method1_main_control_camera_mask = control_camera_mask
                method1_main_action_map_mask = action_map_mask
                method1_main_control_latents = conditioned_control_latents
                method1_keep_mask = None
                method1_active_mask = None

                if method1_enabled:
                    if method1_has_arm_condition and arm_action_mask is None:
                        method1_base_arm_mask = (arm_action_values.float().abs().sum(dim=(1, 2)) > 1e-6).to(torch.float32)
                    elif method1_has_arm_condition:
                        method1_base_arm_mask = arm_action_mask.to(device=latents.device, dtype=torch.float32).view(-1)
                    else:
                        method1_base_arm_mask = torch.zeros((bsz,), device=latents.device, dtype=torch.float32)

                    if method1_has_camera_condition and control_camera_mask is None:
                        method1_base_control_camera_mask = torch.ones((bsz,), device=latents.device, dtype=torch.float32)
                    elif method1_has_camera_condition:
                        method1_base_control_camera_mask = control_camera_mask.to(device=latents.device, dtype=torch.float32).view(-1)
                    else:
                        method1_base_control_camera_mask = torch.zeros((bsz,), device=latents.device, dtype=torch.float32)

                    if method1_has_latent_condition:
                        method1_base_latent_mask = action_map_mask.to(
                            device=latents.device, dtype=torch.float32
                        ).view(-1)
                    else:
                        method1_base_latent_mask = torch.zeros(
                            (bsz,), device=latents.device, dtype=torch.float32
                        )

                    method1_base_condition_mask = torch.maximum(
                        torch.maximum(method1_base_arm_mask, method1_base_control_camera_mask),
                        method1_base_latent_mask,
                    )

                    dropout_prob = min(max(float(args.method1_action_dropout_prob), 0.0), 1.0)
                    if dropout_prob > 0:
                        dropout_mask = torch.rand((bsz,), device=latents.device, generator=torch_rng) < dropout_prob
                    else:
                        dropout_mask = torch.zeros((bsz,), device=latents.device, dtype=torch.bool)
                    method1_keep_mask = (~dropout_mask).to(torch.float32)
                    method1_active_mask = (method1_keep_mask * method1_base_condition_mask).view(bsz, 1, 1, 1, 1)
                    method1_main_arm_action_mask = method1_base_arm_mask * method1_keep_mask
                    method1_main_control_camera_mask = method1_base_control_camera_mask * method1_keep_mask
                    method1_main_action_map_mask = method1_base_latent_mask * method1_keep_mask
                    if method1_has_latent_condition:
                        latent_gate = method1_main_action_map_mask.to(
                            device=conditioned_control_latents.device,
                            dtype=conditioned_control_latents.dtype,
                        ).view(-1, 1, 1, 1, 1)
                        method1_main_control_latents = (
                            latent_gate * conditioned_control_latents
                            + (1.0 - latent_gate) * null_control_latents
                        )

                    if method1_requires_effect_map:
                        target_sigma = min(max(float(args.method1_tau_s), 0.0), 1.0)
                        method1_sigmas_all = noise_scheduler.sigmas.to(
                            device=latents.device, dtype=torch.float32
                        )
                        method1_index = int(
                            torch.argmin(
                                (method1_sigmas_all - target_sigma).abs()
                            ).item()
                        )
                        method1_index = max(
                            0,
                            min(
                                method1_index,
                                len(noise_scheduler.timesteps) - 1,
                            ),
                        )
                        method1_base_timestep = noise_scheduler.timesteps[
                            method1_index
                        ].to(device=latents.device)
                        method1_base_timesteps = method1_base_timestep.repeat(bsz)
                        method1_sigmas = get_sigmas(
                            method1_base_timesteps,
                            n_dim=latents.ndim,
                            dtype=latents.dtype,
                        )
                        if (
                            global_step == 0
                            and step == 0
                            and accelerator.is_main_process
                        ):
                            print(
                                "method1_diagnostic_noise "
                                f"target_sigma={target_sigma:.6f} "
                                f"actual_sigma={method1_sigmas_all[method1_index].item():.6f} "
                                f"scheduler_index={method1_index} "
                                f"timestep={method1_base_timestep.item():.6f}"
                            )
                        method1_noise = torch.randn(
                            latents.size(),
                            device=latents.device,
                            generator=method1_torch_rng,
                            dtype=weight_dtype,
                        )
                        method1_noisy_latents = (
                            (1.0 - method1_sigmas) * latents
                            + method1_sigmas * method1_noise
                        )
                        method1_noisy_latents = apply_inpaint_to_noisy_latents(
                            method1_noisy_latents
                        )
                        method1_timesteps = build_token_timesteps(
                            method1_base_timesteps
                        )
                        method1_null_arm_action_mask = torch.zeros_like(
                            method1_base_arm_mask
                        )
                        method1_null_control_camera_mask = torch.zeros_like(
                            method1_base_control_camera_mask
                        )
                        method1_null_action_map_mask = torch.zeros_like(
                            method1_base_latent_mask
                        )

                        # Keep diagnostics deterministic and invisible to the
                        # main training RNG stream on every FSDP rank.
                        diagnostic_device_index = (
                            accelerator.device.index
                            if accelerator.device.index is not None
                            else torch.cuda.current_device()
                        )
                        with torch.no_grad(), torch.random.fork_rng(
                            devices=[diagnostic_device_index]
                        ):
                            pair_cpu_rng_state = torch.get_rng_state()
                            pair_cuda_rng_state = torch.cuda.get_rng_state(
                                diagnostic_device_index
                            )
                            with torch.cuda.amp.autocast(
                                dtype=weight_dtype
                            ), torch.cuda.device(device=accelerator.device):
                                set_current_action_map_mask(
                                    method1_base_latent_mask
                                )
                                method1_pred_action = transformer3d(
                                    x=method1_noisy_latents,
                                    context=prompt_embeds,
                                    t=method1_timesteps,
                                    seq_len=seq_len,
                                    y=conditioned_control_latents,
                                    y_camera=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                                    y_camera_mask=method1_base_control_camera_mask if method1_has_camera_condition else None,
                                    arm_action=arm_action_values,
                                    arm_action_mask=method1_base_arm_mask if method1_has_arm_condition else None,
                                    full_ref=full_ref if args.add_full_ref_image_in_self_attention else None,
                                )
                                torch.set_rng_state(pair_cpu_rng_state)
                                torch.cuda.set_rng_state(
                                    pair_cuda_rng_state,
                                    device=diagnostic_device_index,
                                )
                                set_current_action_map_mask(
                                    method1_null_action_map_mask
                                )
                                method1_pred_null = transformer3d(
                                    x=method1_noisy_latents,
                                    context=prompt_embeds,
                                    t=method1_timesteps,
                                    seq_len=seq_len,
                                    y=null_control_latents,
                                    y_camera=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                                    y_camera_mask=method1_null_control_camera_mask if method1_has_camera_condition else None,
                                    arm_action=arm_action_values,
                                    arm_action_mask=method1_null_arm_action_mask if method1_has_arm_condition else None,
                                    full_ref=full_ref if args.add_full_ref_image_in_self_attention else None,
                                )
                            method1_effect_map = torch.linalg.vector_norm(
                                (
                                    method1_pred_action.float()
                                    - method1_pred_null.float()
                                ),
                                ord=2,
                                dim=1,
                                keepdim=True,
                            ).detach()
                            del method1_pred_action, method1_pred_null

                set_current_action_map_mask(method1_main_action_map_mask)
                debug_heartbeat(
                    "before_transformer_forward",
                    epoch=epoch,
                    dataloader_step=step,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    seq_len=seq_len,
                    target=target,
                    control_latents=method1_main_control_latents,
                    control_camera_latents=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                    control_camera_mask=control_camera_mask if args.train_mode == "control_camera_ref" else None,
                    action_map_mask=action_map_mask,
                    indices=indices,
                )
                with torch.cuda.amp.autocast(dtype=weight_dtype), torch.cuda.device(device=accelerator.device):
                    noise_pred = transformer3d(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=method1_main_control_latents,
                        y_camera=control_camera_latents if args.train_mode == "control_camera_ref" else None,
                        y_camera_mask=method1_main_control_camera_mask if args.train_mode == "control_camera_ref" and control_camera_latents is not None else None,
                        arm_action=arm_action_values,
                        arm_action_mask=method1_main_arm_action_mask,
                        full_ref=full_ref if args.add_full_ref_image_in_self_attention else None,
                    )
                debug_heartbeat(
                    "after_transformer_forward",
                    epoch=epoch,
                    dataloader_step=step,
                    noise_pred=noise_pred,
                )
                
                def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
                    noise_pred = noise_pred.float()
                    target = target.float()
                    diff = noise_pred - target
                    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
                    mask = (diff.abs() <= threshold).float()
                    masked_loss = mse_loss * mask
                    if weighting is not None:
                        masked_loss = masked_loss * weighting
                    final_loss = masked_loss.mean()
                    return final_loss

                def method1_focused_loss(noise_pred, target, effect_map, active_mask):
                    return method1_focused_flow_loss(
                        noise_pred,
                        target,
                        effect_map,
                        active_mask=active_mask,
                        eps=args.method1_eps,
                        mse_threshold=args.method1_mse_threshold,
                        exclude_first_frame=True,
                        loss_variant=args.method1_loss_variant,
                    )
                
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                method1_weighted_loss_for_log = None
                method1_uniform_loss_for_log = None
                method1_per_sample_loss_for_record = None
                method1_per_sample_uniform_loss_for_record = None
                if method1_enabled:
                    (
                        loss,
                        method1_rho,
                        method1_uniform_loss_for_log,
                        method1_per_sample_loss_for_record,
                        method1_per_sample_uniform_loss_for_record,
                    ) = method1_focused_loss(
                        noise_pred.float(),
                        target.float(),
                        method1_effect_map,
                        method1_active_mask,
                    )
                    method1_weighted_loss_for_log = loss.detach()
                else:
                    method1_rho = None
                    loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                    loss = loss.mean()
                debug_heartbeat(
                    "after_loss",
                    epoch=epoch,
                    dataloader_step=step,
                    loss=loss.detach(),
                    weighting=weighting,
                )
                if args.method1_log_stats and method1_enabled and accelerator.is_main_process:
                    rho_mean = method1_rho.float().mean().detach().item() if method1_rho is not None else 1.0
                    rho_max = method1_rho.float().max().detach().item() if method1_rho is not None else 1.0
                    active_ratio = method1_active_mask.float().mean().detach().item() if method1_active_mask is not None else 0.0
                    print(
                        f"method1_stats step={global_step} "
                        f"variant={args.method1_loss_variant} rho_mean={rho_mean:.6f} "
                        f"rho_max={rho_max:.6f} active_ratio={active_ratio:.6f}"
                    )

                if args.motion_sub_loss and noise_pred.size()[2] > 2:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio

                if method1_sample_loss_recorder is not None:
                    if (
                        method1_per_sample_loss_for_record is None
                        or method1_per_sample_uniform_loss_for_record is None
                    ):
                        raise RuntimeError(
                            "Method1 sample-loss recording is active but both per-sample weighted and "
                            "uniform losses were not produced."
                        )
                    if "idx" not in batch:
                        raise RuntimeError(
                            "Method1 sample-loss recording requires batch['idx'] metadata indices."
                        )
                    metadata_indices = batch["idx"].to(
                        device=method1_per_sample_loss_for_record.device,
                        dtype=torch.float64,
                    ).reshape(-1)
                    per_sample_losses = method1_per_sample_loss_for_record.to(dtype=torch.float64).reshape(-1)
                    per_sample_uniform_losses = method1_per_sample_uniform_loss_for_record.to(
                        dtype=torch.float64
                    ).reshape(-1)
                    if not (
                        metadata_indices.numel()
                        == per_sample_losses.numel()
                        == per_sample_uniform_losses.numel()
                    ):
                        raise RuntimeError(
                            "Method1 sample-loss ID/loss count mismatch: "
                            f"ids={metadata_indices.numel()} weighted={per_sample_losses.numel()} "
                            f"uniform={per_sample_uniform_losses.numel()}."
                        )
                    if method1_active_mask is None:
                        action_conditioned = torch.zeros_like(per_sample_losses)
                    else:
                        action_conditioned = (
                            method1_active_mask.reshape(method1_active_mask.size(0), -1)
                            .any(dim=1)
                            .to(dtype=torch.float64)
                        )
                    local_sample_records = torch.stack(
                        [
                            metadata_indices,
                            per_sample_losses,
                            per_sample_uniform_losses,
                            action_conditioned,
                        ],
                        dim=1,
                    )
                    gathered_sample_records = accelerator.gather(local_sample_records)
                    method1_sample_loss_recorder.record_gathered(
                        epoch,
                        step,
                        global_step,
                        gathered_sample_records,
                    )

                # Gather the losses across all processes for logging (if we use distributed training).
                debug_heartbeat("before_loss_gather", epoch=epoch, dataloader_step=step, loss=loss.detach())
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                debug_heartbeat("after_loss_gather", epoch=epoch, dataloader_step=step, avg_loss=avg_loss.detach())
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                if method1_uniform_loss_for_log is not None:
                    avg_uniform_loss = accelerator.gather(
                        method1_uniform_loss_for_log.repeat(args.train_batch_size)
                    ).mean()
                    train_uniform_loss += avg_uniform_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                debug_heartbeat("before_backward", epoch=epoch, dataloader_step=step, loss=loss.detach())
                accelerator.backward(loss)
                debug_heartbeat("after_backward", epoch=epoch, dataloader_step=step)
                if accelerator.sync_gradients:
                    if (
                        args.require_cap_condition_gradient
                        and not cap_condition_gradient_audit["checked"]
                    ):
                        values = cap_condition_gradient_audit["values"]
                        if args.use_fsdp:
                            for spec in cap_condition_gradient_audit["parameters"]:
                                parameter = spec["parameter"]
                                local_numel = torch.tensor(
                                    [parameter.numel()],
                                    device=accelerator.device,
                                    dtype=torch.int64,
                                )
                                shard_numels = accelerator.gather(local_numel)
                                expected_numel = math.prod(spec["full_shape"])
                                if int(shard_numels.sum().item()) != expected_numel:
                                    raise RuntimeError(
                                        "CAP condition-gradient FSDP shard coverage mismatch: "
                                        f"parameter_shape={spec['full_shape']} "
                                        f"shard_numels={shard_numels.tolist()} "
                                        f"expected_numel={expected_numel}."
                                    )
                                shard_offset = int(
                                    shard_numels[: accelerator.process_index].sum().item()
                                )
                                local_value = local_shard_max_abs(
                                    parameter.grad,
                                    full_shape=spec["full_shape"],
                                    shard_offset=shard_offset,
                                    channel_slice=spec["channel_slice"],
                                )
                                if local_value is None and parameter.numel() > 0:
                                    local_value = torch.full(
                                        (),
                                        float("nan"),
                                        device=accelerator.device,
                                        dtype=torch.float32,
                                    )
                                elif local_value is None:
                                    local_value = torch.zeros(
                                        (), device=accelerator.device, dtype=torch.float32
                                    )
                                values.append(local_value)
                        if values:
                            local_max_abs = torch.stack(values).amax()
                        else:
                            local_max_abs = torch.zeros(
                                (), device=accelerator.device, dtype=torch.float32
                            )
                        gathered_max_abs = accelerator.gather(local_max_abs.reshape(1))
                        finite = bool(torch.isfinite(gathered_max_abs).all().item())
                        max_abs = float(gathered_max_abs.max().item()) if finite else float("nan")
                        ranks_with_nonzero = (
                            int((gathered_max_abs > 0).to(dtype=torch.int64).sum().item())
                            if finite
                            else 0
                        )
                        passed = finite and max_abs > 0.0 and ranks_with_nonzero > 0
                        if accelerator.is_main_process:
                            accelerator.print(
                                "CAP condition gradient audit: "
                                f"mode={args.action_injection} max_abs={max_abs:.9g} "
                                f"ranks_with_nonzero={ranks_with_nonzero}/{accelerator.num_processes} "
                                f"finite={int(finite)} passed={int(passed)}"
                            )
                        if not passed:
                            raise RuntimeError(
                                "Required CAP condition-gradient audit failed for "
                                f"action_injection={args.action_injection}."
                            )
                        cap_condition_gradient_audit["checked"] = True
                        for handle in cap_condition_gradient_audit["handles"]:
                            handle.remove()
                        cap_condition_gradient_audit["handles"].clear()
                        cap_condition_gradient_audit["values"].clear()
                        cap_condition_gradient_audit["parameters"].clear()
                    if not args.use_deepspeed and not args.use_fsdp:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    debug_heartbeat("before_clip_grad_norm", epoch=epoch, dataloader_step=step)
                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    debug_heartbeat("after_clip_grad_norm", epoch=epoch, dataloader_step=step, norm_sum=norm_sum)
                    if args.method1_skip_nonfinite_updates:
                        if not torch.is_tensor(norm_sum):
                            norm_sum = torch.tensor(
                                norm_sum, device=accelerator.device, dtype=torch.float32
                            )
                        local_norm_finite = torch.isfinite(norm_sum).all()
                        rank_norm_finite = accelerator.gather(
                            local_norm_finite.to(
                                device=accelerator.device, dtype=torch.int64
                            ).reshape(1)
                        )
                        if not bool(rank_norm_finite.all().item()):
                            skip_optimizer_update = True
                            nonfinite_update_skip_count += 1
                            rank_norms = accelerator.gather(
                                norm_sum.detach().float().reshape(1)
                            )
                            if accelerator.is_main_process:
                                accelerator.print(
                                    "CAP nonfinite update skipped: "
                                    f"global_step={global_step} epoch={epoch + 1} "
                                    f"dataloader_step={step} rank_norm_finite="
                                    f"{rank_norm_finite.tolist()} rank_norms={rank_norms.tolist()} "
                                    f"skip_count={nonfinite_update_skip_count}"
                                )
                            if (
                                nonfinite_update_skip_count
                                > args.method1_max_nonfinite_update_skips
                            ):
                                raise RuntimeError(
                                    "CAP nonfinite update guard exceeded its skip limit: "
                                    f"skips={nonfinite_update_skip_count} "
                                    f"limit={args.method1_max_nonfinite_update_skips}."
                                )
                    if not args.use_deepspeed and not args.use_fsdp and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                if skip_optimizer_update:
                    # clip_grad_norm_ may have converted an inf norm into NaN
                    # gradients; discard the entire accumulated effective batch.
                    optimizer.zero_grad()
                    debug_heartbeat("after_nonfinite_update_skip", epoch=epoch, dataloader_step=step)
                else:
                    debug_heartbeat("before_optimizer_step", epoch=epoch, dataloader_step=step)
                    optimizer.step()
                    debug_heartbeat("after_optimizer_step", epoch=epoch, dataloader_step=step)
                    lr_scheduler.step()
                    optimizer.zero_grad()
                debug_heartbeat("after_scheduler_zero_grad", epoch=epoch, dataloader_step=step)

            synchronize_for_timing()
            iter_end = time.perf_counter()
            if args.benchmark_timing_path:
                benchmark_update_data_time += data_time
                benchmark_update_compute_time += iter_end - compute_start
                benchmark_update_vae_time += vae_encode_time
                benchmark_update_micro_steps += 1

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                if skip_optimizer_update:
                    # Keep global step and LR schedule tied to actual optimizer
                    # updates, not discarded invalid batches.
                    benchmark_update_data_time = 0.0
                    benchmark_update_compute_time = 0.0
                    benchmark_update_vae_time = 0.0
                    benchmark_update_micro_steps = 0
                    train_loss = 0.0
                    train_uniform_loss = 0.0
                    continue

                if args.use_ema:
                    ema_transformer3d.step(transformer3d.parameters())
                progress_bar.update(1)
                global_step += 1
                logged_train_loss = train_loss
                logged_uniform_loss = train_uniform_loss
                current_lr = lr_scheduler.get_last_lr()[0]
                accelerator.log(
                    {
                        "train_loss": logged_train_loss,
                        "method1_weighted_loss": logged_train_loss,
                        "method1_uniform_loss": logged_uniform_loss,
                        "learning_rate": current_lr,
                    },
                    step=global_step,
                )
                debug_heartbeat(
                    "after_progress_update",
                    epoch=epoch,
                    dataloader_step=step,
                    force=(
                        global_step <= debug_heartbeat_stop_step
                        or (
                            debug_progress_every_steps > 0
                            and global_step % debug_progress_every_steps == 0
                        )
                    ),
                    logged_train_loss=logged_train_loss,
                    lr=current_lr,
                )
                if args.benchmark_timing_path and accelerator.is_main_process:
                    metrics_record = {
                        "global_step": int(global_step),
                        "epoch": int(epoch) + 1,
                        "loss_variant": args.method1_loss_variant,
                        "micro_steps": int(benchmark_update_micro_steps),
                        "data_time_s": float(benchmark_update_data_time),
                        "compute_time_s": float(benchmark_update_compute_time),
                        "vae_encode_time_s": float(benchmark_update_vae_time),
                        "iter_time_s": float(
                            benchmark_update_data_time + benchmark_update_compute_time
                        ),
                        "method1_weighted_loss": float(logged_train_loss),
                        "method1_uniform_loss": float(logged_uniform_loss),
                        "learning_rate": float(current_lr),
                        "cuda_max_memory_allocated_gib": float(
                            torch.cuda.max_memory_allocated(accelerator.device) / (1024 ** 3)
                        ),
                        "cuda_max_memory_reserved_gib": float(
                            torch.cuda.max_memory_reserved(accelerator.device) / (1024 ** 3)
                        ),
                        "unix_time": float(time.time()),
                    }
                    append_jsonl(args.benchmark_timing_path, metrics_record)
                    latest_metrics_path = os.path.join(
                        os.path.dirname(os.path.abspath(args.benchmark_timing_path)),
                        "latest_metrics.json",
                    )
                    write_json_atomic(latest_metrics_path, metrics_record)
                    accelerator.print(
                        "CAP train metrics: "
                        f"step={global_step} epoch={epoch + 1} "
                        f"variant={args.method1_loss_variant} "
                        f"weighted_loss={logged_train_loss:.9g} "
                        f"uniform_loss={logged_uniform_loss:.9g} "
                        f"iter_time_s={metrics_record['iter_time_s']:.3f} "
                        f"data_time_s={metrics_record['data_time_s']:.3f} "
                        f"vae_time_s={metrics_record['vae_encode_time_s']:.3f}"
                    )
                benchmark_update_data_time = 0.0
                benchmark_update_compute_time = 0.0
                benchmark_update_vae_time = 0.0
                benchmark_update_micro_steps = 0
                train_loss = 0.0
                train_uniform_loss = 0.0

                if (
                    args.method1_heatmap_on_checkpoint
                    and args.method1_heatmap_every_steps > 0
                    and global_step % args.method1_heatmap_every_steps == 0
                    and accelerator.is_main_process
                ):
                    save_method1_checkpoint_heatmaps(
                        args,
                        global_step,
                        method1_effect_map,
                        noise_pred,
                        target,
                        method1_rho,
                        pixel_values=pixel_values,
                    )

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or args.use_fsdp or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            if accelerator.is_main_process:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                                checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                                # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                    removing_checkpoints = checkpoints[0:num_to_remove]

                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                        shutil.rmtree(removing_checkpoint, ignore_errors=True)
                            if args.use_deepspeed or args.use_fsdp:
                                accelerator.wait_for_everyone()

                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        debug_heartbeat("before_save_state", epoch=epoch, dataloader_step=step, force=True, save_path=save_path)
                        accelerator.save_state(save_path)
                        debug_heartbeat("after_save_state", epoch=epoch, dataloader_step=step, force=True, save_path=save_path)
                        if accelerator.is_main_process:
                            save_method1_checkpoint_heatmaps(
                                args,
                                global_step,
                                method1_effect_map,
                                noise_pred,
                                target,
                                method1_rho,
                                pixel_values=pixel_values,
                            )
                        logger.info(f"Saved state to {save_path}")

                if args.validation_prompts is not None and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_transformer3d.store(transformer3d.parameters())
                        ema_transformer3d.copy_to(transformer3d.parameters())
                    log_validation(
                        vae,
                        text_encoder,
                        tokenizer,
                        transformer3d,
                        args,
                        config,
                        accelerator,
                        weight_dtype,
                        global_step,
                    )
                    if args.use_ema:
                        # Switch back to the original transformer3d parameters.
                        ema_transformer3d.restore(transformer3d.parameters())

            logs = {"step_loss": loss.detach().item()}
            if method1_weighted_loss_for_log is not None:
                logs["method1_loss"] = method1_weighted_loss_for_log.item()
            if method1_uniform_loss_for_log is not None:
                logs["uniform_loss"] = method1_uniform_loss_for_log.item()
            logs["lr"] = lr_scheduler.get_last_lr()[0]
            progress_bar.set_postfix(**logs)
            last_iter_end = iter_end

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            if method1_sample_loss_recorder is not None:
                epoch_complete = (
                    epoch_last_dataloader_step + 1 >= len(train_dataloader)
                    or (
                        args.resume_from_checkpoint
                        and epoch == first_epoch
                        and batch_sampler.sampler._pos_start == 0
                    )
                )
                finalize_method1_sample_loss_epoch(epoch, epoch_complete)
            break

        if method1_sample_loss_recorder is not None:
            epoch_complete = (
                epoch_last_dataloader_step + 1 >= len(train_dataloader)
                or (
                    args.resume_from_checkpoint
                    and epoch == first_epoch
                    and batch_sampler.sampler._pos_start == 0
                )
            )
            finalize_method1_sample_loss_epoch(epoch, epoch_complete)

        if args.validation_prompts is not None and epoch % args.validation_epochs == 0:
            if args.use_ema:
                # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                ema_transformer3d.store(transformer3d.parameters())
                ema_transformer3d.copy_to(transformer3d.parameters())
            log_validation(
                vae,
                text_encoder,
                tokenizer,
                transformer3d,
                args,
                config,
                accelerator,
                weight_dtype,
                global_step,
            )
            if args.use_ema:
                # Switch back to the original transformer3d parameters.
                ema_transformer3d.restore(transformer3d.parameters())

    if args.method1_force_exit_after_training:
        os._exit(0)

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer3d = unwrap_model(transformer3d)
        if args.use_ema:
            ema_transformer3d.copy_to(transformer3d.parameters())

    if not args.skip_final_checkpoint and (args.use_deepspeed or args.use_fsdp or accelerator.is_main_process):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
        accelerator.save_state(save_path)
        logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
