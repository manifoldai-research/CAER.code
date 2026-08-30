"""Object-temporal attention and denoising schedules for Wan video models.

The utilities in this module are intentionally independent from the main
VideoX-Fun pipelines. They turn a classifier/annotation output into three
runtime controls:

1. cross-attention logit bias for object prompt tokens;
2. per-token timestep tensors, using Wan's existing ``t: [B, seq_len]`` path;
3. latent update gates that can reduce denoising strength outside active
   regions without editing scheduler internals.

Nothing is installed automatically. Callers opt in by loading a schedule and
using the wrapper/callback classes around a single inference run.
"""

from __future__ import annotations

import csv
import json
import math
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


Number = float | int


def _as_pair(value: Sequence[Number] | Mapping[str, Number], *, name: str) -> Tuple[float, float]:
    if isinstance(value, Mapping):
        if "width" in value and "height" in value:
            return float(value["width"]), float(value["height"])
        if "w" in value and "h" in value:
            return float(value["w"]), float(value["h"])
        raise ValueError(f"{name} mapping must contain width/height")
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return float(value[0]), float(value[1])


def _as_box(value: Sequence[Number], *, name: str) -> Tuple[float, float, float, float]:
    if len(value) != 4:
        raise ValueError(f"{name} must contain exactly four values")
    x1, y1, x2, y2 = [float(v) for v in value]
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{name} must be [x1, y1, x2, y2] with x2>=x1 and y2>=y1")
    return x1, y1, x2, y2


def _int_step_from_fraction(value: Optional[float], total_steps: int, default: int) -> int:
    if value is None:
        return int(default)
    value = max(0.0, min(1.0, float(value)))
    if total_steps <= 1:
        return 0
    return int(round(value * (total_steps - 1)))


def _timestep_matrix(value: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Normalize scalar/vector/per-token timesteps to [B, seq_len]."""

    value = value.detach() if not value.requires_grad else value
    batch_size = int(batch_size)
    seq_len = int(seq_len)
    if value.ndim == 0 or value.numel() == 1:
        return value.reshape(1, 1).expand(batch_size, seq_len).clone()
    if value.ndim >= 2:
        matrix = value.reshape(value.shape[0], -1)
        if matrix.shape[0] != batch_size:
            matrix = matrix[:1].expand(batch_size, matrix.shape[1])
        if matrix.shape[1] < seq_len:
            pad = matrix[:, -1:].expand(batch_size, seq_len - matrix.shape[1])
            matrix = torch.cat([matrix, pad], dim=1)
        return matrix[:, :seq_len].clone()

    vector = value.flatten()
    if vector.numel() == batch_size:
        return vector[:, None].expand(batch_size, seq_len).clone()
    if vector.numel() >= seq_len:
        return vector[:seq_len][None, :].expand(batch_size, seq_len).clone()
    pad = vector[-1:].expand(seq_len - vector.numel())
    vector = torch.cat([vector, pad], dim=0)
    return vector[None, :seq_len].expand(batch_size, seq_len).clone()


@dataclass(frozen=True)
class TokenGroupSpec:
    """Prompt phrases that describe one semantic object/action group."""

    name: str
    phrases: Tuple[str, ...] = ()
    label: str = ""
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenGroupSpec":
        return cls(
            name=str(payload["name"]),
            phrases=tuple(str(x) for x in payload.get("phrases", ())),
            label=str(payload.get("label") or payload["name"]),
            color=tuple(float(x) for x in payload.get("color", (1.0, 1.0, 1.0)))[:3],
        )


@dataclass(frozen=True)
class RegionSpec:
    """A spatial region with its own compute/denoising budget."""

    name: str
    role: str = "object"
    box_xyxy: Optional[Tuple[float, float, float, float]] = None
    box_space: str = "source"
    source_size: Optional[Tuple[float, float]] = None
    outside_all_boxes: bool = False
    token_group: Optional[str] = None
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    start_step_fraction: Optional[float] = None
    end_step_fraction: Optional[float] = None
    active_update_weight: float = 1.0
    inactive_update_weight: float = 0.0
    timestep_policy: str = "global"
    attention_scale: float = 1.0

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        inherited_source_size: Optional[Tuple[float, float]] = None,
    ) -> "RegionSpec":
        box = payload.get("box_xyxy")
        source_size = payload.get("source_size")
        if source_size is not None:
            source_size = _as_pair(source_size, name=f"{payload.get('name', 'region')}.source_size")
        else:
            source_size = inherited_source_size
        return cls(
            name=str(payload["name"]),
            role=str(payload.get("role", "object")),
            box_xyxy=_as_box(box, name=f"{payload['name']}.box_xyxy") if box is not None else None,
            box_space=str(payload.get("box_space", "source")),
            source_size=source_size,
            outside_all_boxes=bool(payload.get("outside_all_boxes", False)),
            token_group=payload.get("token_group"),
            start_step=int(payload["start_step"]) if payload.get("start_step") is not None else None,
            end_step=int(payload["end_step"]) if payload.get("end_step") is not None else None,
            start_step_fraction=(
                float(payload["start_step_fraction"])
                if payload.get("start_step_fraction") is not None
                else None
            ),
            end_step_fraction=(
                float(payload["end_step_fraction"])
                if payload.get("end_step_fraction") is not None
                else None
            ),
            active_update_weight=float(payload.get("active_update_weight", 1.0)),
            inactive_update_weight=float(payload.get("inactive_update_weight", 0.0)),
            timestep_policy=str(payload.get("timestep_policy", "global")),
            attention_scale=float(payload.get("attention_scale", 1.0)),
        )

    def step_window(self, total_steps: int) -> Tuple[int, int]:
        start = self.start_step
        end = self.end_step
        if start is None:
            start = _int_step_from_fraction(self.start_step_fraction, total_steps, 0)
        if end is None:
            end = _int_step_from_fraction(self.end_step_fraction, total_steps, max(total_steps - 1, 0))
        start = max(0, min(int(start), max(total_steps - 1, 0)))
        end = max(start, min(int(end), max(total_steps - 1, 0)))
        return start, end

    def is_active_step(self, step_index: int, total_steps: int) -> bool:
        start, end = self.step_window(total_steps)
        return start <= int(step_index) <= end

    def update_weight(self, step_index: int, total_steps: int) -> float:
        if self.is_active_step(step_index, total_steps):
            return float(self.active_update_weight)
        return float(self.inactive_update_weight)

    def effective_step(self, step_index: int, total_steps: int) -> int:
        """Map a global denoising step to this region's timestep embedding step."""

        step = int(step_index)
        start, end = self.step_window(total_steps)
        policy = self.timestep_policy
        if policy == "global":
            return step
        if policy == "clamp_to_window":
            return max(start, min(step, end))
        if policy == "hold_start_until_active":
            return step if step >= start else 0
        if policy == "late_only":
            return step if step >= start else 0
        if policy == "freeze_after_window":
            return min(step, end)
        raise ValueError(f"Unknown timestep_policy for region {self.name}: {policy}")

    def sample_xyxy(self, sample_width: int, sample_height: int) -> Optional[Tuple[float, float, float, float]]:
        if self.box_xyxy is None:
            return None
        x1, y1, x2, y2 = self.box_xyxy
        if self.box_space == "normalized":
            return (
                x1 * float(sample_width),
                y1 * float(sample_height),
                x2 * float(sample_width),
                y2 * float(sample_height),
            )
        if self.box_space == "sample":
            return x1, y1, x2, y2
        if self.box_space == "source":
            if self.source_size is None:
                raise ValueError(f"Region {self.name} uses source box coordinates without source_size")
            source_width, source_height = self.source_size
            sx = float(sample_width) / max(float(source_width), 1.0)
            sy = float(sample_height) / max(float(source_height), 1.0)
            return x1 * sx, y1 * sy, x2 * sx, y2 * sy
        raise ValueError(f"Unknown box_space for region {self.name}: {self.box_space}")


@dataclass(frozen=True)
class PhaseSpec:
    """A causal temporal segment, such as red -> green -> blue."""

    name: str
    active_regions: Tuple[str, ...] = ()
    active_groups: Tuple[str, ...] = ()
    frame_range: Optional[Tuple[float, float]] = None
    positive_logit_bias: float = math.log(4.0)
    inactive_logit_bias: float = -math.log(2.0)
    outside_box_penalty: float = math.log(1.5)
    competitor_box_penalty: float = math.log(2.0)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhaseSpec":
        frame_range = payload.get("frame_range")
        return cls(
            name=str(payload["name"]),
            active_regions=tuple(str(x) for x in payload.get("active_regions", ())),
            active_groups=tuple(str(x) for x in payload.get("active_groups", ())),
            frame_range=(
                _as_pair(frame_range, name=f"{payload['name']}.frame_range")
                if frame_range is not None
                else None
            ),
            positive_logit_bias=float(payload.get("positive_logit_bias", math.log(4.0))),
            inactive_logit_bias=float(payload.get("inactive_logit_bias", -math.log(2.0))),
            outside_box_penalty=float(payload.get("outside_box_penalty", math.log(1.5))),
            competitor_box_penalty=float(payload.get("competitor_box_penalty", math.log(2.0))),
        )


@dataclass
class DynamicAttentionSchedule:
    """Classifier output used to allocate attention, timestep, and updates."""

    name: str
    sample_width: int
    sample_height: int
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    token_groups: Dict[str, TokenGroupSpec] = field(default_factory=dict)
    regions: Dict[str, RegionSpec] = field(default_factory=dict)
    phases: List[PhaseSpec] = field(default_factory=list)
    default_update_weight: float = 1.0

    @classmethod
    def from_json(cls, path: str | Path) -> "DynamicAttentionSchedule":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DynamicAttentionSchedule":
        sample_size = payload.get("sample_size", {})
        sample_width = int(sample_size.get("width", payload.get("sample_width", 1280)))
        sample_height = int(sample_size.get("height", payload.get("sample_height", 704)))
        source_size = payload.get("source_size")
        inherited_source_size = (
            _as_pair(source_size, name="source_size") if source_size is not None else None
        )
        patch_size = tuple(int(x) for x in payload.get("patch_size", (1, 2, 2)))
        if len(patch_size) != 3:
            raise ValueError("patch_size must be [pt, ph, pw]")

        groups_payload = payload.get("token_groups", {})
        if isinstance(groups_payload, Mapping):
            groups_iter = groups_payload.get("groups", [])
        else:
            groups_iter = groups_payload
        token_groups = {group.name: group for group in (TokenGroupSpec.from_dict(x) for x in groups_iter)}
        regions = {
            region.name: region
            for region in (
                RegionSpec.from_dict(x, inherited_source_size=inherited_source_size)
                for x in payload.get("regions", [])
            )
        }
        phases = [PhaseSpec.from_dict(x) for x in payload.get("phases", [])]
        if not phases:
            phases = [PhaseSpec(name="all", active_regions=tuple(regions), active_groups=tuple(token_groups))]

        return cls(
            name=str(payload.get("name", "dynamic_attention_schedule")),
            sample_width=sample_width,
            sample_height=sample_height,
            patch_size=patch_size,
            token_groups=token_groups,
            regions=regions,
            phases=phases,
            default_update_weight=float(payload.get("default_update_weight", 1.0)),
        )

    def query_grid_from_latents(self, latent_shape: Sequence[int]) -> Tuple[int, int, int]:
        """Return the Wan token grid corresponding to latents [B, C, F, H, W]."""

        _, _, frames, height, width = [int(x) for x in latent_shape]
        pt, ph, pw = self.patch_size
        return (
            max(1, math.ceil(frames / max(pt, 1))),
            max(1, math.ceil(height / max(ph, 1))),
            max(1, math.ceil(width / max(pw, 1))),
        )

    @staticmethod
    def query_coordinates(
        grid_shape: Sequence[int],
        query_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        frames, height, width = [int(x) for x in grid_shape]
        idx = query_indices.to(dtype=torch.long)
        spatial = height * width
        frame = torch.div(idx, spatial, rounding_mode="floor").clamp_(0, max(frames - 1, 0))
        rem = torch.remainder(idx, spatial)
        y = torch.div(rem, width, rounding_mode="floor").clamp_(0, max(height - 1, 0))
        x = torch.remainder(rem, width).clamp_(0, max(width - 1, 0))
        return frame, y, x

    def phase_ids_for_frames(self, num_frames: int, device: torch.device) -> torch.Tensor:
        if num_frames <= 0:
            return torch.zeros((0,), device=device, dtype=torch.long)
        phase_ids = torch.full((num_frames,), -1, device=device, dtype=torch.long)
        phases_without_ranges = []
        for idx, phase in enumerate(self.phases):
            if phase.frame_range is None:
                phases_without_ranges.append((idx, phase))
                continue
            start_f, end_f = phase.frame_range
            start = int(math.floor(max(0.0, min(1.0, start_f)) * num_frames))
            end = int(math.ceil(max(0.0, min(1.0, end_f)) * num_frames))
            start = max(0, min(start, num_frames))
            end = max(start + 1, min(end, num_frames))
            phase_ids[start:end] = idx

        if phases_without_ranges:
            total = len(phases_without_ranges)
            for local_idx, (phase_idx, _) in enumerate(phases_without_ranges):
                start = int(math.floor(local_idx * num_frames / total))
                end = int(math.floor((local_idx + 1) * num_frames / total))
                phase_ids[start:max(start + 1, end)] = phase_idx

        if torch.any(phase_ids < 0):
            fallback = 0
            phase_ids = torch.where(phase_ids < 0, torch.full_like(phase_ids, fallback), phase_ids)
        return phase_ids

    def phase_ids_for_queries(self, grid_shape: Sequence[int], query_indices: torch.Tensor) -> torch.Tensor:
        frame_idx, _, _ = self.query_coordinates(grid_shape, query_indices)
        phase_ids = self.phase_ids_for_frames(int(grid_shape[0]), query_indices.device)
        return phase_ids.index_select(0, frame_idx)

    def _region_box_masks(
        self,
        grid_shape: Sequence[int],
        query_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        frames, height, width = [int(x) for x in grid_shape]
        _, y_idx, x_idx = self.query_coordinates(grid_shape, query_indices)
        dtype = torch.float32
        px = (x_idx.to(dtype) + 0.5) * (float(self.sample_width) / max(float(width), 1.0))
        py = (y_idx.to(dtype) + 0.5) * (float(self.sample_height) / max(float(height), 1.0))

        box_masks: Dict[str, torch.Tensor] = {}
        object_union = torch.zeros_like(px, dtype=torch.bool)
        for region in self.regions.values():
            xyxy = region.sample_xyxy(self.sample_width, self.sample_height)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            mask = (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
            box_masks[region.name] = mask
            if not region.outside_all_boxes:
                object_union = object_union | mask

        for region in self.regions.values():
            if region.outside_all_boxes:
                box_masks[region.name] = ~object_union
        return box_masks

    def build_cross_attention_bias(
        self,
        *,
        grid_shape: Sequence[int],
        seq_len: int,
        context_len: int,
        context_offset: int,
        token_group_indices: Mapping[str, Sequence[int]],
        step_index: int,
        total_steps: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        cond_batch_indices: Optional[Sequence[int]] = None,
    ) -> Optional[torch.Tensor]:
        """Build an SDPA attention bias [B, 1, seq_len, context_len].

        ``token_group_indices`` must use absolute prompt-context positions. The
        caller passes ``context_offset=257`` for Wan I2V text tokens and
        ``context_offset=0`` for normal T2V text context.
        """

        valid_len = min(int(seq_len), int(math.prod([int(x) for x in grid_shape])))
        if valid_len <= 0 or not token_group_indices:
            return None
        query_indices = torch.arange(valid_len, device=device, dtype=torch.long)
        phase_ids = self.phase_ids_for_queries(grid_shape, query_indices)
        region_masks = self._region_box_masks(grid_shape, query_indices)
        bias = torch.zeros((batch_size, 1, int(seq_len), int(context_len)), device=device, dtype=dtype)
        if cond_batch_indices is None:
            cond_batch_indices = range(batch_size)

        wrote = False
        for phase_idx, phase in enumerate(self.phases):
            phase_mask = phase_ids == phase_idx
            if not torch.any(phase_mask):
                continue
            for group_name, absolute_indices in token_group_indices.items():
                local_indices = [
                    int(idx) - int(context_offset)
                    for idx in absolute_indices
                    if int(context_offset) <= int(idx) < int(context_offset) + int(context_len)
                ]
                if not local_indices:
                    continue
                values = torch.full((valid_len,), phase.inactive_logit_bias, device=device, dtype=dtype)
                if group_name in phase.active_groups:
                    values = torch.full((valid_len,), phase.positive_logit_bias, device=device, dtype=dtype)
                region_name = self._region_for_group(group_name)
                if region_name:
                    own_mask = region_masks.get(region_name)
                    if own_mask is not None:
                        own_mask_f = own_mask.to(dtype)
                        if group_name in phase.active_groups:
                            values = values - phase.outside_box_penalty * (1.0 - own_mask_f)
                        else:
                            values = values - phase.competitor_box_penalty * own_mask_f
                    region = self.regions.get(region_name)
                    if region is not None:
                        values = values * float(region.attention_scale)
                values = torch.where(phase_mask, values, torch.zeros_like(values))
                idx = torch.tensor(local_indices, device=device, dtype=torch.long)
                for batch_idx in cond_batch_indices:
                    if int(batch_idx) < batch_size:
                        bias[int(batch_idx), 0, :valid_len, idx] += values[:, None]
                wrote = True
        return bias if wrote else None

    def build_timestep_tensor(
        self,
        *,
        base_timestep: torch.Tensor,
        timesteps: Optional[torch.Tensor],
        grid_shape: Sequence[int],
        seq_len: int,
        step_index: int,
        total_steps: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Return a Wan-compatible timestep tensor [B, seq_len]."""

        base = _timestep_matrix(base_timestep.to(device=device, dtype=dtype), batch_size, int(seq_len))
        result = base.clone()
        strength = max(0.0, min(float(strength), 1.0))
        if strength <= 0.0:
            return result
        if timesteps is not None:
            timesteps = timesteps.to(device=device, dtype=dtype).flatten()

        valid_len = min(int(seq_len), int(math.prod([int(x) for x in grid_shape])))
        if valid_len <= 0:
            return result
        query_indices = torch.arange(valid_len, device=device, dtype=torch.long)
        region_masks = self._region_box_masks(grid_shape, query_indices)
        for region_name, mask in region_masks.items():
            region = self.regions[region_name]
            if region.timestep_policy == "global":
                continue
            effective_step = region.effective_step(step_index, total_steps)
            if timesteps is not None and timesteps.numel() > 0:
                value = timesteps[max(0, min(effective_step, timesteps.numel() - 1))]
                result[:, :valid_len] = torch.where(mask[None, :], value, result[:, :valid_len])
            else:
                value = base[:, :valid_len]
                result[:, :valid_len] = torch.where(mask[None, :], value, result[:, :valid_len])
        if strength < 1.0:
            result = base + (result - base) * strength
        return result

    def build_update_mask(
        self,
        *,
        grid_shape: Sequence[int],
        latent_shape: Sequence[int],
        step_index: int,
        total_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return [B, 1, F, H, W] new-latent blending weights."""

        batch, _, latent_f, latent_h, latent_w = [int(x) for x in latent_shape]
        valid_len = int(math.prod([int(x) for x in grid_shape]))
        query_indices = torch.arange(valid_len, device=device, dtype=torch.long)
        region_masks = self._region_box_masks(grid_shape, query_indices)
        grid_values = torch.full((valid_len,), float(self.default_update_weight), device=device, dtype=dtype)
        for region_name, mask in region_masks.items():
            region = self.regions[region_name]
            weight = torch.as_tensor(region.update_weight(step_index, total_steps), device=device, dtype=dtype)
            grid_values = torch.where(mask, weight, grid_values)
        query_grid = grid_values.reshape(1, 1, int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
        update = F.interpolate(query_grid, size=(latent_f, latent_h, latent_w), mode="nearest")
        return update.expand(batch, 1, latent_f, latent_h, latent_w)

    def blend_latents(
        self,
        *,
        previous_latents: torch.Tensor,
        new_latents: torch.Tensor,
        step_index: int,
        total_steps: int,
    ) -> torch.Tensor:
        grid_shape = self.query_grid_from_latents(new_latents.shape)
        mask = self.build_update_mask(
            grid_shape=grid_shape,
            latent_shape=new_latents.shape,
            step_index=step_index,
            total_steps=total_steps,
            device=new_latents.device,
            dtype=new_latents.dtype,
        )
        previous = previous_latents.to(device=new_latents.device, dtype=new_latents.dtype)
        return mask * new_latents + (1.0 - mask) * previous

    def summary_rows(self, grid_shape: Sequence[int], total_steps: int) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for region in self.regions.values():
            start, end = region.step_window(total_steps)
            rows.append(
                {
                    "kind": "region",
                    "name": region.name,
                    "role": region.role,
                    "token_group": region.token_group or "",
                    "start_step": start,
                    "end_step": end,
                    "active_steps": max(0, end - start + 1),
                    "total_steps": total_steps,
                    "active_update_weight": region.active_update_weight,
                    "inactive_update_weight": region.inactive_update_weight,
                    "timestep_policy": region.timestep_policy,
                    "has_box": int(region.box_xyxy is not None or region.outside_all_boxes),
                }
            )
        phase_ids = self.phase_ids_for_frames(int(grid_shape[0]), torch.device("cpu"))
        for idx, phase in enumerate(self.phases):
            frames = torch.where(phase_ids == idx)[0].tolist()
            rows.append(
                {
                    "kind": "phase",
                    "name": phase.name,
                    "active_regions": " ".join(phase.active_regions),
                    "active_groups": " ".join(phase.active_groups),
                    "latent_frames": " ".join(str(int(x)) for x in frames),
                    "positive_logit_bias": phase.positive_logit_bias,
                    "inactive_logit_bias": phase.inactive_logit_bias,
                }
            )
        return rows

    def _region_for_group(self, group_name: str) -> Optional[str]:
        for region in self.regions.values():
            if region.token_group == group_name:
                return region.name
        if group_name in self.regions:
            return group_name
        return None


class DynamicTimestepWrapper:
    """Temporarily replace transformer.forward to emit per-token timesteps."""

    def __init__(
        self,
        transformer: Any,
        schedule: DynamicAttentionSchedule,
        *,
        timesteps: Optional[torch.Tensor] = None,
        total_steps: Optional[int] = None,
        strength: float = 1.0,
    ) -> None:
        self.transformer = transformer
        self.schedule = schedule
        self.timesteps = timesteps
        self.total_steps = total_steps
        self.strength = float(strength)
        self._original_forward = None

    def install(self) -> None:
        if self._original_forward is not None:
            return
        self._original_forward = self.transformer.forward
        original_forward = self._original_forward
        schedule = self.schedule
        wrapper = self

        def patched_forward(this, *args, **kwargs):
            args_list = list(args)
            x = kwargs.get("x")
            if x is None and args_list:
                x = args_list[0]
            if x is not None:
                t_value = kwargs.get("t")
                t_pos = None
                if t_value is None and len(args_list) >= 2:
                    t_value = args_list[1]
                    t_pos = 1
                seq_len = kwargs.get("seq_len")
                if seq_len is None and len(args_list) >= 4:
                    seq_len = args_list[3]
                if t_value is not None and seq_len is not None:
                    step_index = int(getattr(this, "current_steps", 0))
                    total_steps = int(wrapper.total_steps or getattr(this, "num_inference_steps", 0) or step_index + 1)
                    grid_shape = schedule.query_grid_from_latents(x.shape)
                    t_tensor = t_value if isinstance(t_value, torch.Tensor) else torch.as_tensor(t_value, device=x.device)
                    timestep = schedule.build_timestep_tensor(
                        base_timestep=t_tensor,
                        timesteps=wrapper.timesteps,
                        grid_shape=grid_shape,
                        seq_len=int(seq_len),
                        step_index=step_index,
                        total_steps=total_steps,
                        batch_size=int(x.shape[0]),
                        device=x.device,
                        dtype=t_tensor.dtype if isinstance(t_tensor, torch.Tensor) else torch.float32,
                        strength=wrapper.strength,
                    )
                    if "t" in kwargs:
                        kwargs["t"] = timestep
                    elif t_pos is not None:
                        args_list[t_pos] = timestep
            return original_forward(*args_list, **kwargs)

        self.transformer.forward = types.MethodType(patched_forward, self.transformer)

    def restore(self) -> None:
        if self._original_forward is not None:
            self.transformer.forward = self._original_forward
            self._original_forward = None

    def __enter__(self) -> "DynamicTimestepWrapper":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()


class CrossAttentionBiasController:
    """Temporarily patch Wan cross-attention with schedule-controlled logit bias."""

    def __init__(
        self,
        transformer: Any,
        schedule: DynamicAttentionSchedule,
        *,
        token_group_indices: Mapping[str, Sequence[int]],
        injection_steps: Sequence[int],
        injection_blocks: Sequence[int],
        total_steps: int,
        cond_only: bool = True,
        apply_bias: bool = True,
        capture_metrics: bool = False,
        max_metric_queries: int = 1024,
    ) -> None:
        self.transformer = transformer
        self.schedule = schedule
        self.token_group_indices = {str(k): [int(x) for x in v] for k, v in token_group_indices.items()}
        self.injection_steps = set(int(x) for x in injection_steps)
        self.injection_blocks = set(int(x) for x in injection_blocks)
        self.total_steps = int(total_steps)
        self.cond_only = bool(cond_only)
        self.apply_bias = bool(apply_bias)
        self.capture_metrics = bool(capture_metrics)
        self.max_metric_queries = max(1, int(max_metric_queries))
        self.metric_rows: List[Dict[str, Any]] = []
        self._hooks = []
        self._original_forwards = []
        self._branch_tracker_step = None
        self._branch_block0_count = 0
        self._current_branch_is_cond = True

    def install(self) -> None:
        if self._original_forwards:
            return
        for block_idx in sorted(self.injection_blocks | {0}):
            if block_idx < 0 or block_idx >= len(self.transformer.blocks):
                continue
            block = self.transformer.blocks[block_idx]

            def block_pre_hook(mod, hook_args, hook_kwargs, idx=block_idx):
                self._prepare_cross_attention(mod.cross_attn, hook_args, hook_kwargs, idx)

            self._hooks.append(block.register_forward_pre_hook(block_pre_hook, with_kwargs=True))
            if block_idx not in self.injection_blocks:
                continue
            module = block.cross_attn
            original_forward = module.forward

            def patched_forward(mod, x, context, context_lens, dtype=torch.bfloat16, t=0):
                return self._forward(mod, x, context, context_lens, dtype=dtype, t=t)

            module.forward = types.MethodType(patched_forward, module)
            self._original_forwards.append((module, original_forward))

    def restore(self) -> None:
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._original_forwards = []
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __enter__(self) -> "CrossAttentionBiasController":
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()

    def _prepare_cross_attention(self, module, hook_args, hook_kwargs, block_idx: int) -> None:
        x = hook_args[0] if hook_args else hook_kwargs.get("x")
        grid_sizes = hook_kwargs.get("grid_sizes")
        if x is None or grid_sizes is None:
            return
        step = int(getattr(self.transformer, "current_steps", -1))
        self._update_branch_tracker(step, block_idx)
        branch = 1 if x.size(0) > 1 else 0
        branch = min(branch, grid_sizes.size(0) - 1)
        grid = tuple(int(v) for v in grid_sizes[branch].detach().cpu().tolist())
        module._dynamic_schedule_grid = grid
        module._dynamic_schedule_valid_len = int(math.prod(grid))
        module._dynamic_schedule_step = step
        module._dynamic_schedule_block = int(block_idx)
        module._dynamic_schedule_cond = bool(self._current_branch_is_cond)

    def _update_branch_tracker(self, step: int, block_idx: int) -> None:
        if step != self._branch_tracker_step:
            self._branch_tracker_step = step
            self._branch_block0_count = 0
            self._current_branch_is_cond = True
        if int(block_idx) == 0:
            self._branch_block0_count += 1
            self._current_branch_is_cond = self._branch_block0_count % 2 == 1

    def _should_process(self, module) -> bool:
        if self.cond_only and not bool(getattr(module, "_dynamic_schedule_cond", True)):
            return False
        step = int(getattr(module, "_dynamic_schedule_step", -1))
        block = int(getattr(module, "_dynamic_schedule_block", -1))
        return step in self.injection_steps and block in self.injection_blocks

    @staticmethod
    def _sdpa(q, k, v, attn_mask=None):
        q_in = q.permute(0, 2, 1, 3).contiguous()
        k_in = k.permute(0, 2, 1, 3).contiguous()
        v_in = v.permute(0, 2, 1, 3).contiguous()
        out = F.scaled_dot_product_attention(
            q_in,
            k_in,
            v_in,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        return out.permute(0, 2, 1, 3).contiguous()

    def _branch_forward(self, module, q, k, v, context_offset: int, dtype: torch.dtype):
        attn_mask = None
        should_process = self._should_process(module)
        if should_process and self.apply_bias:
            grid = getattr(module, "_dynamic_schedule_grid", None)
            valid_len = int(min(getattr(module, "_dynamic_schedule_valid_len", q.size(1)), q.size(1)))
            if grid is not None and valid_len > 0:
                cond_batch_indices = range(q.size(0))
                attn_mask = self.schedule.build_cross_attention_bias(
                    grid_shape=grid,
                    seq_len=q.size(1),
                    context_len=k.size(1),
                    context_offset=int(context_offset),
                    token_group_indices=self.token_group_indices,
                    step_index=int(getattr(module, "_dynamic_schedule_step", 0)),
                    total_steps=self.total_steps,
                    batch_size=q.size(0),
                    device=q.device,
                    dtype=q.dtype,
                    cond_batch_indices=cond_batch_indices,
                )
        if should_process and self.capture_metrics:
            self._record_attention_metrics(module, q, k, attn_mask, context_offset)
        out = self._sdpa(q, k, v, attn_mask=attn_mask).to(dtype)
        return out

    def _local_indices_for_group(self, group_name: str, context_offset: int, context_len: int) -> List[int]:
        offset = int(context_offset)
        return [
            int(idx) - offset
            for idx in self.token_group_indices.get(group_name, [])
            if offset <= int(idx) < offset + int(context_len)
        ]

    def _record_attention_metrics(self, module, q, k, attn_mask, context_offset: int) -> None:
        grid = getattr(module, "_dynamic_schedule_grid", None)
        if grid is None or not self.token_group_indices:
            return
        valid_len = int(min(getattr(module, "_dynamic_schedule_valid_len", q.size(1)), q.size(1)))
        if valid_len <= 0 or k.size(1) <= 0:
            return
        step = int(getattr(module, "_dynamic_schedule_step", -1))
        block = int(getattr(module, "_dynamic_schedule_block", -1))
        stride = max(1, int(math.ceil(valid_len / float(self.max_metric_queries))))
        query_indices = torch.arange(0, valid_len, stride, device=q.device, dtype=torch.long)
        if query_indices.numel() <= 0:
            return
        phase_ids = self.schedule.phase_ids_for_queries(grid, query_indices)
        with torch.no_grad():
            q_sample = q[0, query_indices].float()
            k_sample = k[0].float()
            scores = torch.einsum("qhd,khd->hqk", q_sample, k_sample) * (float(q_sample.shape[-1]) ** -0.5)
            if attn_mask is not None:
                scores = scores + attn_mask[0, 0, query_indices].float().unsqueeze(0)
            attn = torch.softmax(scores, dim=-1)
            group_local_indices = {
                name: self._local_indices_for_group(name, context_offset, k.size(1))
                for name in self.token_group_indices
            }
            group_local_indices = {name: idx for name, idx in group_local_indices.items() if idx}
            if not group_local_indices:
                return

            for phase_idx, phase in enumerate(self.schedule.phases):
                phase_query_mask = phase_ids == phase_idx
                phase_query_count = int(phase_query_mask.sum().item())
                if phase_query_count <= 0:
                    continue
                phase_attn = attn[:, phase_query_mask]
                rgb_sum = 0.0
                group_values: Dict[str, float] = {}
                for group_name, local_indices in group_local_indices.items():
                    idx_tensor = torch.tensor(local_indices, device=attn.device, dtype=torch.long)
                    value = phase_attn.index_select(dim=-1, index=idx_tensor).sum(dim=-1).mean().item()
                    value = float(value)
                    group_values[group_name] = value
                    if group_name in phase.active_groups or self.schedule._region_for_group(group_name):
                        rgb_sum += value
                for group_name, value in group_values.items():
                    competitor_values = [v for name, v in group_values.items() if name != group_name]
                    self.metric_rows.append(
                        {
                            "step": step,
                            "block": block,
                            "context_offset": int(context_offset),
                            "phase_index": int(phase_idx),
                            "phase": phase.name,
                            "group": group_name,
                            "active_group": int(group_name in phase.active_groups),
                            "mean_attention": value,
                            "share_in_groups": value / max(rgb_sum, 1e-12),
                            "margin_vs_best_competitor": value - max(competitor_values or [0.0]),
                            "sampled_queries": int(query_indices.numel()),
                            "phase_sampled_queries": phase_query_count,
                            "num_group_tokens": len(group_local_indices[group_name]),
                            "bias_applied": int(attn_mask is not None),
                        }
                    )

    def _forward(self, module, x, context, context_lens, dtype=torch.bfloat16, t=0):
        batch, seq_len, heads, head_dim = x.size(0), x.size(1), module.num_heads, module.head_dim
        q = module.norm_q(module.q(x.to(dtype))).view(batch, seq_len, heads, head_dim)

        if hasattr(module, "k_img") and hasattr(module, "v_img") and context.size(1) > 257:
            context_img = context[:, :257]
            context_text = context[:, 257:]
            k_img = module.norm_k_img(module.k_img(context_img.to(dtype))).view(batch, -1, heads, head_dim)
            v_img = module.v_img(context_img.to(dtype)).view(batch, -1, heads, head_dim)
            img_x = self._branch_forward(module, q, k_img, v_img, 0, dtype)
            if context_text.size(1) > 0:
                k_text = module.norm_k(module.k(context_text.to(dtype))).view(batch, -1, heads, head_dim)
                v_text = module.v(context_text.to(dtype)).view(batch, -1, heads, head_dim)
                text_x = self._branch_forward(module, q, k_text, v_text, 257, dtype)
            else:
                text_x = torch.zeros_like(img_x)
            return module.o((img_x + text_x).flatten(2))

        k = module.norm_k(module.k(context.to(dtype))).view(batch, -1, heads, head_dim)
        v = module.v(context.to(dtype)).view(batch, -1, heads, head_dim)
        out = self._branch_forward(module, q, k, v, 0, dtype)
        return module.o(out.flatten(2))


class LatentUpdateGateCallback:
    """Diffusers callback that blends inactive regions with previous latents."""

    tensor_inputs = ["latents"]

    def __init__(
        self,
        schedule: DynamicAttentionSchedule,
        *,
        total_steps: int,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> None:
        self.schedule = schedule
        self.total_steps = int(total_steps)
        self.previous_latents = initial_latents.detach().clone() if initial_latents is not None else None

    def __call__(self, pipeline, step_index: int, timestep, callback_kwargs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        latents = callback_kwargs["latents"]
        if self.previous_latents is None:
            self.previous_latents = latents.detach().clone()
            return {"latents": latents}
        blended = self.schedule.blend_latents(
            previous_latents=self.previous_latents,
            new_latents=latents,
            step_index=int(step_index),
            total_steps=self.total_steps,
        )
        self.previous_latents = blended.detach().clone()
        return {"latents": blended}


def write_csv_rows(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
