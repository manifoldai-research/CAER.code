"""Training-side helpers for region-temporal diffusion forcing.

The exp07 schedule exporter writes dense latent/query-grid targets. This module
loads those targets and applies them to ordinary flow-matching training code
without depending on a specific trainer implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _as_target_size(shape: Sequence[int]) -> Tuple[int, int, int]:
    if len(shape) < 3:
        raise ValueError("target shape must contain at least F,H,W")
    return int(shape[-3]), int(shape[-2]), int(shape[-1])


def _safe_tensor(value: Any, *, device: torch.device, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


@dataclass(frozen=True)
class TrainingBudgetTargets:
    """Dense schedule-derived targets for a single training sample or template."""

    region_ids: np.ndarray
    phase_ids: np.ndarray
    active_object_mask: np.ndarray
    action_mask: np.ndarray
    loss_weights: np.ndarray
    update_budget: np.ndarray
    timestep_budget: np.ndarray
    region_names: Tuple[str, ...]
    action_frame_indices: Tuple[int, ...] = ()

    @classmethod
    def from_npz(cls, path: str | Path) -> "TrainingBudgetTargets":
        payload = np.load(path, allow_pickle=False)
        return cls(
            region_ids=np.asarray(payload["region_ids"]),
            phase_ids=np.asarray(payload["phase_ids"]),
            active_object_mask=np.asarray(payload["active_object_mask"]).astype(bool),
            action_mask=np.asarray(payload["action_mask"]).astype(bool),
            loss_weights=np.asarray(payload["loss_weights"], dtype=np.float32),
            update_budget=np.asarray(payload["update_budget"], dtype=np.float32),
            timestep_budget=np.asarray(payload["timestep_budget"], dtype=np.int64),
            region_names=tuple(str(x) for x in np.asarray(payload["region_names"]).tolist()),
            action_frame_indices=(
                tuple(int(x) for x in np.asarray(payload["action_frame_indices"]).tolist())
                if "action_frame_indices" in payload.files
                else ()
            ),
        )

    @property
    def query_grid_shape(self) -> Tuple[int, int, int]:
        return _as_target_size(self.loss_weights.shape)

    @property
    def num_budget_steps(self) -> int:
        return int(self.timestep_budget.shape[0])

    def loss_weight_tensor(
        self,
        latent_shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype,
        normalize: bool = True,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Return broadcastable loss weights [B,1,F,H,W] for latent losses."""

        batch = int(latent_shape[0])
        target_size = _as_target_size(latent_shape)
        weights = _safe_tensor(self.loss_weights, device=device, dtype=dtype)[None, None]
        if tuple(int(x) for x in weights.shape[-3:]) != tuple(target_size):
            weights = F.interpolate(weights, size=target_size, mode="nearest")
        if normalize:
            mean = weights.mean().clamp_min(torch.finfo(dtype).eps if dtype.is_floating_point else 1e-6)
            weights = weights / mean
        strength = max(0.0, min(float(strength), 1.0))
        if strength < 1.0:
            weights = 1.0 + (weights - 1.0) * strength
        return weights.expand(batch, 1, *target_size)

    def mask_tensor(
        self,
        name: str,
        latent_shape: Sequence[int],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Return a broadcastable mask [B,1,F,H,W]."""

        if name == "active_object":
            values = self.active_object_mask
        elif name == "action":
            values = self.action_mask
        else:
            raise ValueError(f"unknown mask target: {name}")
        batch = int(latent_shape[0])
        target_size = _as_target_size(latent_shape)
        mask = _safe_tensor(values.astype(np.float32), device=device, dtype=dtype)[None, None]
        if tuple(int(x) for x in mask.shape[-3:]) != tuple(target_size):
            mask = F.interpolate(mask, size=target_size, mode="nearest")
        return mask.expand(batch, 1, *target_size)

    def region_id_tensor(
        self,
        latent_shape: Sequence[int],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return nearest-neighbor region ids [B,F,H,W] for a latent tensor."""

        batch = int(latent_shape[0])
        target_size = _as_target_size(latent_shape)
        values = _safe_tensor(self.region_ids.astype(np.float32), device=device)[None, None]
        if tuple(int(x) for x in values.shape[-3:]) != tuple(target_size):
            values = F.interpolate(values, size=target_size, mode="nearest")
        return values[:, 0].round().to(dtype=torch.long).expand(batch, *target_size)

    def per_token_timesteps(
        self,
        *,
        step_indices: torch.Tensor,
        scheduler_timesteps: Optional[torch.Tensor] = None,
        seq_len: Optional[int] = None,
        device: torch.device,
        dtype: torch.dtype,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Return Wan-compatible per-token timesteps [B, seq_len].

        ``step_indices`` are integer indices into the training/scheduler timestep
        list, not raw timestep values. ``scheduler_timesteps`` maps those indices
        to the actual values passed as ``t``. If omitted, the returned tensor uses
        step ids directly.
        """

        step_indices = step_indices.to(device=device, dtype=torch.long).flatten()
        batch = int(step_indices.numel())
        budget_steps = int(self.timestep_budget.shape[0])
        query_count = int(np.prod(self.timestep_budget.shape[1:]))
        if seq_len is None:
            seq_len = query_count
        seq_len = int(seq_len)
        budget = torch.as_tensor(self.timestep_budget, device=device, dtype=torch.long)
        clamped = step_indices.clamp(0, max(budget_steps - 1, 0))
        effective_ids = budget.index_select(0, clamped).reshape(batch, -1)

        if scheduler_timesteps is not None:
            scheduler_timesteps = scheduler_timesteps.to(device=device, dtype=dtype).flatten()
            max_idx = max(int(scheduler_timesteps.numel()) - 1, 0)
            values = scheduler_timesteps.index_select(0, effective_ids.clamp(0, max_idx).reshape(-1)).reshape(batch, -1)
            base = scheduler_timesteps.index_select(0, clamped.clamp(0, max_idx))[:, None]
        else:
            values = effective_ids.to(dtype=dtype)
            base = clamped.to(dtype=dtype)[:, None]

        if values.shape[1] < seq_len:
            pad = base.expand(batch, seq_len - values.shape[1])
            values = torch.cat([values, pad], dim=1)
        else:
            values = values[:, :seq_len]
        base_matrix = base.expand(batch, seq_len)
        strength = max(0.0, min(float(strength), 1.0))
        if strength < 1.0:
            values = base_matrix + (values - base_matrix) * strength
        return values.to(dtype=dtype)

    def summary(self) -> Dict[str, Any]:
        return {
            "query_grid_shape": list(self.query_grid_shape),
            "num_budget_steps": self.num_budget_steps,
            "region_names": list(self.region_names),
            "active_object_ratio": float(np.asarray(self.active_object_mask).mean()),
            "action_ratio": float(np.asarray(self.action_mask).mean()),
            "loss_weight_mean": float(np.asarray(self.loss_weights).mean()),
            "loss_weight_min": float(np.asarray(self.loss_weights).min()),
            "loss_weight_max": float(np.asarray(self.loss_weights).max()),
        }


def region_temporal_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    budget: Optional[TrainingBudgetTargets] = None,
    sigma_weighting: Optional[torch.Tensor] = None,
    threshold: float = 50.0,
    budget_strength: float = 1.0,
    normalize_budget: bool = True,
) -> torch.Tensor:
    """MSE loss compatible with Wan flow-matching trainers.

    This mirrors the existing training scripts' masked MSE, then optionally
    multiplies by schedule-derived region-temporal loss weights. If the budget
    weights are normalized, enabling this loss should not change the global loss
    scale dramatically.
    """

    prediction = prediction.float()
    target = target.float()
    diff = prediction - target
    loss = F.mse_loss(prediction, target, reduction="none")
    loss = loss * (diff.abs() <= float(threshold)).float()
    if sigma_weighting is not None:
        loss = loss * sigma_weighting.to(device=loss.device, dtype=loss.dtype)
    if budget is not None:
        weights = budget.loss_weight_tensor(
            loss.shape,
            device=loss.device,
            dtype=loss.dtype,
            normalize=normalize_budget,
            strength=budget_strength,
        )
        loss = loss * weights
    return loss.mean()


def region_temporal_loss_breakdown(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    budget: TrainingBudgetTargets,
    sigma_weighting: Optional[torch.Tensor] = None,
    threshold: float = 50.0,
    budget_strength: float = 1.0,
    normalize_budget: bool = True,
) -> List[Dict[str, float | int | str]]:
    """Return region/action loss diagnostics for a budget-weighted MSE."""

    prediction = prediction.float()
    target = target.float()
    diff = prediction - target
    base_loss = F.mse_loss(prediction, target, reduction="none")
    base_loss = base_loss * (diff.abs() <= float(threshold)).float()
    if sigma_weighting is not None:
        base_loss = base_loss * sigma_weighting.to(device=base_loss.device, dtype=base_loss.dtype)
    weights = budget.loss_weight_tensor(
        base_loss.shape,
        device=base_loss.device,
        dtype=base_loss.dtype,
        normalize=normalize_budget,
        strength=budget_strength,
    )
    weighted_loss = base_loss * weights
    total_weighted = weighted_loss.sum().clamp_min(1e-12)

    def row_for_mask(name: str, mask: torch.Tensor) -> Dict[str, float | int | str]:
        mask = mask.to(device=base_loss.device, dtype=torch.bool)
        if mask.ndim == 4:
            mask = mask[:, None]
        mask = mask.expand_as(base_loss)
        count = int(mask.sum().item())
        if count <= 0:
            return {
                "name": name,
                "elements": 0,
                "element_ratio": 0.0,
                "mean_weight": 0.0,
                "unweighted_loss_mean": 0.0,
                "weighted_loss_mean": 0.0,
                "weighted_loss_share": 0.0,
            }
        return {
            "name": name,
            "elements": count,
            "element_ratio": float(mask.float().mean().item()),
            "mean_weight": float(weights.expand_as(base_loss)[mask].mean().item()),
            "unweighted_loss_mean": float(base_loss[mask].mean().item()),
            "weighted_loss_mean": float(weighted_loss[mask].mean().item()),
            "weighted_loss_share": float((weighted_loss[mask].sum() / total_weighted).item()),
        }

    rows: List[Dict[str, float | int | str]] = []
    region_ids = budget.region_id_tensor(base_loss.shape, device=base_loss.device)
    for region_id, name in enumerate(budget.region_names):
        rows.append(row_for_mask(f"region:{name}", region_ids == int(region_id)))
    rows.append(row_for_mask("mask:active_object", budget.mask_tensor("active_object", base_loss.shape, device=base_loss.device).bool()))
    rows.append(row_for_mask("mask:action", budget.mask_tensor("action", base_loss.shape, device=base_loss.device).bool()))
    return rows


def _attention_as_probs(attention: torch.Tensor, *, input_is_logits: bool) -> torch.Tensor:
    """Normalize an attention tensor to [B, H, Q, K] probabilities."""

    if attention.ndim == 3:
        attention = attention[:, None]
    elif attention.ndim != 4:
        raise ValueError(f"attention must have shape [B,Q,K] or [B,H,Q,K], got {tuple(attention.shape)}")
    attention = attention.float()
    if input_is_logits:
        attention = attention.softmax(dim=-1)
    return attention


def _token_mask_tensor(
    token_mask: torch.Tensor | Sequence[int] | np.ndarray,
    *,
    batch: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a boolean prompt-token mask [B, K]."""

    if isinstance(token_mask, torch.Tensor):
        mask = token_mask.to(device=device)
    else:
        mask = torch.as_tensor(token_mask, device=device)
    if mask.ndim == 1:
        if mask.dtype == torch.bool:
            values = mask
        else:
            values = torch.zeros(int(key_len), device=device, dtype=torch.bool)
            indices = mask.to(dtype=torch.long)
            indices = indices[(indices >= 0) & (indices < int(key_len))]
            values[indices] = True
        if values.numel() < key_len:
            values = F.pad(values, (0, key_len - values.numel()), value=False)
        elif values.numel() > key_len:
            values = values[:key_len]
        return values[None].expand(batch, key_len)
    if mask.ndim != 2:
        raise ValueError(f"token mask must have shape [K] or [B,K], got {tuple(mask.shape)}")
    if mask.shape[0] == 1 and batch > 1:
        mask = mask.expand(batch, mask.shape[1])
    if int(mask.shape[0]) != int(batch):
        raise ValueError(f"token mask batch {mask.shape[0]} does not match attention batch {batch}")
    mask = mask.to(dtype=torch.bool)
    if mask.shape[1] < key_len:
        mask = F.pad(mask, (0, key_len - mask.shape[1]), value=False)
    elif mask.shape[1] > key_len:
        mask = mask[:, :key_len]
    return mask


def _budget_query_shape_for_attention(
    budget: TrainingBudgetTargets,
    query_shape: Optional[Sequence[int]],
    query_len: int,
) -> Tuple[int, int, int]:
    if query_shape is not None:
        return _as_target_size(tuple(int(v) for v in query_shape))
    default = budget.query_grid_shape
    if int(np.prod(default)) == int(query_len):
        return default
    return default


def _resize_flat_mask_to_query_len(mask: torch.Tensor, query_len: int) -> torch.Tensor:
    """Pad or truncate a flattened [B, Q] mask when a model has extra context tokens."""

    if mask.shape[1] < query_len:
        return F.pad(mask, (0, int(query_len) - int(mask.shape[1])), value=False)
    if mask.shape[1] > query_len:
        return mask[:, :query_len]
    return mask


def _active_region_query_masks(
    budget: TrainingBudgetTargets,
    *,
    batch: int,
    query_len: int,
    device: torch.device,
    query_shape: Optional[Sequence[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Return flattened active-query masks keyed by region name."""

    target_shape = _budget_query_shape_for_attention(budget, query_shape, query_len)
    latent_shape = (batch, 1, *target_shape)
    active = budget.mask_tensor("active_object", latent_shape, device=device, dtype=torch.float32)[:, 0].bool()
    region_ids = budget.region_id_tensor(latent_shape, device=device)
    masks: Dict[str, torch.Tensor] = {}
    for region_id, name in enumerate(budget.region_names):
        if name == "background":
            continue
        mask = (active & (region_ids == int(region_id))).reshape(batch, -1)
        masks[name] = _resize_flat_mask_to_query_len(mask, query_len)
    return masks


def _active_object_query_mask(
    budget: TrainingBudgetTargets,
    *,
    batch: int,
    query_len: int,
    device: torch.device,
    query_shape: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    target_shape = _budget_query_shape_for_attention(budget, query_shape, query_len)
    latent_shape = (batch, 1, *target_shape)
    mask = budget.mask_tensor("active_object", latent_shape, device=device, dtype=torch.float32)[:, 0].bool()
    return _resize_flat_mask_to_query_len(mask.reshape(batch, -1), query_len)


def region_temporal_prompt_attention_loss(
    attention: torch.Tensor,
    *,
    budget: TrainingBudgetTargets,
    token_masks: Mapping[str, torch.Tensor | Sequence[int] | np.ndarray] | torch.Tensor | Sequence[int] | np.ndarray,
    query_shape: Optional[Sequence[int]] = None,
    input_is_logits: bool = False,
    active_target_mass: float = 0.35,
    inactive_ceiling: float = 0.05,
    active_weight: float = 1.0,
    inactive_weight: float = 0.25,
) -> torch.Tensor:
    """Train cross-attention to follow the object-time schedule.

    ``attention`` is a captured text cross-attention map with shape ``[B,Q,K]``
    or ``[B,H,Q,K]``. ``token_masks`` can either be:

    - a single prompt-token mask/index list for all active objects; or
    - a mapping from ``region_name`` to prompt-token mask/index list, e.g.
      ``{"red_block": [17, 18], "green_block": [23, 24]}``.

    The loss is intentionally a soft mass constraint instead of a hard
    one-hot target: active object query tokens should allocate at least
    ``active_target_mass`` attention to their matching prompt tokens, while
    non-active query tokens are only weakly discouraged from leaking too much
    mass to those same tokens. A small external loss weight should be used in a
    full trainer because text attention is only one part of the denoising model.
    """

    probs = _attention_as_probs(attention, input_is_logits=input_is_logits)
    batch, _, query_len, key_len = probs.shape
    device = probs.device

    if isinstance(token_masks, Mapping):
        query_masks = _active_region_query_masks(
            budget,
            batch=batch,
            query_len=query_len,
            device=device,
            query_shape=query_shape,
        )
        items = [(name, mask, token_masks[name]) for name, mask in query_masks.items() if name in token_masks]
    else:
        items = [
            (
                "active_object",
                _active_object_query_mask(
                    budget,
                    batch=batch,
                    query_len=query_len,
                    device=device,
                    query_shape=query_shape,
                ),
                token_masks,
            )
        ]

    losses: List[torch.Tensor] = []
    for _, query_mask, token_mask in items:
        prompt_mask = _token_mask_tensor(token_mask, batch=batch, key_len=key_len, device=device)
        if not bool(prompt_mask.any().item()) or not bool(query_mask.any().item()):
            continue
        token_mass = (probs * prompt_mask[:, None, None, :].to(dtype=probs.dtype)).sum(dim=-1).mean(dim=1)
        active_mask = query_mask.to(device=device, dtype=torch.bool)
        active_mass = token_mass[active_mask]
        if active_mass.numel() > 0 and float(active_weight) > 0.0:
            active_target = torch.as_tensor(float(active_target_mass), device=device, dtype=token_mass.dtype)
            active_loss = F.relu(active_target - active_mass).square().mean()
            losses.append(active_loss * float(active_weight))
        if float(inactive_weight) > 0.0:
            inactive_mask = ~active_mask
            inactive_mass = token_mass[inactive_mask]
            if inactive_mass.numel() > 0:
                ceiling = torch.as_tensor(float(inactive_ceiling), device=device, dtype=token_mass.dtype)
                inactive_loss = F.relu(inactive_mass - ceiling).square().mean()
                losses.append(inactive_loss * float(inactive_weight))

    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).sum()


def region_temporal_attention_breakdown(
    attention: torch.Tensor,
    *,
    budget: TrainingBudgetTargets,
    token_masks: Mapping[str, torch.Tensor | Sequence[int] | np.ndarray] | torch.Tensor | Sequence[int] | np.ndarray,
    query_shape: Optional[Sequence[int]] = None,
    input_is_logits: bool = False,
    active_target_mass: float = 0.35,
    inactive_ceiling: float = 0.05,
) -> List[Dict[str, float | int | str]]:
    """Return prompt-attention alignment diagnostics for reports/smoke tests."""

    probs = _attention_as_probs(attention, input_is_logits=input_is_logits)
    batch, _, query_len, key_len = probs.shape
    device = probs.device
    if isinstance(token_masks, Mapping):
        query_masks = _active_region_query_masks(
            budget,
            batch=batch,
            query_len=query_len,
            device=device,
            query_shape=query_shape,
        )
        items = [(name, mask, token_masks[name]) for name, mask in query_masks.items() if name in token_masks]
    else:
        items = [
            (
                "active_object",
                _active_object_query_mask(
                    budget,
                    batch=batch,
                    query_len=query_len,
                    device=device,
                    query_shape=query_shape,
                ),
                token_masks,
            )
        ]

    rows: List[Dict[str, float | int | str]] = []
    for name, query_mask, token_mask in items:
        prompt_mask = _token_mask_tensor(token_mask, batch=batch, key_len=key_len, device=device)
        token_mass = (probs * prompt_mask[:, None, None, :].to(dtype=probs.dtype)).sum(dim=-1).mean(dim=1)
        active_mask = query_mask.to(device=device, dtype=torch.bool)
        inactive_mask = ~active_mask
        active_mass = token_mass[active_mask]
        inactive_mass = token_mass[inactive_mask]
        active_loss = (
            F.relu(torch.as_tensor(float(active_target_mass), device=device, dtype=token_mass.dtype) - active_mass)
            .square()
            .mean()
            if active_mass.numel() > 0
            else token_mass.sum() * 0.0
        )
        inactive_loss = (
            F.relu(inactive_mass - torch.as_tensor(float(inactive_ceiling), device=device, dtype=token_mass.dtype))
            .square()
            .mean()
            if inactive_mass.numel() > 0
            else token_mass.sum() * 0.0
        )
        rows.append(
            {
                "name": name,
                "active_queries": int(active_mask.sum().item()),
                "inactive_queries": int(inactive_mask.sum().item()),
                "prompt_tokens": int(prompt_mask[0].sum().item()) if prompt_mask.numel() else 0,
                "active_mass_mean": float(active_mass.mean().item()) if active_mass.numel() > 0 else 0.0,
                "inactive_mass_mean": float(inactive_mass.mean().item()) if inactive_mass.numel() > 0 else 0.0,
                "active_target_mass": float(active_target_mass),
                "inactive_ceiling": float(inactive_ceiling),
                "active_loss": float(active_loss.item()),
                "inactive_loss": float(inactive_loss.item()),
            }
        )
    return rows


def _region_name_to_default_phrase(name: str) -> str:
    return str(name).replace("_", " ").strip()


def _phrase_values_for_region(
    region_name: str,
    phrases_by_region: Optional[Mapping[str, str | Sequence[str]]] = None,
) -> List[str]:
    if phrases_by_region is None or region_name not in phrases_by_region:
        return [_region_name_to_default_phrase(region_name)]
    values = phrases_by_region[region_name]
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _find_subsequence_positions(sequence: Sequence[int], pattern: Sequence[int]) -> List[int]:
    pattern = [int(x) for x in pattern if int(x) >= 0]
    if not pattern or len(pattern) > len(sequence):
        return []
    positions: List[int] = []
    last = len(sequence) - len(pattern)
    for start in range(last + 1):
        if all(int(sequence[start + offset]) == int(value) for offset, value in enumerate(pattern)):
            positions.extend(range(start, start + len(pattern)))
    return positions


def build_region_prompt_token_masks(
    input_ids: torch.Tensor,
    tokenizer: Any,
    region_names: Sequence[str],
    *,
    phrases_by_region: Optional[Mapping[str, str | Sequence[str]]] = None,
    context_len: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Build prompt-token masks for region names by phrase-token matching.

    ``input_ids`` should be the padded tokenizer ids for the training prompt
    batch. Each region defaults to the phrase obtained by replacing underscores
    with spaces, e.g. ``long_blue_block -> long blue block``. A phrase map can
    override that default for generic prompts or synonyms.
    """

    if input_ids.ndim == 1:
        input_ids = input_ids[None]
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [B,L], got {tuple(input_ids.shape)}")
    batch, prompt_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    key_len = int(context_len or prompt_len)
    target_device = device or input_ids.device
    input_ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.long)
    masks: Dict[str, torch.Tensor] = {}
    special_ids = set()
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            special_ids.add(int(value))

    for region_name in region_names:
        if region_name == "background":
            continue
        mask = torch.zeros(batch, key_len, dtype=torch.bool)
        phrase_token_lists: List[List[int]] = []
        for phrase in _phrase_values_for_region(region_name, phrases_by_region):
            encoded = tokenizer(
                phrase,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            token_ids = [int(x) for x in token_ids if int(x) not in special_ids]
            if token_ids:
                phrase_token_lists.append(token_ids)

        for batch_idx in range(batch):
            sequence = [int(x) for x in input_ids_cpu[batch_idx].tolist()]
            for token_ids in phrase_token_lists:
                positions = _find_subsequence_positions(sequence, token_ids)
                for position in positions:
                    if 0 <= position < key_len:
                        mask[batch_idx, position] = True
        masks[region_name] = mask.to(device=target_device)
    return masks


def parse_attention_layer_indices(spec: str | Sequence[int] | None, num_layers: int) -> List[int]:
    """Parse layer selections such as ``last4``, ``all``, or ``8-17,23``."""

    if spec is None:
        return []
    if not isinstance(spec, str):
        return sorted({int(x) for x in spec if 0 <= int(x) < int(num_layers)})
    spec = spec.strip().lower()
    if not spec:
        return []
    if spec == "all":
        return list(range(int(num_layers)))
    if spec.startswith("last"):
        count_text = spec[4:]
        count = int(count_text) if count_text else 1
        start = max(0, int(num_layers) - max(0, count))
        return list(range(start, int(num_layers)))
    indices: List[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start, end = int(start_text), int(end_text)
            step = 1 if end >= start else -1
            indices.extend(range(start, end + step, step))
        else:
            indices.append(int(chunk))
    return sorted({idx for idx in indices if 0 <= idx < int(num_layers)})


def _select_evenly_spaced(indices: torch.Tensor, max_count: int) -> torch.Tensor:
    indices = indices.flatten().to(dtype=torch.long)
    if max_count <= 0 or indices.numel() <= max_count:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, steps=int(max_count), device=indices.device)
    positions = positions.round().to(dtype=torch.long).unique(sorted=True)
    return indices.index_select(0, positions)


class ObjectTemporalCrossAttentionRecorder:
    """Sample Wan cross-attention logits for object-temporal training loss.

    The recorder registers lightweight forward hooks on selected Wan cross-
    attention modules. It recomputes q/k for a small, deterministic set of
    active-object and inactive query tokens, so it avoids materializing full
    ``[B, heads, all_video_tokens, all_text_tokens]`` attention maps.

    This first implementation is intended for non-checkpointed forward passes:
    the hook loss must be produced during the same forward graph that the
    trainer later adds to the main loss.
    """

    def __init__(
        self,
        budget: TrainingBudgetTargets,
        *,
        max_active_queries_per_region: int = 64,
        max_inactive_queries_per_region: int = 64,
        active_target_mass: float = 0.35,
        inactive_ceiling: float = 0.08,
        inactive_weight: float = 0.25,
    ) -> None:
        self.budget = budget
        self.max_active_queries_per_region = int(max_active_queries_per_region)
        self.max_inactive_queries_per_region = int(max_inactive_queries_per_region)
        self.active_target_mass = float(active_target_mass)
        self.inactive_ceiling = float(inactive_ceiling)
        self.inactive_weight = float(inactive_weight)
        self.token_masks: Optional[Mapping[str, torch.Tensor | Sequence[int] | np.ndarray]] = None
        self.handles: List[Any] = []
        self._losses: List[torch.Tensor] = []
        self._rows: List[Dict[str, float | int | str]] = []
        self._query_cache: Dict[Tuple[int, str], Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = {}

    def set_token_masks(self, token_masks: Mapping[str, torch.Tensor | Sequence[int] | np.ndarray]) -> None:
        self.token_masks = token_masks

    def clear(self) -> None:
        self._losses = []
        self._rows = []

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def attach(self, model: torch.nn.Module, *, layers: str | Sequence[int] | None = "last4") -> "ObjectTemporalCrossAttentionRecorder":
        self.remove()
        blocks = getattr(model, "blocks", None)
        if blocks is None:
            raise ValueError("model does not expose a Wan-style .blocks ModuleList")
        layer_indices = parse_attention_layer_indices(layers, len(blocks))
        if not layer_indices:
            raise ValueError(f"no valid attention layers selected from spec {layers!r}")
        for layer_idx in layer_indices:
            module = getattr(blocks[int(layer_idx)], "cross_attn", None)
            if module is None:
                continue
            handle = module.register_forward_hook(self._make_hook(int(layer_idx)))
            self.handles.append(handle)
        if not self.handles:
            raise ValueError("no cross-attention modules were found for selected layers")
        return self

    def has_loss(self) -> bool:
        return bool(self._losses)

    def loss(self) -> Optional[torch.Tensor]:
        if not self._losses:
            return None
        return torch.stack(self._losses).mean()

    def breakdown(self) -> List[Dict[str, float | int | str]]:
        return list(self._rows)

    def _query_sets(self, query_len: int, device: torch.device) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        cache_key = (int(query_len), str(device))
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        masks = _active_region_query_masks(self.budget, batch=1, query_len=int(query_len), device=device)
        query_sets: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for name, mask in masks.items():
            active = mask[0].nonzero(as_tuple=False).flatten()
            inactive = (~mask[0]).nonzero(as_tuple=False).flatten()
            active = _select_evenly_spaced(active, self.max_active_queries_per_region)
            inactive = _select_evenly_spaced(inactive, self.max_inactive_queries_per_region)
            if active.numel() > 0:
                query_sets[name] = (active, inactive)
        self._query_cache[cache_key] = query_sets
        return query_sets

    def _make_hook(self, layer_idx: int):
        def hook(module: torch.nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if self.token_masks is None or not torch.is_grad_enabled():
                return
            if len(inputs) < 2:
                return
            x = inputs[0]
            context = inputs[1]
            context_lens = inputs[2] if len(inputs) > 2 else None
            dtype = inputs[3] if len(inputs) > 3 and isinstance(inputs[3], torch.dtype) else x.dtype
            if x.ndim != 3 or context.ndim != 3:
                return
            batch = int(x.shape[0])
            heads = int(getattr(module, "num_heads"))
            head_dim = int(getattr(module, "head_dim"))
            q = module.norm_q(module.q(x.to(dtype))).view(batch, -1, heads, head_dim)
            k = module.norm_k(module.k(context.to(dtype))).view(batch, -1, heads, head_dim)
            query_len = int(q.shape[1])
            key_len = int(k.shape[1])
            valid_key_mask = None
            if context_lens is not None:
                lens = torch.as_tensor(context_lens, device=x.device, dtype=torch.long).flatten()
                if lens.numel() == 1 and batch > 1:
                    lens = lens.expand(batch)
                if lens.numel() == batch:
                    valid_key_mask = torch.arange(key_len, device=x.device)[None] < lens[:, None].clamp(0, key_len)

            scale = float(head_dim) ** -0.5
            query_sets = self._query_sets(query_len, x.device)
            for region_name, (active_indices, inactive_indices) in query_sets.items():
                if region_name not in self.token_masks:
                    continue
                prompt_mask = _token_mask_tensor(
                    self.token_masks[region_name],
                    batch=batch,
                    key_len=key_len,
                    device=x.device,
                )
                if not bool(prompt_mask.any().item()) or active_indices.numel() == 0:
                    continue

                def token_mass_for(indices: torch.Tensor) -> Optional[torch.Tensor]:
                    if indices.numel() == 0:
                        return None
                    selected_q = q.index_select(1, indices).float()
                    logits = torch.einsum("bqhd,bkhd->bhqk", selected_q, k.float()) * scale
                    if valid_key_mask is not None:
                        logits = logits.masked_fill(~valid_key_mask[:, None, None, :], -1e4)
                    probs = logits.softmax(dim=-1)
                    mass = (probs * prompt_mask[:, None, None, :].to(dtype=probs.dtype)).sum(dim=-1).mean(dim=1)
                    return mass

                active_mass = token_mass_for(active_indices)
                if active_mass is None:
                    continue
                active_target = torch.as_tensor(self.active_target_mass, device=x.device, dtype=active_mass.dtype)
                active_loss = F.relu(active_target - active_mass).square().mean()
                region_loss = active_loss

                inactive_mass = token_mass_for(inactive_indices)
                inactive_loss = active_loss * 0.0
                if inactive_mass is not None and self.inactive_weight > 0.0:
                    ceiling = torch.as_tensor(self.inactive_ceiling, device=x.device, dtype=inactive_mass.dtype)
                    inactive_loss = F.relu(inactive_mass - ceiling).square().mean()
                    region_loss = region_loss + inactive_loss * self.inactive_weight
                self._losses.append(region_loss)
                self._rows.append(
                    {
                        "layer": int(layer_idx),
                        "region": region_name,
                        "active_queries": int(active_indices.numel()),
                        "inactive_queries": int(inactive_indices.numel()),
                        "prompt_tokens": int(prompt_mask[0].sum().item()) if prompt_mask.numel() else 0,
                        "active_mass_mean": float(active_mass.detach().mean().item()),
                        "inactive_mass_mean": float(inactive_mass.detach().mean().item()) if inactive_mass is not None else 0.0,
                        "active_loss": float(active_loss.detach().item()),
                        "inactive_loss": float(inactive_loss.detach().item()),
                    }
                )

        return hook


def load_training_budget(path: Optional[str | Path]) -> Optional[TrainingBudgetTargets]:
    if path is None or str(path) == "":
        return None
    return TrainingBudgetTargets.from_npz(path)
