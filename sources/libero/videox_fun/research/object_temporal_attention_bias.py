"""Training-time object-token temporal cross-attention bias for Wan.

This helper intentionally does not add a new loss. It patches Wan cross-
attention modules so ordinary diffusion training keeps the same MSE objective,
while selected object prompt tokens receive a logit bias during the configured
frame window for each training sample.
"""

from __future__ import annotations

import math
import os
import types
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from videox_fun.research.region_temporal_training import parse_attention_layer_indices


def _find_subsequence_positions(sequence: Sequence[int], needle: Sequence[int]) -> List[int]:
    if not needle or len(needle) > len(sequence):
        return []
    positions: List[int] = []
    n = len(needle)
    for idx in range(0, len(sequence) - n + 1):
        if list(sequence[idx : idx + n]) == list(needle):
            positions.extend(range(idx, idx + n))
    return positions


def _encode_phrase(tokenizer: Any, phrase: str) -> List[int]:
    encoded = tokenizer(
        phrase,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    special_ids = {
        int(x)
        for x in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
        )
        if x is not None
    }
    return [int(x) for x in token_ids if int(x) not in special_ids]


class ObjectTemporalAttentionBiasController:
    """Patch Wan cross-attention with per-batch object/time logit bias."""

    def __init__(
        self,
        transformer: torch.nn.Module,
        tokenizer: Any,
        *,
        layers: str | Sequence[int] = "all",
        source_num_frames: int = 81,
        default_active_logit_bias: Optional[float] = None,
        default_inactive_logit_bias: Optional[float] = None,
        require_token_match: bool = True,
    ) -> None:
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.layers = layers
        self.source_num_frames = max(1, int(source_num_frames))
        self.default_active_logit_bias = default_active_logit_bias
        self.default_inactive_logit_bias = default_inactive_logit_bias
        self.require_token_match = bool(require_token_match)
        self.attention_chunk_size = max(1, int(os.environ.get("OBJECT_TEMPORAL_BIAS_ATTENTION_CHUNK_SIZE", "1024")))
        self._original_forwards: List[Any] = []
        self._hooks: List[Any] = []
        self._batch_specs: List[List[Dict[str, Any]]] = []

    def attach(self) -> "ObjectTemporalAttentionBiasController":
        if self._original_forwards:
            return self
        blocks = getattr(self.transformer, "blocks", None)
        if blocks is None:
            raise ValueError("transformer does not expose Wan-style .blocks")
        layer_indices = parse_attention_layer_indices(self.layers, len(blocks))
        if not layer_indices:
            raise ValueError(f"no valid object-temporal attention layers selected: {self.layers!r}")
        for block_idx in layer_indices:
            block = blocks[int(block_idx)]

            def block_pre_hook(mod, hook_args, hook_kwargs, idx=int(block_idx)):
                self._prepare_cross_attention(mod.cross_attn, hook_args, hook_kwargs, idx)

            self._hooks.append(block.register_forward_pre_hook(block_pre_hook, with_kwargs=True))
            module = block.cross_attn
            original_forward = module.forward

            def patched_forward(mod, x, context, context_lens, dtype=torch.bfloat16, t=0):
                return self._forward(mod, x, context, context_lens, dtype=dtype, t=t)

            module.forward = types.MethodType(patched_forward, module)
            self._original_forwards.append((module, original_forward))
        return self

    def remove(self) -> None:
        for module, original_forward in self._original_forwards:
            module.forward = original_forward
        self._original_forwards = []
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def set_batch(
        self,
        object_temporal_batch: Sequence[Sequence[Mapping[str, Any]]],
        input_ids: torch.Tensor,
        *,
        source_num_frames: Optional[int] = None,
    ) -> None:
        if input_ids.ndim == 1:
            input_ids = input_ids[None]
        input_ids_cpu = input_ids.detach().to(device="cpu", dtype=torch.long)
        source_frames = max(1, int(source_num_frames or self.source_num_frames))
        batch_specs: List[List[Dict[str, Any]]] = []
        any_match = False
        for batch_idx in range(int(input_ids_cpu.shape[0])):
            entries = object_temporal_batch[batch_idx] if batch_idx < len(object_temporal_batch) else []
            sequence = [int(x) for x in input_ids_cpu[batch_idx].tolist()]
            sample_specs: List[Dict[str, Any]] = []
            for entry in entries or []:
                object_name = str(entry.get("object", "")).strip()
                if not object_name:
                    continue
                token_positions: List[int] = []
                for phrase in [object_name, object_name.replace("_", " ")]:
                    token_ids = _encode_phrase(self.tokenizer, phrase)
                    token_positions.extend(_find_subsequence_positions(sequence, token_ids))
                token_positions = sorted(set(token_positions))
                if not token_positions:
                    continue
                any_match = True
                frame_range = entry.get("window_relative_frame_range") or entry.get("segment_frame_range") or [0, source_frames - 1]
                start_frame = max(0, int(frame_range[0]))
                end_frame = min(source_frames - 1, int(frame_range[1]))
                if end_frame < start_frame:
                    start_frame, end_frame = end_frame, start_frame
                active = self.default_active_logit_bias
                if active is None:
                    active = float(entry.get("active_logit_bias", math.log(4.0)))
                inactive = self.default_inactive_logit_bias
                if inactive is None:
                    inactive = float(entry.get("inactive_logit_bias", 0.0))
                sample_specs.append(
                    {
                        "object": object_name,
                        "token_positions": token_positions,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "source_num_frames": source_frames,
                        "active_logit_bias": float(active),
                        "inactive_logit_bias": float(inactive),
                    }
                )
            batch_specs.append(sample_specs)
        if self.require_token_match and object_temporal_batch and not any_match:
            raise ValueError(
                "object-temporal attention bias is enabled, but no object phrase matched prompt tokens in this batch"
            )
        self._batch_specs = batch_specs

    def _prepare_cross_attention(self, module: torch.nn.Module, hook_args: Sequence[Any], hook_kwargs: Mapping[str, Any], block_idx: int) -> None:
        x = hook_args[0] if hook_args else hook_kwargs.get("x")
        grid_sizes = hook_kwargs.get("grid_sizes")
        if x is None or grid_sizes is None:
            return
        branch = 1 if x.size(0) > 1 else 0
        branch = min(branch, grid_sizes.size(0) - 1)
        grid = tuple(int(v) for v in grid_sizes[branch].detach().cpu().tolist())
        module._object_temporal_bias_grid = grid
        module._object_temporal_bias_valid_len = int(math.prod(grid))
        module._object_temporal_bias_block = int(block_idx)

    @staticmethod
    def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
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

    def _build_mask(
        self,
        module: torch.nn.Module,
        q: torch.Tensor,
        context_len: int,
        *,
        query_offset: int = 0,
    ) -> Optional[torch.Tensor]:
        if not self._batch_specs:
            return None
        grid = getattr(module, "_object_temporal_bias_grid", None)
        if grid is None:
            return None
        batch, seq_len = int(q.size(0)), int(q.size(1))
        valid_total = int(getattr(module, "_object_temporal_bias_valid_len", seq_len))
        valid_len = int(max(0, min(valid_total - int(query_offset), seq_len)))
        if valid_len <= 0:
            return None
        frames, height, width = [int(x) for x in grid]
        query_indices = torch.arange(
            int(query_offset),
            int(query_offset) + valid_len,
            device=q.device,
            dtype=torch.long,
        )
        query_frames = torch.div(query_indices, max(height * width, 1), rounding_mode="floor").clamp(0, max(frames - 1, 0))
        mask = torch.zeros((batch, 1, seq_len, int(context_len)), device=q.device, dtype=q.dtype)
        wrote = False
        for batch_idx in range(batch):
            specs = self._batch_specs[batch_idx] if batch_idx < len(self._batch_specs) else []
            for spec in specs:
                source_frames = max(1, int(spec["source_num_frames"]))
                if frames <= 1:
                    source_query_frames = torch.zeros_like(query_frames)
                else:
                    source_query_frames = torch.round(query_frames.float() * float(source_frames - 1) / float(frames - 1)).long()
                active = (source_query_frames >= int(spec["start_frame"])) & (source_query_frames <= int(spec["end_frame"]))
                values = torch.where(
                    active,
                    torch.full_like(source_query_frames, float(spec["active_logit_bias"]), dtype=q.dtype),
                    torch.full_like(source_query_frames, float(spec["inactive_logit_bias"]), dtype=q.dtype),
                )
                local_positions = [int(pos) for pos in spec["token_positions"] if 0 <= int(pos) < int(context_len)]
                if not local_positions:
                    continue
                idx = torch.as_tensor(local_positions, device=q.device, dtype=torch.long)
                mask[batch_idx, 0, :valid_len, idx] += values[:, None]
                wrote = True
        return mask if wrote else None

    def _branch_forward(self, module: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, dtype: torch.dtype, *, apply_bias: bool) -> torch.Tensor:
        if not apply_bias or not any(self._batch_specs):
            return self._sdpa(q, k, v, attn_mask=None).to(dtype)
        return self._sdpa_chunked_with_bias(module, q, k, v, dtype=dtype)

    def _sdpa_chunked_with_bias(
        self,
        module: torch.nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        chunk_size = max(1, int(self.attention_chunk_size))
        outputs: List[torch.Tensor] = []
        scale = 1.0 / math.sqrt(max(1, int(q.size(-1))))
        k_heads = k.permute(0, 2, 3, 1).contiguous()
        v_heads = v.permute(0, 2, 1, 3).contiguous()
        for start in range(0, int(q.size(1)), chunk_size):
            end = min(start + chunk_size, int(q.size(1)))
            q_chunk = q[:, start:end]
            attn_mask = self._build_mask(module, q_chunk, k.size(1), query_offset=start)
            if attn_mask is None:
                outputs.append(self._sdpa(q_chunk, k, v, attn_mask=None).to(dtype))
                continue
            q_heads = q_chunk.permute(0, 2, 1, 3).contiguous()
            scores = torch.matmul(q_heads, k_heads) * scale
            scores = scores + attn_mask
            probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
            out = torch.matmul(probs, v_heads).permute(0, 2, 1, 3).contiguous()
            outputs.append(out.to(dtype))
            del q_heads, scores, probs, out, attn_mask
        return torch.cat(outputs, dim=1)

    def _forward(self, module: torch.nn.Module, x: torch.Tensor, context: torch.Tensor, context_lens: Any, dtype=torch.bfloat16, t=0):
        batch, seq_len, heads, head_dim = x.size(0), x.size(1), module.num_heads, module.head_dim
        q = module.norm_q(module.q(x.to(dtype))).view(batch, seq_len, heads, head_dim)

        if hasattr(module, "k_img") and hasattr(module, "v_img") and context.size(1) > 257:
            context_img = context[:, :257]
            context_text = context[:, 257:]
            k_img = module.norm_k_img(module.k_img(context_img.to(dtype))).view(batch, -1, heads, head_dim)
            v_img = module.v_img(context_img.to(dtype)).view(batch, -1, heads, head_dim)
            img_x = self._branch_forward(module, q, k_img, v_img, dtype, apply_bias=False)
            if context_text.size(1) > 0:
                k_text = module.norm_k(module.k(context_text.to(dtype))).view(batch, -1, heads, head_dim)
                v_text = module.v(context_text.to(dtype)).view(batch, -1, heads, head_dim)
                text_x = self._branch_forward(module, q, k_text, v_text, dtype, apply_bias=True)
            else:
                text_x = torch.zeros_like(img_x)
            return module.o((img_x + text_x).flatten(2))

        k = module.norm_k(module.k(context.to(dtype))).view(batch, -1, heads, head_dim)
        v = module.v(context.to(dtype)).view(batch, -1, heads, head_dim)
        out = self._branch_forward(module, q, k, v, dtype, apply_bias=True)
        return module.o(out.flatten(2))
