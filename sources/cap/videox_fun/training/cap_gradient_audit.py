import math

import torch


def local_shard_max_abs(
    grad,
    *,
    full_shape,
    shard_offset,
    channel_slice=None,
):
    """Return the selected max gradient from one flattened FSDP parameter shard."""
    full_shape = tuple(int(value) for value in full_shape)
    full_numel = math.prod(full_shape)
    shard_offset = int(shard_offset)
    if grad is None:
        return None

    flat_grad = grad.detach().reshape(-1)
    shard_end = shard_offset + flat_grad.numel()
    if shard_offset < 0 or shard_end > full_numel:
        raise ValueError(
            f"FSDP shard [{shard_offset}, {shard_end}) is outside parameter "
            f"with shape {full_shape} and numel {full_numel}."
        )
    if flat_grad.numel() == 0:
        return flat_grad.new_zeros((), dtype=torch.float32)
    if channel_slice is None:
        return flat_grad.abs().amax().to(dtype=torch.float32)
    if len(full_shape) < 2:
        raise ValueError("Channel-selective gradient audit requires a rank >= 2 parameter.")

    start, stop, step = channel_slice.indices(full_shape[1])
    if step != 1:
        raise ValueError("Channel-selective gradient audit requires a contiguous slice.")
    inner_numel = math.prod(full_shape[2:])
    global_indices = torch.arange(
        shard_offset,
        shard_end,
        device=flat_grad.device,
        dtype=torch.int64,
    )
    channel_indices = torch.div(
        global_indices,
        inner_numel,
        rounding_mode="floor",
    ).remainder(full_shape[1])
    selected = flat_grad[(channel_indices >= start) & (channel_indices < stop)]
    if selected.numel() == 0:
        return flat_grad.new_zeros((), dtype=torch.float32)
    return selected.abs().amax().to(dtype=torch.float32)
