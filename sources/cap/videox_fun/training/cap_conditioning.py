import torch


def _per_sample_gate(mask, batch_size, *, device, dtype, name):
    if mask is None:
        return torch.ones((batch_size, 1, 1, 1, 1), device=device, dtype=dtype)
    values = torch.as_tensor(mask, device=device)
    if values.numel() == 1:
        values = values.reshape(1).expand(batch_size)
    elif values.ndim == 0 or values.shape[0] != batch_size:
        raise ValueError(f"{name} must be scalar or have batch dimension {batch_size}")
    else:
        values = values.reshape(batch_size, -1)
        if values.shape[1] > 1 and not bool((values == values[:, :1]).all()):
            raise ValueError(f"{name} is inconsistent within a sample")
        values = values[:, 0]
    if values.is_floating_point() and not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} contains non-finite values")
    if not bool(((values == 0) | (values == 1)).all()):
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return values.to(dtype=dtype).view(batch_size, 1, 1, 1, 1)


def expand_patch_embedding_weight(source_weight, target_weight):
    """Expand only the input-channel axis and zero-initialize every new channel."""
    if source_weight.ndim != 5 or target_weight.ndim != 5:
        raise ValueError("patch_embedding weights must be rank-5 Conv3d tensors")
    if source_weight.shape[0] != target_weight.shape[0] or source_weight.shape[2:] != target_weight.shape[2:]:
        raise ValueError(
            f"patch_embedding non-channel shape mismatch: source={tuple(source_weight.shape)} "
            f"target={tuple(target_weight.shape)}"
        )
    if source_weight.shape[1] > target_weight.shape[1]:
        raise ValueError(
            f"refusing to truncate patch_embedding channels: "
            f"source={source_weight.shape[1]} target={target_weight.shape[1]}"
        )
    if source_weight.shape == target_weight.shape:
        return source_weight
    expanded = torch.zeros(
        target_weight.shape,
        device="cpu",
        dtype=target_weight.dtype,
    )
    expanded[:, : source_weight.shape[1]] = source_weight.to(
        device=expanded.device, dtype=expanded.dtype
    )
    return expanded


def build_action_map_control_latents(
    control_latents,
    action_map_latents,
    *,
    latent_channels,
    action_map_mask=None,
):
    if control_latents.ndim != 5 or action_map_latents.ndim != 5:
        raise ValueError("control and action-map latents must be rank-5 [B,C,F,H,W]")
    if control_latents.shape[0] != action_map_latents.shape[0]:
        raise ValueError("control and action-map batch sizes differ")
    if control_latents.shape[-2:] != action_map_latents.shape[-2:]:
        raise ValueError("control and action-map spatial grids differ")
    if action_map_latents.shape[1] != latent_channels:
        raise ValueError(
            f"action-map channels must equal latent_channels={latent_channels}, "
            f"got {action_map_latents.shape[1]}"
        )
    if control_latents.shape[1] < latent_channels:
        raise ValueError("control tensor has fewer channels than the video latent")

    null_control = control_latents
    conditioned = control_latents.clone()
    latent_start = conditioned.shape[1] - latent_channels
    base_video_control = control_latents[:, latent_start:].clone()
    action_video_control = base_video_control.clone()
    frames_to_use = min(conditioned.shape[2], action_map_latents.shape[2])
    if frames_to_use > 1:
        action_video_control[:, :, 1:frames_to_use] = action_map_latents[:, :, 1:frames_to_use]
    gate = _per_sample_gate(
        action_map_mask,
        conditioned.shape[0],
        device=conditioned.device,
        dtype=conditioned.dtype,
        name="action_map_mask",
    )
    conditioned[:, latent_start:] = (
        gate * action_video_control + (1.0 - gate) * base_video_control
    )
    return conditioned, null_control


def pack_camera_condition(camera_values):
    """Pack [B,F,6,H,W] pixel rays into [B,24,(F+3)/4,H,W]."""
    if camera_values.ndim != 5:
        raise ValueError("camera_values must have shape [B,F,6,H,W]")
    batch, frames, channels, height, width = camera_values.shape
    if channels != 6:
        raise ValueError(f"camera_values must have 6 Pluecker channels, got {channels}")
    if frames < 1 or (frames - 1) % 4 != 0:
        raise ValueError(f"camera frame count must have form 4n+1, got {frames}")
    camera = camera_values.transpose(1, 2).contiguous()
    camera = torch.cat(
        [camera[:, :, :1].repeat_interleave(4, dim=2), camera[:, :, 1:]],
        dim=2,
    )
    packed_frames = camera.shape[2] // 4
    camera = camera.transpose(1, 2).contiguous()
    camera = camera.view(batch, packed_frames, 4, channels, height, width)
    camera = camera.transpose(2, 3).contiguous()
    camera = camera.view(batch, packed_frames, channels * 4, height, width)
    return camera.transpose(1, 2).contiguous()


def build_poseanything_condition_latents(
    video_latents,
    skeleton_latents,
    black_skeleton_latents,
    *,
    skeleton_mask=None,
):
    if video_latents.shape != skeleton_latents.shape:
        raise ValueError(
            f"skeleton latent shape mismatch: video={tuple(video_latents.shape)} "
            f"skeleton={tuple(skeleton_latents.shape)}"
        )
    if black_skeleton_latents.shape != video_latents.shape:
        raise ValueError(
            f"black-skeleton latent shape mismatch: video={tuple(video_latents.shape)} "
            f"black={tuple(black_skeleton_latents.shape)}"
        )
    skeleton = skeleton_latents.to(device=video_latents.device, dtype=video_latents.dtype)
    null_skeleton = black_skeleton_latents.to(
        device=video_latents.device, dtype=video_latents.dtype
    )
    gate = _per_sample_gate(
        skeleton_mask,
        video_latents.shape[0],
        device=video_latents.device,
        dtype=video_latents.dtype,
        name="skeleton_mask",
    )
    conditioned = gate * skeleton + (1.0 - gate) * null_skeleton
    return conditioned, null_skeleton
