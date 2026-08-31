"""Optional pixel-domain Arm Method1 MSE heatmaps for inference.

The base error compares decoded generated/GT RGB pixels.  Weighted modes use
the training implementation's latent-token Arm weights as ``rho`` and apply
them at full video resolution.  This module keeps those formulas and the
visualization code free of the training entrypoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


WEIGHT_MODES = ("MSE", "CAER")
WEIGHTED_MODES = WEIGHT_MODES
EFFECT_MODES = ("CAER",)
OUTPUT_NAMES = {
    "MSE": "MSE",
    "CAER": "CAER",
}


def parse_weight_selection(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse ``none``, ``all``, or a comma-separated subset.

    The return value follows the canonical training order so manifests and
    output directories are stable even when a user changes list ordering.
    """

    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.strip().upper()
        if not raw or raw == "none":
            return ()
        raw_parts = raw.split(",")
        if any(not part.strip() for part in raw_parts):
            raise ValueError("MSE heatmap selection contains an empty mode")
        parts = [part.strip().upper() for part in raw_parts]
    else:
        parts = [str(part).strip().upper() for part in value]
        if any(not part for part in parts):
            raise ValueError("MSE heatmap selection contains an empty mode")
    if not parts or parts == ["none"]:
        return ()
    if "none" in parts:
        raise ValueError("MSE heatmap selection 'none' cannot be combined with other modes")
    if "all" in parts:
        if parts != ["all"]:
            raise ValueError("MSE heatmap selection 'all' cannot be combined with other modes")
        return WEIGHT_MODES
    unknown = sorted(set(parts).difference(WEIGHT_MODES))
    if unknown:
        raise ValueError(
            "unknown MSE heatmap mode(s): "
            + ", ".join(unknown)
            + "; choose none, all, or a subset of "
            + ", ".join(WEIGHT_MODES)
        )
    duplicates = sorted({part for part in parts if parts.count(part) > 1})
    if duplicates:
        raise ValueError("duplicate MSE heatmap mode(s): " + ", ".join(duplicates))
    selected = set(parts)
    return tuple(mode for mode in WEIGHT_MODES if mode in selected)


def format_weight_selection(modes: Iterable[str]) -> str:
    selected = parse_weight_selection(modes)
    return "none" if not selected else ",".join(selected)


def output_name(mode: str) -> str:
    if mode not in OUTPUT_NAMES:
        raise ValueError(f"unknown MSE heatmap mode: {mode!r}")
    return OUTPUT_NAMES[mode]


def _validate_prediction_shapes(prediction, target):
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "prediction and target must share shape [B,C,T,H,W]; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )


def _to_unit_interval(video):
    """Return a floating video in the RGB ``[0, 1]`` convention.

    The inference pipeline currently returns decoded frames in ``[0, 1]``;
    accepting the common ``[-1, 1]`` and ``[0, 255]`` forms keeps this helper
    safe for callers that obtain frames from a different decoder.
    """

    import torch

    values = video.float()
    minimum = float(values.detach().amin().item())
    maximum = float(values.detach().amax().item())
    if minimum >= -1.0001 and maximum <= 1.0001:
        if minimum < -1e-4:
            values = (values + 1.0) / 2.0
    elif maximum <= 255.0 + 1e-3 and minimum >= -1e-3:
        values = values / 255.0
    else:
        raise ValueError(
            "RGB video values must be in [0,1], [-1,1], or [0,255]; "
            f"got range [{minimum}, {maximum}]"
        )
    return values.clamp(0.0, 1.0)


def compute_pixel_mse_map(prediction, target):
    """Compute true RGB pixel MSE as ``[B, 1, T, H, W]``.

    This is intentionally separate from :func:`compute_weight_maps`: the
    latter operates on latent flow tensors for the Method1 training signal,
    while this function compares decoded RGB frames at the requested output
    resolution.
    """

    import torch

    if not torch.is_tensor(prediction) or not torch.is_tensor(target):
        raise ValueError("prediction and target must be torch tensors")
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError(
            "prediction and target must share RGB video shape [B,3,T,H,W]; "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.shape[1] != 3:
        raise ValueError(
            "prediction and target must have three RGB channels; "
            f"got {prediction.shape[1]}"
        )
    prediction_f = _to_unit_interval(prediction)
    target_f = _to_unit_interval(target)
    return (prediction_f - target_f).square().mean(dim=1, keepdim=True)


def _future_mean(values, exclude_first_frame: bool):
    future = values[:, :, 1:] if exclude_first_frame and values.size(2) > 1 else values
    return future.mean(dim=(1, 2, 3, 4), keepdim=True)


def _normalize_token_map(values, eps: float, exclude_first_frame: bool):
    import torch

    mean = _future_mean(values, exclude_first_frame)
    normalized = values / mean.clamp_min(float(eps))
    return torch.where(mean > float(eps), normalized, torch.ones_like(normalized))


def compute_rho_maps(
    prediction,
    target,
    effect_map,
    modes: Iterable[str],
    *,
    eps: float = 1e-6,
    exclude_first_frame: bool = True,
):
    """Return Method1 ``rho`` maps on the future latent-token domain.

    ``rho`` follows ``videox_fun.training.method1_focused_loss`` exactly.  The
    returned maps have shape ``[B, 1, T-1, H, W]`` when the first frame is
    excluded, matching the loss's future-token domain.  ``MSE`` uses unit
    weights and ``CAER`` uses detached action-effect weights.
    """

    import torch

    selected = parse_weight_selection(modes)
    if not selected:
        return {}
    _validate_prediction_shapes(prediction, target)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    prediction_f = prediction.float()
    target_f = target.float()
    residual = prediction_f - target_f
    residual_map = torch.linalg.vector_norm(residual, ord=2, dim=1, keepdim=True)

    effect_f = None
    if any(mode in EFFECT_MODES for mode in selected):
        if effect_map is None:
            raise ValueError("effect_map is required for weighted MSE heatmaps")
        effect_f = effect_map.detach().float()
        expected = (prediction.shape[0], 1, *prediction.shape[2:])
        if tuple(effect_f.shape) != expected:
            raise ValueError(f"effect_map must have shape {expected}; got {tuple(effect_f.shape)}")
        if not bool(torch.isfinite(effect_f).all().item()):
            raise ValueError("effect_map contains non-finite values")

    rho_by_mode = {}
    for mode in selected:
        if mode == "MSE":
            rho_by_mode[mode] = torch.ones_like(residual_map)
        elif mode == "CAER":
            rho_by_mode[mode] = _normalize_token_map(effect_f, eps, exclude_first_frame).detach()

    drop_first = exclude_first_frame and prediction.shape[2] > 1
    if drop_first:
        return {mode: value[:, :, 1:] for mode, value in rho_by_mode.items()}
    return rho_by_mode


def compute_weight_maps(
    prediction,
    target,
    effect_map,
    modes: Iterable[str],
    *,
    eps: float = 1e-6,
    exclude_first_frame: bool = True,
):
    """Return latent-token ``MSE`` or ``rho*MSE`` maps.

    This remains the Method1 training-domain diagnostic.  Inference heatmap
    videos use :func:`compute_pixel_mse_map` instead, optionally multiplied by
    the ``rho`` maps returned by :func:`compute_rho_maps`.
    """

    import torch

    selected = parse_weight_selection(modes)
    if not selected:
        return {}
    _validate_prediction_shapes(prediction, target)
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    residual = prediction.float() - target.float()
    mse = residual.pow(2).mean(dim=1, keepdim=True)
    drop_first = exclude_first_frame and mse.size(2) > 1
    if drop_first:
        mse = mse[:, :, 1:]
    rho_by_mode = compute_rho_maps(
        prediction,
        target,
        effect_map,
        selected,
        eps=eps,
        exclude_first_frame=exclude_first_frame,
    )
    return {
        mode: mse if mode == "MSE" else mse * rho_by_mode[mode]
        for mode in selected
    }


def render_map_to_video(
    values,
    frame_count: int,
    *,
    scale: int = 8,
    output_size: tuple[int, int] | None = None,
):
    """Upsample a future-token map to video pixels.

    By default the historical ``latent_size * scale`` output is retained for
    standalone diagnostics.  Inference passes ``output_size=(height,width)``
    so a latent ``rho`` map can be applied to a full-resolution RGB MSE map.
    """

    import torch
    import torch.nn.functional as F

    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if output_size is not None:
        if len(output_size) != 2 or any(int(value) <= 0 for value in output_size):
            raise ValueError(f"output_size must be (positive height, width), got {output_size}")
        height, width = (int(output_size[0]), int(output_size[1]))
    tensor = values.detach().float() if torch.is_tensor(values) else torch.as_tensor(values).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 5 or tensor.shape[0] < 1 or tensor.shape[1] < 1:
        raise ValueError(f"map must have shape [B,1,T,H,W], got {tuple(tensor.shape)}")
    tensor = tensor[:1, :1]
    if output_size is None:
        height = max(int(tensor.shape[-2]) * int(scale), 1)
        width = max(int(tensor.shape[-1]) * int(scale), 1)
    if tensor.shape[2] == 0:
        return np.zeros(
            (int(frame_count), height, width),
            dtype=np.float32,
        )
    # The loss excludes the fixed first frame. Render exactly one zero output
    # frame, then map the future latent tokens to the remaining video frames;
    # interpolating a prepended zero would create an extra artificial zero band.
    future_count = max(int(frame_count) - 1, 1)
    future = F.interpolate(
        tensor,
        size=(future_count, height, width),
        mode="trilinear",
        align_corners=False,
    )
    first = torch.zeros_like(future[:, :, :1])
    rendered = torch.cat([first, future], dim=2)[:, :, : int(frame_count)]
    return rendered[0, 0].cpu().numpy().astype(np.float32)


def colorize(values: np.ndarray, vmax: float | None = None) -> tuple[np.ndarray, float]:
    """Colorize nonnegative maps with a compact perceptual blue-yellow-red scale."""

    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    safe = np.where(finite, np.maximum(array, 0.0), 0.0)
    if vmax is None:
        positive = safe[safe > 0]
        vmax = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    vmax = max(float(vmax), 1e-8)
    stops = np.asarray(
        [[49, 54, 149], [69, 117, 180], [116, 173, 209], [253, 174, 97], [215, 48, 39]],
        dtype=np.float32,
    )
    scaled = np.clip(safe / vmax, 0.0, 1.0) * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int64)
    high = np.clip(low + 1, 0, len(stops) - 1)
    frac = scaled - low
    rgb = stops[low] * (1.0 - frac[..., None]) + stops[high] * frac[..., None]
    return np.uint8(np.clip(rgb, 0, 255)), vmax


def _estimate_vmax(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    positive = finite[finite > 0]
    return max(float(np.percentile(positive, 99.0)) if positive.size else 1.0, 1e-8)


def save_heatmap_video(values: np.ndarray, path: Path, fps: int) -> Path:
    """Save a colorized heatmap sequence, falling back to GIF if ffmpeg is absent.

    Frames are colorized one at a time so full-resolution pixel maps do not
    require a multi-gigabyte intermediate RGB tensor.
    """

    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"heatmap values must have shape [frames,height,width], got {array.shape}")
    vmax = _estimate_vmax(array)
    pad_h = array.shape[1] % 2
    pad_w = array.shape[2] % 2

    def iter_frames():
        for frame in array:
            colored, _ = colorize(frame, vmax=vmax)
            if pad_h or pad_w:
                # H.264 requires even dimensions even when macroblock padding is off.
                colored = np.pad(colored, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            yield colored

    try:
        if path.suffix.lower() == ".mp4":
            writer = imageio.get_writer(path, fps=int(fps), macro_block_size=1)
            try:
                for frame in iter_frames():
                    writer.append_data(frame)
            finally:
                writer.close()
        else:
            imageio.mimsave(path, list(iter_frames()), fps=int(fps))
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"video writer produced an empty file: {path}")
        return path
    except Exception:
        if path.is_file():
            path.unlink()
        fallback = path.with_suffix(".gif")
        imageio.mimsave(
            fallback,
            list(iter_frames()),
            duration=1.0 / max(int(fps), 1),
            loop=0,
        )
        return fallback


def save_heatmap_array(values: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=np.asarray(values, dtype=np.float32))
    return path


def heatmap_stats(values: np.ndarray) -> dict[str, float | list[int]]:
    array = np.asarray(values, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        finite = np.asarray([0.0], dtype=np.float32)
    return {
        "shape": list(array.shape),
        "min": float(finite.min()),
        "mean": float(finite.mean()),
        "max": float(finite.max()),
        "p99": float(np.percentile(finite, 99.0)),
    }


def positive_percentile_vmax(
    values: np.ndarray,
    *,
    percentile: float = 99.0,
    exclude_first_frame: bool = True,
) -> float:
    """Compute one positive latent-domain scale for a complete episode."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1:
        raise ValueError(f"weight values must include a time dimension, got {array.shape}")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0,100]")
    scale_source = array[1:] if exclude_first_frame and len(array) > 1 else array
    positive = scale_source[np.isfinite(scale_source) & (scale_source > 0)]
    return float(np.percentile(positive, percentile)) if positive.size else 1.0


def smooth_latent_spatially(
    values: np.ndarray,
    *,
    sigma: float = 1.5,
) -> np.ndarray:
    """Gaussian-smooth latent maps spatially without mixing time steps."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"latent values must have shape [time,height,width], got {array.shape}")
    if sigma < 0:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0:
        return array.copy()
    radius = max(int(np.ceil(3.0 * float(sigma))), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    kernel /= kernel.sum()

    width_padded = np.pad(array, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    horizontal = sum(
        float(weight) * width_padded[..., index : index + array.shape[-1]]
        for index, weight in enumerate(kernel)
    )
    height_padded = np.pad(horizontal, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    smoothed = sum(
        float(weight) * height_padded[:, index : index + array.shape[-2], :]
        for index, weight in enumerate(kernel)
    )
    return np.asarray(smoothed, dtype=np.float32)


def normalize_latent_product_chunks(
    left_chunks: Sequence[np.ndarray],
    right_chunks: Sequence[np.ndarray],
    *,
    eps: float = 1e-8,
) -> list[np.ndarray]:
    """Return ``left * right / mean(left * right)`` for each latent chunk."""

    if len(left_chunks) != len(right_chunks):
        raise ValueError("latent product inputs must have the same number of chunks")
    if eps <= 0:
        raise ValueError("eps must be positive")
    normalized = []
    for index, (left, right) in enumerate(zip(left_chunks, right_chunks)):
        left_array = np.asarray(left, dtype=np.float32)
        right_array = np.asarray(right, dtype=np.float32)
        if left_array.shape != right_array.shape or left_array.ndim != 3:
            raise ValueError(
                f"latent chunk {index} shapes must match [time,height,width]: "
                f"{left_array.shape} vs {right_array.shape}"
            )
        product = left_array * right_array
        if not np.isfinite(product).all():
            raise ValueError(f"latent product chunk {index} contains non-finite values")
        mean = float(product.mean())
        normalized.append(
            product / mean if mean > float(eps) else np.ones_like(product)
        )
    return [np.asarray(chunk, dtype=np.float32) for chunk in normalized]


def render_latent_chunks_to_video(
    latent_chunks: Sequence[np.ndarray],
    frame_count: int,
    *,
    output_size: tuple[int, int],
    spatial_sigma: float = 1.5,
    future_frames_per_chunk: int = 16,
) -> np.ndarray:
    """Smooth latent chunks and reproduce their chunk-wise video interpolation."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    remaining = frame_count - 1
    rendered_chunks = []
    for latent in latent_chunks:
        if remaining <= 0:
            break
        future_count = min(int(future_frames_per_chunk), remaining)
        smoothed = smooth_latent_spatially(latent, sigma=spatial_sigma)
        rendered = render_map_to_video(
            smoothed,
            int(future_frames_per_chunk) + 1,
            output_size=output_size,
        )
        rendered_chunks.append(rendered[1 : future_count + 1])
        remaining -= future_count
    if remaining != 0:
        raise ValueError(f"latent chunks are missing {remaining} future frames")
    first = np.zeros((1, int(output_size[0]), int(output_size[1])), dtype=np.float32)
    return np.concatenate([first, *rendered_chunks], axis=0)


def episode_response_vmin(values: np.ndarray, vmax: float) -> float:
    """Return the episode-wide minimum P99-scaled response after frame zero."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim < 1:
        raise ValueError(f"weight values must include a time dimension, got {array.shape}")
    scale_source = array[1:] if len(array) > 1 else array
    finite = scale_source[np.isfinite(scale_source)]
    if finite.size == 0:
        return 0.0
    return float(np.clip(finite.min() / max(float(vmax), 1e-8), 0.0, 1.0))


def normalize_weight_response(
    values: np.ndarray,
    vmax: float,
    *,
    vmin: float = 0.0,
) -> np.ndarray:
    """Map one episode-wide response interval to ``[0,1]``."""

    array = np.asarray(values, dtype=np.float32)
    finite = np.where(np.isfinite(array), array, 0.0)
    scaled = finite / max(float(vmax), 1e-8)
    response = np.clip(
        (scaled - float(vmin)) / max(1.0 - float(vmin), 1e-8),
        0.0,
        1.0,
    )
    return response.astype(np.float32, copy=False)


def _linear_interpolation_matrix(input_size: int, output_size: int) -> np.ndarray:
    """Return the 1D ``align_corners=False`` linear interpolation operator."""

    if input_size <= 0 or output_size <= 0:
        raise ValueError("interpolation sizes must be positive")
    matrix = np.zeros((output_size, input_size), dtype=np.float64)
    for output_index in range(output_size):
        source = max(
            (output_index + 0.5) * input_size / output_size - 0.5,
            0.0,
        )
        lower = min(int(np.floor(source)), input_size - 1)
        upper = min(lower + 1, input_size - 1)
        fraction = source - lower
        matrix[output_index, lower] += 1.0 - fraction
        matrix[output_index, upper] += fraction
    return matrix


def recover_latent_chunks_from_interpolated_weights(
    weights: np.ndarray,
    *,
    latent_shape: tuple[int, int, int] = (4, 44, 80),
    future_frames_per_chunk: int = 16,
) -> tuple[list[np.ndarray], float]:
    """Invert the visualization's separable trilinear interpolation on CPU."""

    values = np.asarray(weights, dtype=np.float32)
    if values.ndim != 3 or len(values) < 2:
        raise ValueError(f"weights must have shape [frames,height,width], got {values.shape}")
    latent_time, latent_height, latent_width = latent_shape
    output_height, output_width = values.shape[1:]
    height_matrix = _linear_interpolation_matrix(latent_height, output_height)
    width_matrix = _linear_interpolation_matrix(latent_width, output_width)
    time_matrix = _linear_interpolation_matrix(latent_time, future_frames_per_chunk)

    height_indices = np.floor(
        (np.arange(latent_height, dtype=np.float64) + 0.5)
        * output_height
        / latent_height
    ).astype(np.int64)
    width_indices = np.floor(
        (np.arange(latent_width, dtype=np.float64) + 0.5)
        * output_width
        / latent_width
    ).astype(np.int64)
    inverse_height = np.linalg.inv(height_matrix[height_indices])
    inverse_width = np.linalg.inv(width_matrix[width_indices])

    latent_chunks = []
    maximum_error = 0.0
    future = values[1:]
    rng = np.random.default_rng(42)
    for start in range(0, len(future), future_frames_per_chunk):
        chunk = future[start : start + future_frames_per_chunk]
        if len(chunk) < latent_time:
            raise ValueError(
                f"partial chunk has {len(chunk)} frames; need at least {latent_time}"
            )
        sampled = chunk[:, height_indices][:, :, width_indices].astype(np.float64)
        spatial = np.einsum(
            "ih,thw,jw->tij",
            inverse_height,
            sampled,
            inverse_width,
            optimize=True,
        )
        inverse_time = np.linalg.pinv(time_matrix[: len(chunk)])
        latent = np.einsum("kt,tij->kij", inverse_time, spatial, optimize=True)
        latent_chunks.append(latent.astype(np.float32))

        for _ in range(64):
            t = int(rng.integers(0, len(chunk)))
            h = int(rng.integers(0, output_height))
            w = int(rng.integers(0, output_width))
            reconstructed = np.einsum(
                "k,i,j,kij->",
                time_matrix[t],
                height_matrix[h],
                width_matrix[w],
                latent,
                optimize=True,
            )
            maximum_error = max(
                maximum_error,
                abs(float(reconstructed) - float(chunk[t, h, w])),
            )
    return latent_chunks, maximum_error


def normalized_sigmoid(response: np.ndarray, *, k: float = 18.0) -> np.ndarray:
    """Apply an endpoint-preserving sigmoid response curve on ``[0,1]``."""

    if k <= 0:
        raise ValueError("k must be positive")
    values = np.clip(np.asarray(response, dtype=np.float32), 0.0, 1.0)
    low = 1.0 / (1.0 + np.exp(float(k) / 2.0))
    high = 1.0 / (1.0 + np.exp(-float(k) / 2.0))
    curved = 1.0 / (1.0 + np.exp(-float(k) * (values - 0.5)))
    return np.clip((curved - low) / (high - low), 0.0, 1.0).astype(
        np.float32, copy=False
    )


def _reference_colorize(response: np.ndarray) -> np.ndarray:
    """Apply the sigmoid response curve and nonuniform reference palette."""

    values = normalized_sigmoid(response, k=12.0)
    levels = np.asarray(
        [0.0, 0.125, 0.375, 0.625, 0.875, 1.0], dtype=np.float32
    )
    stops = np.asarray(
        [
            [0, 0, 128],
            [0, 0, 255],
            [0, 255, 255],
            [255, 255, 0],
            [255, 0, 0],
            [128, 0, 0],
        ],
        dtype=np.float32,
    )
    channels = [
        np.interp(values, levels, stops[:, channel]) for channel in range(3)
    ]
    return np.uint8(np.clip(np.stack(channels, axis=-1), 0, 255))


def smooth_weight_response(
    response: np.ndarray,
    *,
    blur_radius: float = 12.0,
) -> np.ndarray:
    """Apply one light Gaussian blur without rescaling its response."""

    from PIL import Image, ImageFilter

    source = np.clip(np.asarray(response, dtype=np.float32), 0.0, 1.0)
    if source.ndim != 2:
        raise ValueError(f"response must have shape [height,width], got {source.shape}")
    if blur_radius < 0:
        raise ValueError("blur_radius must be nonnegative")
    image = Image.fromarray(np.uint8(np.round(source * 255.0)), mode="L")
    if blur_radius == 0:
        return source.copy()
    blurred = image.filter(ImageFilter.GaussianBlur(radius=float(blur_radius)))
    return np.clip(np.asarray(blurred, dtype=np.float32) / 255.0, 0.0, 1.0)


def overlay_weight_response(
    rgb_frame: np.ndarray,
    response: np.ndarray,
    *,
    blur_radius: float = 12.0,
) -> np.ndarray:
    """Overlay the reference heatmap with the fixed blend."""

    rgb = np.asarray(rgb_frame)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"rgb_frame must have shape [height,width,3], got {rgb.shape}")
    if tuple(rgb.shape[:2]) != tuple(np.asarray(response).shape):
        raise ValueError(
            f"RGB/response size mismatch: rgb={rgb.shape[:2]} response={np.asarray(response).shape}"
        )
    smooth = smooth_weight_response(response, blur_radius=blur_radius)
    heat = _reference_colorize(smooth)
    return np.uint8(np.clip(
        rgb.astype(np.float32) * 0.55 + heat.astype(np.float32) * 0.65,
        0,
        255,
    ))


def encode_video_to_latents(pipeline: Any, video, device):
    """Encode a normalized ``[B,C,T,H,W]`` video with the inference VAE."""

    import torch

    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("video must be a torch tensor with shape [B,C,T,H,W]")
    batch, channels, frames, height, width = video.shape
    if channels != 3:
        raise ValueError(f"video must have 3 channels, got {channels}")
    video = video.float()
    if float(video.detach().amax().item()) > 1.0 + 1e-4:
        video = video / 255.0
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    processed = pipeline.image_processor.preprocess(flat, height=height, width=width)
    processed = processed.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    vae_dtype = getattr(pipeline.vae, "dtype", torch.float32)
    with torch.inference_mode():
        encoded = pipeline.vae.encode(processed.to(device=device, dtype=vae_dtype))[0]
        if hasattr(encoded, "mode"):
            encoded = encoded.mode()
        elif hasattr(encoded, "latent_dist"):
            encoded = encoded.latent_dist.mode()
    return encoded.to(device=device)


class Method1HeatmapCapture:
    """Capture the first denoising forward and run fixed-sigma diagnostics."""

    _BATCH_TENSOR_KEYS = {
        "x",
        "context",
        "t",
        "y",
        "y_camera",
        "y_camera_mask",
        "arm_action",
        "arm_action_mask",
        "full_ref",
    }

    def __init__(
        self,
        pipeline: Any,
        target_latents,
        modes: Iterable[str],
        *,
        sigma: float = 0.5,
        eps: float = 1e-6,
    ):
        import torch

        self.pipeline = pipeline
        self.target_latents = target_latents
        self.modes = parse_weight_selection(modes)
        if not self.modes:
            raise ValueError("Method1HeatmapCapture requires at least one mode")
        if not 0.0 < float(sigma) < 1.0:
            raise ValueError(f"sigma must be in (0,1), got {sigma}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        if not torch.is_tensor(target_latents) or target_latents.ndim != 5:
            raise ValueError("target_latents must have shape [B,C,T,H,W]")
        self.sigma = float(sigma)
        self.eps = float(eps)
        self.maps: dict[str, Any] = {}
        self.rho_maps: dict[str, Any] = {}
        self.error: BaseException | None = None
        self.seen = False
        self.busy = False
        self.handles = []

    def _slice_batch(self, value, source_batch: int, batch: int):
        import torch

        if not torch.is_tensor(value) or value.ndim == 0:
            return value
        if value.shape[0] == batch:
            return value
        if value.shape[0] == source_batch and source_batch == 2 * batch:
            return value[batch:]
        return value

    def _diagnostic_timestep(self, original_t, batch: int, device):
        import torch

        num_train = int(
            getattr(getattr(self.pipeline, "scheduler", None), "config", {}).get(
                "num_train_timesteps", 1000
            )
            if hasattr(getattr(self.pipeline, "scheduler", None), "config")
            else 1000
        )
        value = float(self.sigma * num_train)
        if not torch.is_tensor(original_t):
            return torch.full((batch,), value, device=device, dtype=torch.float32)
        t = original_t
        if t.ndim == 0:
            return torch.full((), value, device=device, dtype=t.dtype)
        if t.ndim == 1:
            return torch.full((batch,), value, device=device, dtype=t.dtype)
        t = t[:batch]
        return torch.where(t != 0, torch.full_like(t, value), torch.zeros_like(t))

    def _base_kwargs(self, kwargs: Mapping[str, Any], source_batch: int, batch: int):
        return {
            key: self._slice_batch(value, source_batch, batch)
            if key in self._BATCH_TENSOR_KEYS
            else value
            for key, value in kwargs.items()
        }

    def _forward_hook(self, module, args, kwargs, output):
        import torch

        if self.seen or self.busy:
            return output
        self.seen = True
        self.busy = True
        try:
            x = kwargs.get("x")
            if not torch.is_tensor(x) or x.ndim != 5:
                raise ValueError("transformer first forward did not expose tensor kwarg x")
            clean_latents = self.target_latents.to(device=x.device, dtype=x.dtype)
            batch = clean_latents.shape[0]
            source_batch = x.shape[0]
            if source_batch not in (batch, 2 * batch):
                raise ValueError(
                    f"transformer batch {source_batch} is incompatible with target batch {batch}"
                )
            base = self._base_kwargs(kwargs, source_batch, batch)
            noise = base["x"].float()
            init_sigma = float(getattr(self.pipeline.scheduler, "init_noise_sigma", 1.0))
            if init_sigma > 0:
                noise = noise / init_sigma
            diag_x = (1.0 - self.sigma) * clean_latents.float() + self.sigma * noise
            if diag_x.shape[2] > 1:
                # Match the pipeline's fixed first-frame inpaint condition.
                diag_x[:, :, :1] = base["x"][:, :, :1].float()
            diag_kwargs = dict(base)
            diag_kwargs["x"] = diag_x.to(dtype=x.dtype)
            diag_kwargs["t"] = self._diagnostic_timestep(
                kwargs.get("t"), batch, x.device
            )
            with torch.no_grad():
                prediction = module(**diag_kwargs)
                if isinstance(prediction, (tuple, list)):
                    prediction = prediction[0]
                effect = None
                if any(mode in EFFECT_MODES for mode in self.modes):
                    null_kwargs = dict(diag_kwargs)
                    arm_mask = null_kwargs.get("arm_action_mask")
                    if torch.is_tensor(arm_mask):
                        null_kwargs["arm_action_mask"] = torch.zeros_like(arm_mask)
                    # Camera CAP uses y_camera/y_camera_mask instead of the
                    # arm action mask.  A missing mask means the condition is
                    # active, so create an explicit zero mask for the null
                    # diagnostic branch.
                    camera = null_kwargs.get("y_camera")
                    camera_mask = null_kwargs.get("y_camera_mask")
                    if torch.is_tensor(camera):
                        if torch.is_tensor(camera_mask):
                            null_kwargs["y_camera_mask"] = torch.zeros_like(camera_mask)
                        else:
                            null_kwargs["y_camera_mask"] = torch.zeros(
                                (camera.shape[0],),
                                device=camera.device,
                                dtype=torch.float32,
                            )
                    null_prediction = module(**null_kwargs)
                    if isinstance(null_prediction, (tuple, list)):
                        null_prediction = null_prediction[0]
                    effect = torch.linalg.vector_norm(
                        prediction.float() - null_prediction.float(),
                        ord=2,
                        dim=1,
                        keepdim=True,
                    )
                    del null_prediction
                diagnostic_target = noise - clean_latents.float()
                self.maps = {
                    mode: value.detach().float().cpu()
                    for mode, value in compute_weight_maps(
                        prediction,
                        diagnostic_target,
                        effect,
                        self.modes,
                        eps=self.eps,
                        exclude_first_frame=True,
                    ).items()
                }
                self.rho_maps = {
                    mode: value.detach().float().cpu()
                    for mode, value in compute_rho_maps(
                        prediction,
                        diagnostic_target,
                        effect,
                        self.modes,
                        eps=self.eps,
                        exclude_first_frame=True,
                    ).items()
                }
                del prediction, effect, diag_x, clean_latents, noise, diagnostic_target
        except BaseException as exc:  # surface it after the original forward returns
            self.error = exc
        finally:
            self.busy = False
        return output

    def __enter__(self):
        modules = [getattr(self.pipeline, "transformer", None)]
        transformer_2 = getattr(self.pipeline, "transformer_2", None)
        if transformer_2 is not None:
            modules.append(transformer_2)
        modules = [module for module in modules if module is not None]
        if not modules:
            raise ValueError("pipeline has no transformer to capture")
        for module in modules:
            try:
                self.handles.append(module.register_forward_hook(self._forward_hook, with_kwargs=True))
            except TypeError as exc:
                raise RuntimeError("PyTorch with_kwargs forward hooks are required for heatmaps") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc is None:
            if self.error is not None:
                raise RuntimeError("MSE heatmap diagnostic forward failed") from self.error
            if not self.seen:
                raise RuntimeError("MSE heatmap capture saw no transformer forward")
        return False
