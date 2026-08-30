# CAP S/E Heatmaps During Inference

This is the canonical handoff document for adding the current `S_only` and
`E_only` heatmaps to another inference entrypoint. Read this file together
with the following source files before changing an inference script:

1. `scripts/wan2.2_fun/arm_mse_heatmap.py` (formulas, interpolation,
   smoothing, colors, and overlay).
2. `scripts/wan2.2_fun/visualize_cap_arm_weights.py` (the complete generated
   video export flow, manifest, cache checks, and PNG layout).
3. `scripts/wan2.2_fun/visualize_cap_gt_weights.py` (the direct diagnostic
   forward over GT frames; no diffusion video is generated).
4. `scripts/wan2.2_fun/visualize_cap_gt_videos.py` (full 17-frame MP4 export,
   including reuse of existing cases and appending new random cases).
5. `scripts/wan2.2_fun/infer_cap_arm_sample.py` (model architecture and
   checkpoint loading).
6. Dataset-specific instructions: `poseanything-dreamzero-actionmap/POSEANYTHING.md`
   for PoseAnything, and
   `poseanything-dreamzero-actionmap/LIBERO_TI2V_LOSS_TRAINING.md` for LIBERO.

Do not invent a second heatmap implementation. Reuse the helpers in
`arm_mse_heatmap.py` and copy the constants in `RENDERING_CONFIG` from
`visualize_cap_arm_weights.py`.

## What Is Being Visualized

The model produces a latent prediction and a latent target/noise residual.
`compute_rho_maps()` returns the normalized diagnostic maps on the latent
token grid, with the fixed first frame removed:

```text
E_only = ||prediction - target||_2 / mean_future(||prediction - target||_2)
S_only = effect_map / mean_future(effect_map)
```

The means are computed per sample/episode over all future latent frames and
spatial tokens. In the standard CAP path, `effect_map` is the L2 difference
between a conditional transformer forward and a null-condition forward. The
normalization is not a loss change and must not be written back to model
weights or NPZ source arrays.

Use `compute_rho_maps(prediction, target, effect_map, ("s_only", "e_only"),
exclude_first_frame=True)`. The returned tensor is normally
`[batch, 1, future_frames, latent_height, latent_width]`. Do not include the
fixed first frame in the mean.

## Two Supported Integration Modes

### A. Existing generated-video inference

This is the normal path when the inference script already calls the diffusion
pipeline and has generated RGB frames.

1. Encode the GT target chunk with the same inference VAE used by the
   pipeline.
2. Create `Method1HeatmapCapture(pipeline, target_latents,
   ("s_only", "e_only"), sigma=0.5, eps=1e-6)`.
3. Enter the capture context around the existing `pipeline(...)` call. Do not
   replace or reorder the generation loop:

```python
capture = weight_viz.Method1HeatmapCapture(
    pipeline,
    target_latents,
    ("s_only", "e_only"),
    sigma=0.5,
    eps=1e-6,
)
with torch.inference_mode(), capture:
    sample = pipeline(...)
rho = capture.rho_maps
generated_frames = sample_to_rgb_frames(sample)
```

The hook observes the first transformer forward and runs the fixed-sigma
diagnostic forward internally. It is removed when the context exits. If the
inference code uses classifier-free guidance (a doubled batch), the helper
already handles the batch slicing. The caller must still provide a valid
`target_latents` tensor with shape `[B,C,T,H,W]`.

For chunked inference, keep one latent map per chunk, render each chunk with
`render_map_to_video()`, and concatenate chunks exactly as
`visualize_cap_arm_weights.py:capture_episode_weights()` does. Do not average
or renormalize maps across episodes.

### B. GT-only diagnostic inference

Use this mode when no generated video is needed. Keep the RGB video and any
control input from the dataset, encode the GT RGB/control inputs, and perform
the same fixed-sigma conditional/null transformer forwards. The reference
implementation is `visualize_cap_gt_weights.py:diagnose_case()`.

This mode still performs model diagnostic forwards; it is not a pixel
frame-difference approximation. For PoseAnything, the conditional input is
the skeleton latent and the null input is a black-skeleton latent. For
LIBERO, use the existing action/mask and reference-latent construction in
`diagnose_case()`. Preserve the architecture-specific tensor shapes.

## Canonical PNG Rendering

The current renderer is deliberately episode-local and has one shared scale
per weight type. For each `S_only` or `E_only` map:

1. Apply `smooth_latent_spatially(latent, sigma=1.5)` independently in space
   for each latent frame. Never smooth across time and never modify the saved
   raw NPZ array.
2. Trilinearly interpolate the future latent map to the requested RGB output
   size with `render_map_to_video(..., output_size=(height, width))`. The
   helper prepends one zero response for the fixed first frame.
3. Before interpolation, compute one episode-wide `vmax` from all positive,
   finite latent values after the first frame:

```python
vmax = positive_percentile_vmax(
    latent, percentile=99.0, exclude_first_frame=False
)
```

   In the generated-video path, the latent maps are already future-only, so
   `exclude_first_frame=False` is correct. In a path that still contains the
   fixed frame, use `True`.
4. Compute the display minimum from the interpolated response, excluding the
   first frame, and map every frame with the same episode scale:

```python
vmin = episode_response_vmin(interpolated, vmax)
response = normalize_weight_response(interpolated, vmax, vmin=vmin)
```

   This is the current `episode_interpolated_global_min_to_latent_positive_p99`
   behavior. There is no P2/P99.5 range, no per-frame range, no min-max pass,
   and no post-blur rescaling.
5. For each PNG only, apply exactly one pixel Gaussian blur:

```python
smooth = smooth_weight_response(response[frame], blur_radius=12.0)
```

   The response is clipped to `[0, 1]` before/after this blur. Do not blur or
   overwrite the NPZ source weights.
6. `overlay_weight_response()` passes the blurred result through the
   normalized sigmoid with `k=12`, then uses the six-stop reference palette
   with piecewise-linear interpolation at levels
   `0.0, 0.125, 0.375, 0.625, 0.875, 1.0`:

```text
(0, 0, 128), (0, 0, 255), (0, 255, 255),
(255, 255, 0), (255, 0, 0), (128, 0, 0)
```

7. Blend the heatmap over the RGB frame with the fixed formula:

```python
overlay = np.uint8(np.clip(
    rgb_frame.astype(np.float32) * 0.55
    + heat.astype(np.float32) * 0.65,
    0,
    255,
))
```

Do not substitute a colormap, alpha curve, neighborhood blend, second
normalization, or a different blur radius without changing the manifest
contract and updating this README.

## Output Contract

Use the existing per-case layout; do not create GIFs:

```text
<output-dir>/
  selection.json                 # random seed and selected IDs, if sampling
  case_<id>_<name>/
    S_only_weights.npz           # raw rendered rho; never display-smoothed/rescaled
    E_only_weights.npz
    S_only/frame_0004.png
    S_only/frame_0008.png
    ...
    E_only/frame_0004.png
    E_only/frame_0008.png
    ...
    manifest.json
  manifest.json                  # root manifest listing all cases
```

The PNG frame stride and final-frame behavior must match the host inference
script. The NPZ arrays must remain the raw trilinearly rendered rho weights,
without latent display smoothing, response normalization, or pixel blur, and
the manifest must record the actual `vmax`, `vmin`, selected frames, source
RGB/control paths, and the complete `RENDERING_CONFIG`. In particular, the
manifest values should be:

```text
normalization = episode_interpolated_global_min_to_latent_positive_p99
vmin = episode_interpolated_min_excluding_first_frame
percentile = 99.0
latent_spatial_smoothing = gaussian_sigma_1.5_before_interpolation
blur = single_gaussian_radius_12
response_rescaled_after_blur = false
color_response_curve = normalized_sigmoid_k12_after_blur
colormap = six_stop_blue_cyan_yellow_red_reference
overlay = 0.55_rgb_plus_0.65_heat
output = png
```

`visualize_cap_arm_weights.py:load_complete_episode_report()` is the
canonical completeness check. Reuse it or implement the same checks before
skipping an existing case; a stale rendering config must invalidate the
cache.

## Minimal Export Skeleton

The following is intentionally only the rendering portion. Keep the host
inference code responsible for loading the model, selecting data, and
producing RGB frames:

```python
import numpy as np
from PIL import Image
import arm_mse_heatmap as weight_viz

latent = rho[mode][0, 0].detach().cpu().numpy().astype(np.float32)
raw = weight_viz.render_map_to_video(
    rho[mode], frame_count, output_size=(height, width)
)
display = weight_viz.render_map_to_video(
    weight_viz.smooth_latent_spatially(latent, sigma=1.5),
    frame_count,
    output_size=(height, width),
)
vmax = weight_viz.positive_percentile_vmax(
    latent, percentile=99.0, exclude_first_frame=False
)
vmin = weight_viz.episode_response_vmin(display, vmax)
response = weight_viz.normalize_weight_response(display, vmax, vmin=vmin)

np.savez_compressed(case_dir / f"{name}_weights.npz", weights=raw)
for frame_index in selected_frames:
    overlay = weight_viz.overlay_weight_response(
        rgb_frames[frame_index], response[frame_index], blur_radius=12.0
    )
    Image.fromarray(overlay, mode="RGB").save(
        case_dir / name / f"frame_{frame_index:04d}.png",
        format="PNG",
        compress_level=2,
    )
```

Do not call `normalize_weight_response()` independently for each frame or
each chunk. Compute the episode scale after collecting all chunks, then
render all PNGs with that scale.

## Verification Checklist

Before reporting success, check all of the following:

- `rho` contains both `s_only` and `e_only`, and both exclude the fixed first
  frame in their means.
- The latent NPZ is finite and unchanged by display smoothing.
- Every episode/weight has exactly one latent positive P99 `vmax`.
- Interpolation is trilinear and targets the RGB frame size.
- Only one Gaussian blur with radius 12 is applied to each response frame.
- No P2/P99.5, per-frame min-max, neighborhood blend, or post-blur
  normalization appears in the PNG path.
- PNGs are RGB, have the expected dimensions, and no GIF is generated.
- Each case manifest points to real RGB/control inputs and lists every PNG and
  NPZ; the root manifest lists every case.
- Run `python -m unittest discover -s VideoX-Fun-CAP/tests -p
  'test_weight_visualization_rendering.py'` and inspect at least one PNG for
  nonzero spatial variation.

## Direct GT Example

For a random GT-only batch, use the existing script rather than rebuilding a
second pipeline:

```bash
PYTHONPATH=sources/cap \
python sources/cap/scripts/wan2.2_fun/visualize_cap_gt_weights.py \
  --mode poseanything \
  --checkpoint <checkpoint> \
  --metadata <poseanything-metadata.json> \
  --output-dir <checkpoint>/gt_weight_visualization_random10 \
  --sample-count 10 \
  --device cuda:0
```

For a generated-video inference entrypoint, use
`visualize_cap_arm_weights.py` as the export reference and put the
`Method1HeatmapCapture` context around the existing pipeline call. The
dataset README remains authoritative for how to locate RGB, skeleton, action,
annotation, and instruction files; this README is authoritative only for the
S/E diagnostic and rendering contract.

## Full GT MP4 Export

`visualize_cap_gt_videos.py` writes `S_only.mp4` and `E_only.mp4` for every
case. It renders all 17 sampled GT frames at `1280x704` and 8 FPS. Pass the
previous `gt_weight_visualization_random10` directory as `--existing-dir` to
reuse its saved NPZ maps, then set `--sample-count 20` to append ten new cases:

```bash
PYTHONPATH=sources/cap/scripts/wan2.2_fun \
python sources/cap/scripts/wan2.2_fun/visualize_cap_gt_videos.py \
  --mode poseanything \
  --checkpoint <checkpoint> \
  --metadata <poseanything-metadata.json> \
  --existing-dir <checkpoint>/gt_weight_visualization_random10 \
  --output-dir <checkpoint>/gt_weight_video_visualization_random20 \
  --sample-count 20 --device cuda:0
```

The output root manifest records all 20 IDs and each case manifest records
the two MP4 paths. These videos use GT RGB backgrounds with the fixed-sigma
diagnostic S/E maps; they are not diffusion-generated RGB videos.

## Prompt For Another Codex

The following prompt is sufficient when asking another Codex to add these
heatmaps to a different inference entrypoint:

```text
Read these files completely before editing:
- sources/cap/scripts/wan2.2_fun/HEATMAP_VISUALIZATION_README.md
- sources/cap/scripts/wan2.2_fun/arm_mse_heatmap.py
- sources/cap/scripts/wan2.2_fun/visualize_cap_arm_weights.py
- the dataset/model README used by this inference entrypoint

Incrementally add simultaneous S_only/E_only heatmap capture and PNG export
to the existing inference flow. Reuse arm_mse_heatmap.py; do not rewrite the
formulas or renderer, and do not change model inference, loss, checkpoint
weights, or generated RGB outputs. Preserve the host script's episode/case
selection and resume behavior. Save PNG, NPZ, and manifest artifacts using
the exact rendering/output contract in HEATMAP_VISUALIZATION_README.md, then
run the documented rendering tests and validate one complete case.
```
