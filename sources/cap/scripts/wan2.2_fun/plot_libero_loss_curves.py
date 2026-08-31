#!/usr/bin/env python3
"""Draw the three requested CAP loss comparisons without matplotlib."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


STAT_RE = re.compile(
    r"method1_stats step=(?P<step>\d+) variant=(?P<variant>[\w-]+)"
    r".*?focused_loss=(?P<focused>[0-9.eE+-]+)"
    r" uniform_loss=(?P<uniform>[0-9.eE+-]+)"
)


def parse_stats(path: Path, expected_variant: str, max_step: int) -> dict[str, np.ndarray]:
    rows: dict[int, tuple[float, float]] = {}
    for line in path.open("r", encoding="utf-8", errors="ignore"):
        match = STAT_RE.search(line)
        if match is None or match.group("variant") != expected_variant:
            continue
        step = int(match.group("step"))
        if 0 <= step <= max_step:
            rows[step] = (float(match.group("focused")), float(match.group("uniform")))
    expected = set(range(max_step + 1))
    missing = sorted(expected.difference(rows))
    if missing:
        raise RuntimeError(f"{path}: missing {len(missing)} method1_stats steps, first={missing[:5]}")
    steps = np.arange(max_step + 1, dtype=np.float64)
    return {
        "step": steps,
        "focused_loss": np.asarray([rows[i][0] for i in range(max_step + 1)], dtype=np.float64),
        "uniform_loss": np.asarray([rows[i][1] for i in range(max_step + 1)], dtype=np.float64),
    }


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def rolling_quantile(values: np.ndarray, quantile: float, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, (half, half), mode="edge")
    output = np.empty_like(values)
    for index in range(len(values)):
        output[index] = np.quantile(padded[index : index + window], quantile)
    return output


def envelope(values: np.ndarray, sigma: float, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = gaussian_smooth(values, sigma)
    lower = rolling_quantile(values, 0.10, window)
    upper = rolling_quantile(values, 0.90, window)
    return center, lower, upper


def font(size: int, bold: bool = False):
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_chart(
    output: Path,
    title: str,
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    sigma: float,
    envelope_window: int,
    x_start: float = 0.0,
    x_end: float = 5000.0,
) -> None:
    scale = 2
    width, height = 1800 * scale, 1000 * scale
    left, right, top, bottom = 150 * scale, 70 * scale, 100 * scale, 150 * scale
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    all_values = np.concatenate([values[np.isfinite(values)] for _, values, _ in series])
    # Include the warm-up spike so no real step is clipped from the chart.
    y_max = max(float(all_values.max()) * 1.08, 1e-3)
    y_max = math.ceil(y_max * 10.0) / 10.0
    if x_end <= x_start:
        raise ValueError("x_end must be greater than x_start")

    image = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(image)
    title_font = font(30 * scale, bold=True)
    label_font = font(21 * scale)
    tick_font = font(17 * scale)
    legend_font = font(18 * scale, bold=True)
    draw.text((left, 32 * scale), title, fill=(30, 35, 45), font=title_font)

    def x_pixel(value: float) -> int:
        fraction = (value - x_start) / (x_end - x_start)
        return int(round(plot_left + fraction * (plot_right - plot_left)))

    def y_pixel(value: float) -> int:
        pixel = plot_bottom - (value / y_max) * (plot_bottom - plot_top)
        return int(round(np.clip(pixel, plot_top, plot_bottom)))

    # Grid and axes.
    for tick in range(0, 6):
        value = y_max * tick / 5.0
        y = y_pixel(value)
        draw.line((plot_left, y, plot_right, y), fill=(220, 224, 230), width=scale)
        text = f"{value:.2f}"
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((plot_left - (box[2] - box[0]) - 14 * scale, y - (box[3] - box[1]) // 2), text, fill=(85, 90, 100), font=tick_font)
    tick_values = sorted({int(x_start), *range(1000, int(x_end) + 1, 1000), int(x_end)})
    for value in tick_values:
        x = x_pixel(value)
        draw.line((x, plot_top, x, plot_bottom), fill=(232, 235, 240), width=scale)
        text = str(value)
        box = draw.textbbox((0, 0), text, font=tick_font)
        draw.text((x - (box[2] - box[0]) // 2, plot_bottom + 14 * scale), text, fill=(85, 90, 100), font=tick_font)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=(65, 70, 80), width=2 * scale)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=(65, 70, 80), width=2 * scale)
    draw.text(((plot_left + plot_right) // 2 - 65 * scale, height - 70 * scale), "Training step", fill=(45, 50, 60), font=label_font)
    draw.text((25 * scale, (plot_top + plot_bottom) // 2 - 25 * scale), "Loss", fill=(45, 50, 60), font=label_font)

    for name, values, color in series:
        center, lower, upper = envelope(values, sigma, envelope_window)
        points_lower = [(x_pixel(x_start + i), y_pixel(float(v))) for i, v in enumerate(lower)]
        points_upper = [(x_pixel(x_start + i), y_pixel(float(v))) for i, v in enumerate(upper)]
        points_center = [(x_pixel(x_start + i), y_pixel(float(v))) for i, v in enumerate(center)]
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        polygon = points_lower + list(reversed(points_upper))
        layer_draw.polygon(polygon, fill=(*color, 42))
        layer_draw.line(points_lower, fill=(*color, 155), width=2 * scale, joint="curve")
        layer_draw.line(points_upper, fill=(*color, 155), width=2 * scale, joint="curve")
        image = Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.line(points_center, fill=color, width=4 * scale, joint="curve")

    # Legend describes the central line; each matching translucent band is the local P10-P90 envelope.
    legend_x, legend_y = plot_left + 25 * scale, plot_top + 20 * scale
    for name, _, color in series:
        draw.line((legend_x, legend_y + 12 * scale, legend_x + 40 * scale, legend_y + 12 * scale), fill=color, width=4 * scale)
        draw.text((legend_x + 52 * scale, legend_y), name, fill=(35, 40, 50), font=legend_font)
        legend_y += 34 * scale
    note = f"center: Gaussian smooth sigma={sigma:g}; band: rolling P10-P90 window={envelope_window}"
    draw.text((plot_left, plot_bottom + 62 * scale), note, fill=(100, 105, 115), font=tick_font)

    output.parent.mkdir(parents=True, exist_ok=True)
    image.resize((width // scale, height // scale), Image.Resampling.LANCZOS).save(output, format="PNG", optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-step", type=int, default=4999)
    parser.add_argument("--smooth-sigma", type=float, default=20.0)
    parser.add_argument("--envelope-window", type=int, default=101)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--s-root", type=Path, required=True, help="LIBERO CAER run directory")
    parser.add_argument("--u-root", type=Path, required=True, help="LIBERO MSE run directory")
    args = parser.parse_args()
    if args.max_step < 1 or args.smooth_sigma <= 0 or args.envelope_window < 3 or args.envelope_window % 2 == 0:
        raise ValueError("max-step >= 1, smooth-sigma > 0, and odd envelope-window >= 3 are required")

    s_root = args.s_root
    u_root = args.u_root
    s_log = next((s_root / "logs").glob("train_*.log"))
    u_log = next((u_root / "logs").glob("train_*.log"))
    s = parse_stats(s_log, "CAER", args.max_step)
    u = parse_stats(u_log, "MSE", args.max_step)
    series = {
        "caer_focused": ("caer focused_loss", s["focused_loss"], (220, 55, 55)),
        "caer_uniform": ("caer uniform_loss", s["uniform_loss"], (239, 126, 34)),
        "uniform_uniform": ("uniform uniform_loss", u["uniform_loss"], (45, 105, 205)),
    }
    out = args.output_dir.resolve()
    draw_chart(out / "01_caer_two_losses.png", "caer: focused loss and uniform loss", [series["caer_focused"], series["caer_uniform"]], args.smooth_sigma, args.envelope_window)
    draw_chart(out / "02_uniform_loss_comparison.png", "uniform loss: caer vs uniform", [series["caer_uniform"], series["uniform_uniform"]], args.smooth_sigma, args.envelope_window)
    draw_chart(out / "03_all_three_losses.png", "all requested losses", [series["caer_focused"], series["caer_uniform"], series["uniform_uniform"]], args.smooth_sigma, args.envelope_window)

    with (out / "loss_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "caer_focused_loss", "caer_uniform_loss", "uniform_uniform_loss"])
        for i in range(args.max_step + 1):
            writer.writerow([i, s["focused_loss"][i], s["uniform_loss"][i], u["uniform_loss"][i]])
    metadata = {
        "caer_log": str(s_log),
        "uniform_log": str(u_log),
        "steps": [0, args.max_step],
        "points_per_series": args.max_step + 1,
        "smoothing": {"center": "gaussian", "sigma": args.smooth_sigma, "envelope": "rolling_quantile_p10_p90", "window": args.envelope_window},
        "colors": {"caer_focused": "#dc3737", "caer_uniform": "#ef7e22", "uniform_uniform": "#2d69cd"},
    }
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(out)
    print("points", args.max_step + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
