#!/usr/bin/env python3
"""Plot Method1 uniform loss for the five Arm CAP ablation runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


VARIANTS = ("MSE", "CAER")


def latest_run(root: Path, variant: str) -> Path:
    variant_root = root / variant
    candidates = sorted(
        path
        for path in variant_root.glob("20*")
        if path.is_dir() and (path / "train_metrics.jsonl").is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"no timestamped run with train_metrics.jsonl found for {variant}: {variant_root}"
        )
    return candidates[-1]


def read_uniform_loss(path: Path, start: int | None, end: int | None) -> tuple[list[int], list[float], int]:
    steps: list[int] = []
    losses: list[float] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A live JSONL file can have a partially flushed final line.
                malformed += 1
                continue
            step = record.get("global_step")
            loss = record.get("method1_uniform_loss")
            if not isinstance(step, (int, float)) or not isinstance(loss, (int, float)):
                continue
            step = int(step)
            loss = float(loss)
            if not math.isfinite(loss):
                continue
            if start is not None and step < start:
                continue
            if end is not None and step > end:
                continue
            steps.append(step)
            losses.append(loss)
    return steps, losses, malformed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the MSE diagnostic from the latest timestamped Arm run of "
            "MSE and CAER."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("ARM_RUNS_ROOT", "outputs/arm")),
        help="Arm ablation root containing one directory per variant.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        metavar="STEP",
        help="first global_step to include (inclusive); default: beginning",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        metavar="STEP",
        help="last global_step to include (inclusive); default: end",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path; default: <root>/plots/arm_uniform_loss_comparison.png",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=int(os.environ.get("BIN_SIZE", "200")),
        metavar="STEPS",
        help="steps per terminal summary bin; default: BIN_SIZE or 200",
    )
    parser.add_argument(
        "--anchor-step",
        type=int,
        default=int(os.environ.get("ANCHOR_STEP", "1")),
        metavar="STEP",
        help="first step of the bin grid; default: ANCHOR_STEP or 1",
    )
    parser.add_argument(
        "--print-last-bins",
        type=int,
        default=int(os.environ.get("PRINT_LAST_BINS", "0")),
        metavar="N",
        help="only print the latest N bins; 0 prints all bins",
    )
    parser.add_argument("--dpi", type=int, default=160, help="output DPI (default: 160)")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font, fill) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, xy[1] - (bbox[3] - bbox[1]) / 2), text, font=font, fill=fill)


def format_tick(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def draw_plot(
    series: dict[str, tuple[list[float], list[float]]],
    output: Path,
    range_label: str,
    dpi: int,
    x_bounds: tuple[float, float],
) -> None:
    width, height = 1800, 1000
    left, right, top, bottom = 135, 70, 105, 135
    plot_left, plot_right = left, width - right
    plot_top, plot_bottom = top, height - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    label_font = load_font(22)
    tick_font = load_font(19)
    legend_font = load_font(21)
    text_color = (35, 35, 35)
    grid_color = (220, 224, 230)
    axis_color = (70, 70, 70)
    colors = {
        "MSE": (31, 119, 180),
        "CAER": (44, 160, 44),
    }

    all_losses = [loss for _, losses in series.values() for loss in losses]
    x_min, x_max = x_bounds
    y_min, y_max = min(all_losses), max(all_losses)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_span = y_max - y_min
    y_pad = y_span * 0.08 if y_span else max(abs(y_max) * 0.08, 1.0)
    y_min -= y_pad
    y_max += y_pad

    def x_pixel(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def y_pixel(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    draw_centered(draw, (width / 2, 35), f"Arm CAP ablation: Method1 uniform loss ({range_label})", title_font, text_color)

    tick_count = 6
    for index in range(tick_count):
        fraction = index / (tick_count - 1)
        x_value = x_min + fraction * (x_max - x_min)
        x = x_pixel(x_value)
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
        draw_centered(draw, (x, plot_bottom + 28), f"{x_value:.0f}", tick_font, text_color)
        y_value = y_min + fraction * (y_max - y_min)
        y = y_pixel(y_value)
        draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
        draw.text((plot_left - 18, y - 12), format_tick(y_value), font=tick_font, fill=text_color, anchor="ra")

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=axis_color, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=axis_color, width=2)
    draw_centered(draw, ((plot_left + plot_right) / 2, height - 45), "Global step", label_font, text_color)
    draw.text((80, (plot_top + plot_bottom) / 2), "Uniform loss", font=label_font, fill=text_color, anchor="mm")

    for variant, (steps, losses) in series.items():
        points = [(x_pixel(step), y_pixel(loss)) for step, loss in zip(steps, losses)]
        if len(points) >= 2:
            draw.line(points, fill=colors[variant], width=3, joint="curve")
        for x, y in points:
            draw.ellipse(
                (x - 4, y - 4, x + 4, y + 4),
                fill=colors[variant],
                outline="white",
                width=1,
            )

    legend_x, legend_y = plot_right - 220, plot_top + 18
    for variant in VARIANTS:
        draw.line((legend_x, legend_y + 10, legend_x + 32, legend_y + 10), fill=colors[variant], width=4)
        draw.text((legend_x + 44, legend_y), variant, font=legend_font, fill=text_color)
        legend_y += 31

    image.save(output, format="PNG", dpi=(dpi, dpi))


def summarize_bins(
    steps: list[int],
    losses: list[float],
    bin_size: int,
    anchor_step: int,
) -> dict[int, dict[str, float | int]]:
    buckets: dict[int, list[float]] = {}
    observed_steps: dict[int, list[int]] = {}
    for step, loss in zip(steps, losses):
        bin_start = anchor_step + ((step - anchor_step) // bin_size) * bin_size
        buckets.setdefault(bin_start, []).append(loss)
        observed_steps.setdefault(bin_start, []).append(step)

    summaries: dict[int, dict[str, float | int]] = {}
    for bin_start, values in buckets.items():
        bin_steps = observed_steps[bin_start]
        summaries[bin_start] = {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "first": values[0],
            "last": values[-1],
            "n": len(values),
            "observed_start": min(bin_steps),
            "observed_end": max(bin_steps),
        }
    return summaries


def binned_plot_series(
    series: dict[str, tuple[list[int], list[float]]],
    bin_size: int,
    anchor_step: int,
) -> dict[str, tuple[list[float], list[float]]]:
    result: dict[str, tuple[list[float], list[float]]] = {}
    for variant in VARIANTS:
        bins = summarize_bins(*series[variant], bin_size, anchor_step)
        centers: list[float] = []
        means: list[float] = []
        for bin_start in sorted(bins):
            summary = bins[bin_start]
            centers.append(
                (int(summary["observed_start"]) + int(summary["observed_end"])) / 2
            )
            means.append(float(summary["mean"]))
        result[variant] = (centers, means)
    return result


def print_quantification(
    series: dict[str, tuple[list[int], list[float]]],
    bin_size: int,
    anchor_step: int,
    print_last_bins: int,
) -> None:
    print()
    print("=== SELECTED-RANGE UNIFORM LOSS SUMMARY ===")
    print(
        f"{'variant':<9} {'points':>7} {'steps':>15} {'mean':>10} {'median':>10} "
        f"{'min':>10} {'max':>10} {'first':>10} {'last':>10}"
    )
    for variant in VARIANTS:
        steps, losses = series[variant]
        if not steps:
            print(f"{variant:<9} {0:>7} {'N/A':>15}")
            continue
        print(
            f"{variant:<9} {len(losses):>7} {f'{steps[0]}-{steps[-1]}':>15} "
            f"{statistics.fmean(losses):>10.6f} {statistics.median(losses):>10.6f} "
            f"{min(losses):>10.6f} {max(losses):>10.6f} "
            f"{losses[0]:>10.6f} {losses[-1]:>10.6f}"
        )

    binned = {
        variant: summarize_bins(*series[variant], bin_size, anchor_step)
        for variant in VARIANTS
    }
    bin_starts = sorted(
        {bin_start for variant_bins in binned.values() for bin_start in variant_bins}
    )
    if print_last_bins > 0:
        bin_starts = bin_starts[-print_last_bins:]

    print()
    print(
        f"=== BINNED MEAN UNIFORM LOSS (BIN_SIZE={bin_size}, ANCHOR_STEP={anchor_step}) ==="
    )
    print(f"{'bin':<15} " + " ".join(f"{variant + ' mean (n)':>20}" for variant in VARIANTS))
    for bin_start in bin_starts:
        cells = []
        for variant in VARIANTS:
            summary = binned[variant].get(bin_start)
            if summary is None:
                cells.append(f"{'N/A':>20}")
            else:
                cells.append(
                    f"{float(summary['mean']):.6f} ({int(summary['n']):>4})".rjust(20)
                )
        bin_end = bin_start + bin_size - 1
        print(f"{f'{bin_start}-{bin_end}':<15} " + " ".join(cells))
    if print_last_bins > 0:
        total_bins = len(
            {bin_start for variant_bins in binned.values() for bin_start in variant_bins}
        )
        if len(bin_starts) < total_bins:
            print(f"... showing the last {len(bin_starts)} of {total_bins} bins")


def main() -> int:
    args = parse_args()
    if args.start is not None and args.start < 0:
        raise SystemExit("--start must be >= 0")
    if args.end is not None and args.end < 0:
        raise SystemExit("--end must be >= 0")
    if args.start is not None and args.end is not None and args.start > args.end:
        raise SystemExit("--start must be <= --end")
    if args.bin_size <= 0:
        raise SystemExit("--bin-size must be > 0")
    if args.print_last_bins < 0:
        raise SystemExit("--print-last-bins must be >= 0")

    root = args.root.expanduser().resolve()
    output = args.output or (root / "plots" / "arm_uniform_loss_comparison.png")
    output = output.expanduser().resolve()

    series: dict[str, tuple[list[int], list[float]]] = {}
    for variant in VARIANTS:
        run_dir = latest_run(root, variant)
        steps, losses, malformed = read_uniform_loss(
            run_dir / "train_metrics.jsonl", args.start, args.end
        )
        series[variant] = (steps, losses)
        suffix = f"; skipped malformed lines={malformed}" if malformed else ""
        print(f"{variant}: run={run_dir} points={len(steps)}{suffix}")

    if not any(steps for steps, _ in series.values()):
        raise SystemExit("the selected range contains no uniform-loss points")

    output.parent.mkdir(parents=True, exist_ok=True)
    range_label = "all steps"
    if args.start is not None or args.end is not None:
        range_label = f"steps {args.start if args.start is not None else 'beginning'}"
        range_label += f"-{args.end if args.end is not None else 'end'}"
    range_label += f", bin mean={args.bin_size}"
    observed_steps = [step for steps, _ in series.values() for step in steps]
    x_start = min(observed_steps)
    if x_start <= 1:
        x_start = 0
    draw_plot(
        binned_plot_series(series, args.bin_size, args.anchor_step),
        output,
        range_label,
        args.dpi,
        (x_start, max(observed_steps)),
    )
    print(f"saved: {output}")
    print_quantification(
        series,
        bin_size=args.bin_size,
        anchor_step=args.anchor_step,
        print_last_bins=args.print_last_bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
