#!/usr/bin/env python3
"""Plot CAP arm CAER/MSE losses from TensorBoard event files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from plot_libero_loss_curves import draw_chart


def load_scalar(event_path: Path, tag: str, start: int, end: int) -> dict[int, float]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    values = accumulator.Scalars(tag)
    result = {int(value.step): float(value.value) for value in values if start <= value.step <= end}
    expected = set(range(start, end + 1))
    missing = sorted(expected.difference(result))
    if missing:
        raise RuntimeError(f"{event_path}: {tag} missing {len(missing)} steps, first={missing[:5]}")
    return result


def first_scalar_event(run_root: Path) -> Path:
    candidates = []
    for path in run_root.rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if "method1_uniform_loss" in accumulator.Tags().get("scalars", []):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"no scalar event with method1_uniform_loss under {run_root}")
    # The original event is the one beginning at step 1; resumed events begin later.
    candidates.sort(key=lambda path: min(v.step for v in EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload().Scalars("method1_uniform_loss")))
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s-root", type=Path, required=True, help="Arm caer run directory")
    parser.add_argument("--u-root", type=Path, required=True, help="Arm uniform run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--end", type=int, default=5000)
    args = parser.parse_args()
    start, end = args.start, args.end
    s_root = args.s_root
    u_root = args.u_root
    s_event = first_scalar_event(s_root)
    u_event = first_scalar_event(u_root)
    s_weighted = load_scalar(s_event, "method1_weighted_loss", start, end)
    s_uniform = load_scalar(s_event, "method1_uniform_loss", start, end)
    u_uniform = load_scalar(u_event, "method1_uniform_loss", start, end)
    steps = np.arange(start, end + 1, dtype=np.float64)
    s_weighted_values = np.asarray([s_weighted[int(step)] for step in steps])
    s_uniform_values = np.asarray([s_uniform[int(step)] for step in steps])
    u_uniform_values = np.asarray([u_uniform[int(step)] for step in steps])

    out = args.output_dir.resolve()
    series = {
        "CAER": ("CAER weighted_loss", s_weighted_values, (220, 55, 55)),
        "CAER_MSE": ("CAER MSE_loss", s_uniform_values, (239, 126, 34)),
        "MSE": ("MSE loss", u_uniform_values, (45, 105, 205)),
    }
    draw_chart(out / "01_CAER_two_losses.png", "CAER: weighted loss and MSE loss", [series["CAER"], series["CAER_MSE"]], 20.0, 101, x_start=start, x_end=end)
    draw_chart(out / "02_MSE_loss_comparison.png", "MSE loss: CAER vs MSE", [series["CAER_MSE"], series["MSE"]], 20.0, 101, x_start=start, x_end=end)
    draw_chart(out / "03_all_two_losses.png", "CAER and MSE losses", [series["CAER"], series["CAER_MSE"], series["MSE"]], 20.0, 101, x_start=start, x_end=end)
    with (out / "loss_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "caer_weighted_loss", "caer_uniform_loss", "uniform_uniform_loss"])
        for index, step in enumerate(steps.astype(int)):
            writer.writerow([step, s_weighted_values[index], s_uniform_values[index], u_uniform_values[index]])
    metadata = {
        "caer_event": str(s_event),
        "uniform_event": str(u_event),
        "steps": [start, end],
        "points_per_series": len(steps),
        "smoothing": {"center": "gaussian", "sigma": 20.0, "envelope": "rolling_quantile_p10_p90", "window": 101},
        "colors": {"caer_weighted": "#dc3737", "caer_uniform": "#ef7e22", "uniform_uniform": "#2d69cd"},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(out)
    print("caer_event", s_event)
    print("uniform_event", u_event)
    print("points", len(steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
