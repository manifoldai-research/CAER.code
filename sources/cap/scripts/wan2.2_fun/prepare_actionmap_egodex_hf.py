#!/usr/bin/env python3
"""Render EgoDex hand transforms as action-map videos and CAP metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import cv2
import h5py
import numpy as np


FINGER_PARTS = (
    "Metacarpal",
    "Knuckle",
    "IntermediateBase",
    "IntermediateTip",
    "Tip",
)
FINGERS = ("Thumb", "IndexFinger", "MiddleFinger", "RingFinger", "LittleFinger")
SIDES = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a Hugging Face UCBProject/EgoDex snapshot into paired RGB/"
            "hand-skeleton action-map metadata for VideoX-Fun-CAP."
        )
    )
    parser.add_argument("--egodex-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=8, help="0 means all pairs.")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def uniform_indices(frame_count: int, num_frames: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 1:
        return [0]
    return [round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]


def scalar_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def prompt_from_attrs(attrs: h5py.AttributeManager) -> str:
    if scalar_text(attrs.get("llm_type", "")) == "reversible":
        direction = scalar_text(attrs.get("which_llm_description", "1"))
        key = "llm_description" if direction == "1" else "llm_description2"
        return scalar_text(attrs.get(key, attrs.get("llm_description", "")))
    return scalar_text(attrs.get("llm_description", ""))


def required_joint_names() -> list[str]:
    names: list[str] = []
    for side in SIDES:
        names.extend((f"{side}Forearm", f"{side}Hand"))
        for finger in FINGERS:
            parts = FINGER_PARTS[1:] if finger == "Thumb" else FINGER_PARTS
            names.extend(f"{side}{finger}{part}" for part in parts)
    return names


def probe_video(path: Path) -> tuple[int, int, int, float]:
    reader = cv2.VideoCapture(str(path))
    if not reader.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(reader.get(cv2.CAP_PROP_FPS))
    ok, _ = reader.read()
    reader.release()
    if not ok or width <= 0 or height <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid video stream: {path}")
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0
    return width, height, frame_count, fps


def project(point: np.ndarray, intrinsic: np.ndarray) -> tuple[int, int] | None:
    if point.shape != (3,) or not np.isfinite(point).all() or point[2] <= 1e-6:
        return None
    pixel = intrinsic @ point
    x = float(pixel[0] / pixel[2])
    y = float(pixel[1] / pixel[2])
    if not math.isfinite(x) or not math.isfinite(y) or abs(x) > 1e7 or abs(y) > 1e7:
        return None
    return int(round(x)), int(round(y))


def draw_segment(
    image: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    point_a = project(first, intrinsic)
    point_b = project(second, intrinsic)
    if point_a is None or point_b is None:
        return
    cv2.line(image, point_a, point_b, color, thickness, cv2.LINE_AA)
    radius = max(2, thickness)
    cv2.circle(image, point_a, radius, color, -1, cv2.LINE_AA)
    cv2.circle(image, point_b, radius, color, -1, cv2.LINE_AA)


def render_pair(
    hdf5_path: Path,
    video_path: Path,
    output_path: Path,
    confidence_threshold: float,
    overwrite: bool,
    validate_only: bool,
) -> tuple[int, int, int, float, str]:
    width, height, video_frames, fps = probe_video(video_path)
    names = required_joint_names()
    with h5py.File(hdf5_path, "r") as handle:
        if "camera/intrinsic" not in handle or "transforms/camera" not in handle:
            raise ValueError("HDF5 is missing camera intrinsics or transforms")
        missing = [name for name in names if f"transforms/{name}" not in handle]
        if missing:
            raise ValueError("HDF5 is missing hand transforms: " + ", ".join(missing[:5]))
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        camera = np.asarray(handle["transforms/camera"], dtype=np.float64)
        transforms = {
            name: np.asarray(handle[f"transforms/{name}"], dtype=np.float64)
            for name in names
        }
        confidences = {
            name: np.asarray(handle[f"confidences/{name}"], dtype=np.float32)
            for name in names
            if f"confidences/{name}" in handle
        }
        prompt = prompt_from_attrs(handle.attrs)

    frame_count = min(
        [video_frames, len(camera), *(len(value) for value in transforms.values())]
    )
    if frame_count <= 0:
        raise ValueError("RGB video and HDF5 have no overlapping frames")
    if validate_only:
        return width, height, frame_count, fps, prompt
    if output_path.is_file() and not overwrite:
        return width, height, frame_count, fps, prompt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(
        str(temporary), cv2.VideoWriter_fourcc(*"FFV1"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the FFV1 action-map writer")

    thickness = max(2, round(min(width, height) / 240))
    color = (255, 144, 32)

    def point_is_active(name: str, frame_index: int) -> bool:
        values = confidences.get(name)
        if values is None or frame_index >= len(values):
            return True
        value = float(values[frame_index])
        return math.isfinite(value) and value >= confidence_threshold

    try:
        for frame_index in range(frame_count):
            image = np.zeros((height, width, 3), dtype=np.uint8)
            camera_inverse = np.linalg.inv(camera[frame_index])
            points: dict[str, np.ndarray] = {}
            for name, values in transforms.items():
                if point_is_active(name, frame_index):
                    points[name] = (camera_inverse @ values[frame_index])[:3, 3]

            for side in SIDES:
                forearm = f"{side}Forearm"
                hand = f"{side}Hand"
                if forearm in points and hand in points:
                    draw_segment(
                        image,
                        points[forearm],
                        points[hand],
                        intrinsic,
                        color,
                        thickness,
                    )
                for finger in FINGERS:
                    parts = FINGER_PARTS[1:] if finger == "Thumb" else FINGER_PARTS
                    sequence = [hand, *(f"{side}{finger}{part}" for part in parts)]
                    for first, second in zip(sequence, sequence[1:]):
                        if first in points and second in points:
                            draw_segment(
                                image,
                                points[first],
                                points[second],
                                intrinsic,
                                color,
                                thickness,
                            )
            writer.write(image)
        writer.release()
        os.replace(temporary, output_path)
    except Exception:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise
    return width, height, frame_count, fps, prompt


def main() -> int:
    args = parse_args()
    root = args.egodex_root.resolve()
    output_root = args.output_root.resolve()
    if not root.is_dir():
        raise SystemExit(f"EgoDex root is not a directory: {root}")
    if args.num_frames < 1 or (args.num_frames - 1) % 4 != 0:
        raise SystemExit("--num-frames must have form 4n+1")
    if args.confidence_threshold < 0:
        raise SystemExit("--confidence-threshold must be nonnegative")

    candidates = [
        path
        for path in sorted(root.rglob("*.hdf5"))
        if path.with_suffix(".mp4").is_file()
    ]
    if args.max_samples > 0:
        candidates = candidates[: args.max_samples]
    if not candidates:
        raise SystemExit(f"No sibling .hdf5/.mp4 pairs found under {root}")

    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for hdf5_path in candidates:
        video_path = hdf5_path.with_suffix(".mp4")
        relative = hdf5_path.relative_to(root)
        action_path = output_root / "action_map" / relative.with_suffix(".avi")
        try:
            width, height, frame_count, fps, prompt = render_pair(
                hdf5_path,
                video_path,
                action_path,
                args.confidence_threshold,
                args.overwrite,
                args.validate_only,
            )
            record = {
                "type": "video",
                "file_path": str(video_path.resolve()),
                "video_path": str(video_path.resolve()),
                "text": prompt,
                "control_type": "action_map",
                "control_file_path": str(action_path.resolve()),
                "pose_video_path": str(action_path.resolve()),
                "video_sample_n_frames": args.num_frames,
                "video_sample_stride": 1,
                "sampling": {
                    "mode": "explicit_frame_indices_egodex_uniform",
                    "frame_indices": uniform_indices(frame_count, args.num_frames),
                    "num_frames": args.num_frames,
                    "padding": "repeat_indices" if frame_count < args.num_frames else "none",
                },
                "source": {
                    "dataset": "UCBProject/EgoDex",
                    "relative_hdf5": relative.as_posix(),
                    "width": width,
                    "height": height,
                    "fps": fps,
                },
            }
            records.append(record)
            print(f"ok {relative}")
        except Exception as exc:
            failure = {"sample": relative.as_posix(), "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print("failed " + json.dumps(failure, sort_keys=True))

    if not records:
        raise SystemExit("No EgoDex samples were converted or validated")
    if not args.validate_only:
        output_root.mkdir(parents=True, exist_ok=True)
        metadata_path = output_root / "metadata_actionmap_egodex.json"
        temporary = metadata_path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, metadata_path)
        failures_path = output_root / "failed_samples.jsonl"
        with failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"metadata={metadata_path} samples={len(records)} failures={len(failures)}")
    else:
        print(f"validated={len(records)} failures={len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
