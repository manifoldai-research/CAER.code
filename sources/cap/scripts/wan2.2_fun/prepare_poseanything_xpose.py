#!/usr/bin/env python
import argparse
import ast
import csv
import concurrent.futures
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
from decord import VideoReader
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert XPose archives into aligned RGB/skeleton videos for CAP training."
    )
    parser.add_argument("--xpose_root", default=os.environ.get("XPOSE_ROOT", "data/XPose"))
    parser.add_argument("--output_root", default=os.environ.get("POSE_DATA_ROOT", "data/XPose/processed"))
    parser.add_argument(
        "--final_output_root",
        default=None,
        help="Optional published root recorded in metadata when writing to local staging.",
    )
    parser.add_argument("--batches", nargs="*", default=None, help="Batch stems such as batch_0000.")
    parser.add_argument("--max_samples", type=int, default=8, help="0 means all usable samples.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent batch workers for conversion; max_samples>0 stays serial.",
    )
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--fps", type=float, default=0.0, help="0 preserves the source-video FPS.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    return parser.parse_args()


def probe_video(path):
    reader = VideoReader(str(path), num_threads=1)
    if len(reader) <= 0:
        raise RuntimeError(f"Video has no frames: {path}")
    frame = reader[0].asnumpy()
    fps = float(reader.get_avg_fps())
    return int(frame.shape[1]), int(frame.shape[0]), int(len(reader)), fps


def uniform_indices(frame_count, num_frames):
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if num_frames <= 1:
        return [0]
    return [round(i * (frame_count - 1) / (num_frames - 1)) for i in range(num_frames)]


def parse_list(value, field_name, sample_name):
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid {field_name} for {sample_name}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} must be a list for {sample_name}")
    return parsed


def archive_batches(root, requested):
    root = Path(root)
    stems = requested or sorted(path.stem for path in root.glob("batch_*.csv"))
    for stem in stems:
        csv_path = root / f"{stem}.csv"
        zip_path = root / f"{stem}.zip"
        if not csv_path.is_file():
            print(f"skip {stem}: missing {csv_path}")
            continue
        if not zip_path.is_file() or not zipfile.is_zipfile(zip_path):
            print(f"skip {stem}: archive is missing or incomplete: {zip_path}")
            continue
        yield stem, csv_path, zip_path


def extract_member(archive, member, output_path, overwrite):
    output_path = Path(output_path)
    if output_path.is_file() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with archive.open(member) as source, open(temporary, "wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    os.replace(temporary, output_path)


def write_skeleton_video(
    archive,
    pose_members,
    output_path,
    width,
    height,
    frame_count,
    fps,
    overwrite,
):
    output_path = Path(output_path)
    if output_path.is_file() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp.avi")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"FFV1"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the FFV1 skeleton writer")
    try:
        black = np.zeros((height, width, 3), dtype=np.uint8)
        for frame_index in range(frame_count):
            member = pose_members.get(frame_index)
            if member is None:
                writer.write(black)
                continue
            with archive.open(member) as handle:
                image = Image.open(io.BytesIO(handle.read())).convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        writer.release()
        os.replace(temporary, output_path)
    except Exception:
        writer.release()
        temporary.unlink(missing_ok=True)
        raise


def convert_row(
    archive,
    names,
    row,
    output_root,
    final_output_root,
    num_frames,
    fps,
    overwrite,
    validate_only,
):
    sample_name = row["file_name"]
    video_member = f"{sample_name}/{sample_name}.mp4"
    if video_member not in names:
        raise FileNotFoundError(f"Archive has no {video_member}")

    image_names = parse_list(row["condition_images"], "condition_images", sample_name)
    condition_indices = parse_list(row["condition_index"], "condition_index", sample_name)
    if len(image_names) != len(condition_indices):
        raise ValueError(
            f"condition_images/index length mismatch for {sample_name}: "
            f"{len(image_names)} != {len(condition_indices)}"
        )
    pose_members = {}
    for image_name, frame_index in zip(image_names, condition_indices):
        member = f"{sample_name}/pose/{image_name}"
        if member in names:
            pose_members[int(frame_index)] = member
    if not pose_members:
        raise ValueError(f"No readable pose frames for {sample_name}")

    rgb_path = Path(output_root) / "rgb" / f"{sample_name}.mp4"
    skeleton_path = Path(output_root) / "skeleton" / f"{sample_name}.avi"
    published_root = Path(final_output_root or output_root)
    published_rgb_path = published_root / "rgb" / f"{sample_name}.mp4"
    published_skeleton_path = published_root / "skeleton" / f"{sample_name}.avi"
    if validate_only:
        return None

    extract_member(archive, video_member, rgb_path, overwrite)
    width, height, rgb_frame_count, source_fps = probe_video(rgb_path)
    aligned_frame_count = min(rgb_frame_count, max(pose_members) + 1)
    if aligned_frame_count <= 0:
        raise ValueError(f"No aligned frames for {sample_name}")
    write_skeleton_video(
        archive,
        pose_members,
        skeleton_path,
        width,
        height,
        aligned_frame_count,
        fps if fps > 0 else source_fps,
        overwrite,
    )
    skeleton_width, skeleton_height, skeleton_frame_count, _ = probe_video(skeleton_path)
    if (skeleton_width, skeleton_height) != (width, height):
        raise RuntimeError(f"Skeleton resolution mismatch for {sample_name}")
    usable_frames = min(aligned_frame_count, skeleton_frame_count)
    frame_indices = uniform_indices(usable_frames, num_frames)
    return {
        "type": "video",
        "file_path": str(published_rgb_path.resolve()),
        "video_path": str(published_rgb_path.resolve()),
        "text": row.get("text", ""),
        "control_type": "poseanything",
        "control_file_path": str(published_skeleton_path.resolve()),
        "skeleton_video_path": str(published_skeleton_path.resolve()),
        "video_sample_n_frames": num_frames,
        "video_sample_stride": 1,
        "sampling": {
            "mode": "explicit_frame_indices_xpose_aligned",
            "frame_indices": frame_indices,
            "num_frames": num_frames,
            "padding": "repeat_indices" if usable_frames < num_frames else "none",
        },
        "source": {"dataset": "Ryan241005/XPose", "sample_name": sample_name},
    }


def convert_batch(
    batch_name,
    csv_path,
    zip_path,
    output_root,
    final_output_root,
    num_frames,
    fps,
    overwrite,
    validate_only,
    max_samples=0,
    row_start=0,
    row_end=None,
    task_label=None,
):
    """Convert one archive in an isolated process and return structured results."""
    records = []
    failures = []
    attempts = 0
    with zipfile.ZipFile(zip_path) as archive, open(
        csv_path, newline="", encoding="utf-8"
    ) as handle:
        names = set(archive.namelist())
        for row_index, row in enumerate(csv.DictReader(handle)):
            if row_index < row_start:
                continue
            if row_end is not None and row_index >= row_end:
                break
            if max_samples > 0 and attempts >= max_samples:
                break
            attempts += 1
            try:
                record = convert_row(
                    archive,
                    names,
                    row,
                    output_root,
                    final_output_root,
                    num_frames,
                    fps,
                    overwrite,
                    validate_only,
                )
                if record is not None:
                    record["source"]["batch"] = batch_name
                    records.append(record)
                else:
                    records.append(
                        {"validated_sample": row["file_name"], "batch": batch_name}
                    )
            except Exception as exc:
                failures.append(
                    {
                        "batch": batch_name,
                        "sample": row.get("file_name"),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if attempts % 100 == 0:
                label = task_label or batch_name
                print(
                    f"progress {label}: attempts={attempts} "
                    f"records={len(records)} failures={len(failures)}",
                    flush=True,
                )
    return batch_name, row_start, records, failures, attempts


def convert_batch_worker(values):
    return convert_batch(*values)


def csv_record_count(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def allocate_batch_chunks(batch_counts, worker_count):
    """Assign at most worker_count chunks, weighted by batch row counts."""
    chunk_counts = [1] * len(batch_counts)
    for _ in range(max(worker_count - len(batch_counts), 0)):
        split_index = max(
            range(len(batch_counts)),
            key=lambda index: batch_counts[index] / chunk_counts[index],
        )
        chunk_counts[split_index] += 1
    return chunk_counts


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be positive; got {args.workers}")
    if args.max_samples > 0 and args.workers != 1:
        raise ValueError("--workers can be greater than 1 only with --max_samples 0")
    output_root = Path(args.output_root)
    final_output_root = Path(args.final_output_root or args.output_root)
    metadata_path = output_root / "metadata_poseanything.json"
    failures_path = output_root / "failed_samples.jsonl"
    records = []
    failures = []
    batch_specs = list(archive_batches(args.xpose_root, args.batches))
    requested_batches = [stem for stem, _, _ in batch_specs]
    completed_batches = []
    limit = None if args.max_samples <= 0 else args.max_samples
    attempts = 0

    if args.workers > 1:
        # Each worker owns its ZipFile; the parent is the sole metadata writer.
        batch_counts = [csv_record_count(csv_path) for _, csv_path, _ in batch_specs]
        chunk_counts = allocate_batch_chunks(batch_counts, args.workers)
        worker_args = []
        for (batch_name, csv_path, zip_path), batch_count, chunk_count in zip(
            batch_specs, batch_counts, chunk_counts
        ):
            chunk_size = (batch_count + chunk_count - 1) // chunk_count
            for chunk_index in range(chunk_count):
                row_start = chunk_index * chunk_size
                row_end = min(row_start + chunk_size, batch_count)
                if row_start >= row_end:
                    continue
                worker_args.append(
                    (
                        batch_name,
                        str(csv_path),
                        str(zip_path),
                        str(output_root),
                        str(final_output_root),
                        args.num_frames,
                        args.fps,
                        args.overwrite,
                        args.validate_only,
                        0,
                        row_start,
                        row_end,
                        f"{batch_name}[{chunk_index + 1}/{chunk_count}]",
                    )
                )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(args.workers, len(worker_args))
        ) as pool:
            results = pool.map(convert_batch_worker, worker_args)
            for (
                batch_name,
                row_start,
                batch_records,
                batch_failures,
                batch_attempts,
            ) in results:
                records.extend(batch_records)
                failures.extend(batch_failures)
                attempts += batch_attempts
                print(
                    f"chunk {batch_name}@{row_start}: records={len(batch_records)} "
                    f"failures={len(batch_failures)} attempts={batch_attempts}",
                    flush=True,
                )
        completed_batches.extend(requested_batches)
    else:
        for batch_name, csv_path, zip_path in batch_specs:
            remaining = 0 if limit is None else max(limit - attempts, 0)
            if limit is not None and remaining == 0:
                break
            result = convert_batch(
                batch_name,
                str(csv_path),
                str(zip_path),
                str(output_root),
                str(final_output_root),
                args.num_frames,
                args.fps,
                args.overwrite,
                args.validate_only,
                remaining,
            )
            _, _, batch_records, batch_failures, batch_attempts = result
            records.extend(batch_records)
            failures.extend(batch_failures)
            attempts += batch_attempts
            completed_batches.append(batch_name)
            print(
                f"batch {batch_name}: records={len(batch_records)} "
                f"failures={len(batch_failures)} attempts={batch_attempts}",
                flush=True,
            )

    if not records:
        raise RuntimeError("No XPose samples were converted or validated")
    if not args.validate_only:
        output_root.mkdir(parents=True, exist_ok=True)
        temporary = metadata_path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, metadata_path)
        with open(failures_path, "w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
        manifest = {
            "complete": args.max_samples == 0 and completed_batches == requested_batches,
            "max_samples": args.max_samples,
            "requested_batches": requested_batches,
            "completed_batches": completed_batches,
            "attempts": attempts,
            "records": len(records),
            "failures": len(failures),
            "metadata": str((final_output_root / metadata_path.name).resolve()),
        }
        manifest_path = output_root / "poseanything_full_manifest.json"
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        with open(manifest_tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        os.replace(manifest_tmp, manifest_path)
        print(f"metadata={metadata_path} samples={len(records)} failures={len(failures)}")
    else:
        print(f"validated={len(records)} failures={len(failures)}")


if __name__ == "__main__":
    main()
