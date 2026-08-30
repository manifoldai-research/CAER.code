import csv
import json
import math
import os


def padded_epoch_sample_count(num_samples, num_processes, gradient_accumulation_steps, batch_size):
    values = {
        "num_samples": num_samples,
        "num_processes": num_processes,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "batch_size": batch_size,
    }
    for name, value in values.items():
        if int(value) != value or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    multiple = int(num_processes) * int(gradient_accumulation_steps) * int(batch_size)
    return ((int(num_samples) + multiple - 1) // multiple) * multiple


class Method1SampleLossRecorder:
    CSV_FIELDS = (
        "epoch",
        "metadata_index",
        "sample_id",
        "episode_id",
        "task",
        "start_frame",
        "file_path",
        "weighted_loss_rank_desc",
        "uniform_loss_rank_desc",
        "weighted_loss_mean",
        "weighted_loss_min",
        "weighted_loss_max",
        "uniform_loss_mean",
        "uniform_loss_min",
        "uniform_loss_max",
        "observations",
        "action_conditioned_observations",
        "first_optimizer_step_before",
        "last_optimizer_step_before",
    )

    def __init__(self, output_dir, metadata, accelerator, flush_every=64):
        self.output_dir = os.path.abspath(output_dir)
        self.metadata = metadata
        self.accelerator = accelerator
        self.flush_every = max(int(flush_every), 1)
        self.current_epoch = None
        self.aggregates = {}
        self.raw_path = None
        self.pending_lines = []
        self.previous_epoch = None
        self.previous_losses = None
        if accelerator.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)

    def load_previous_epoch(self, epoch):
        if not self.accelerator.is_main_process:
            return None
        epoch = int(epoch)
        if epoch < 0:
            self.previous_epoch = None
            self.previous_losses = None
            return None
        path = os.path.join(
            self.output_dir,
            f"epoch_{epoch + 1:03d}_sample_losses.csv",
        )
        if not os.path.isfile(path):
            self.previous_epoch = None
            self.previous_losses = None
            return None

        losses = {}
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required_fields = {
                "metadata_index",
                "weighted_loss_mean",
                "uniform_loss_mean",
            }
            if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
                raise RuntimeError(
                    f"Previous Method1 sample-loss CSV has invalid columns: {path}"
                )
            for line_number, row in enumerate(reader, 2):
                if (
                    row["weighted_loss_mean"] == "N/A"
                    or row["uniform_loss_mean"] == "N/A"
                ):
                    continue
                try:
                    metadata_index = int(row["metadata_index"])
                    weighted_loss = float(row["weighted_loss_mean"])
                    uniform_loss = float(row["uniform_loss_mean"])
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Malformed previous Method1 sample loss at {path}:{line_number}"
                    ) from exc
                if not math.isfinite(weighted_loss) or not math.isfinite(uniform_loss):
                    raise RuntimeError(
                        f"Non-finite previous Method1 sample loss at {path}:{line_number}"
                    )
                losses[metadata_index] = {
                    "weighted": weighted_loss,
                    "uniform": uniform_loss,
                }

        self.previous_epoch = epoch
        self.previous_losses = losses
        return path

    def _metadata_row(self, metadata_index):
        item = self.metadata[metadata_index] if 0 <= metadata_index < len(self.metadata) else {}
        return {
            "metadata_index": metadata_index,
            "sample_id": str(metadata_index),
            "episode_id": item.get("episode_id", item.get("episode", "")),
            "task": item.get("task", ""),
            "start_frame": item.get("start_frame", ""),
            "file_path": item.get("file_path", ""),
        }

    @staticmethod
    def _atomic_csv(path, fieldnames, rows):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_json(path, value):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)

    @staticmethod
    def _display(value):
        return "N/A" if value is None else value

    def _flush_visits(self):
        if not self.accelerator.is_main_process or not self.pending_lines:
            return
        with open(self.raw_path, "a", encoding="utf-8") as handle:
            handle.writelines(self.pending_lines)
        self.pending_lines.clear()

    def _accumulate(
        self,
        metadata_index,
        weighted_loss,
        uniform_loss,
        action_conditioned,
        optimizer_step_before,
    ):
        aggregate = self.aggregates.get(metadata_index)
        if aggregate is None:
            self.aggregates[metadata_index] = {
                "weighted_sum": weighted_loss,
                "weighted_min": weighted_loss,
                "weighted_max": weighted_loss,
                "uniform_sum": uniform_loss,
                "uniform_min": uniform_loss,
                "uniform_max": uniform_loss,
                "count": 1,
                "action_count": action_conditioned,
                "first_step": optimizer_step_before,
                "last_step": optimizer_step_before,
            }
            return
        aggregate["weighted_sum"] += weighted_loss
        aggregate["weighted_min"] = min(aggregate["weighted_min"], weighted_loss)
        aggregate["weighted_max"] = max(aggregate["weighted_max"], weighted_loss)
        aggregate["uniform_sum"] += uniform_loss
        aggregate["uniform_min"] = min(aggregate["uniform_min"], uniform_loss)
        aggregate["uniform_max"] = max(aggregate["uniform_max"], uniform_loss)
        aggregate["count"] += 1
        aggregate["action_count"] += action_conditioned
        aggregate["last_step"] = optimizer_step_before

    def start_epoch(self, epoch, resume_optimizer_step=None):
        if not self.accelerator.is_main_process:
            return None
        if self.current_epoch is not None:
            raise RuntimeError(
                "Method1 sample-loss recorder started a new epoch before finalizing the previous one."
            )
        self.current_epoch = int(epoch)
        self.aggregates = {}
        self.pending_lines = []
        self.raw_path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}_visits.jsonl")
        if resume_optimizer_step is None:
            with open(self.raw_path, "w", encoding="utf-8"):
                pass
            return 0

        resume_optimizer_step = int(resume_optimizer_step)
        if resume_optimizer_step < 0:
            raise ValueError(
                "resume_optimizer_step must be non-negative; "
                f"got {resume_optimizer_step}"
            )

        retained_records = []
        if os.path.isfile(self.raw_path):
            with open(self.raw_path, encoding="utf-8") as handle:
                lines = handle.readlines()
            nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
            final_nonempty_index = nonempty_indices[-1] if nonempty_indices else -1
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    if index == final_nonempty_index:
                        break
                    raise RuntimeError(
                        f"Invalid Method1 visit record at {self.raw_path}:{index + 1}"
                    ) from exc
                try:
                    record_epoch = int(record["epoch"])
                    metadata_index = int(record["metadata_index"])
                    weighted_loss = float(record["weighted_loss"])
                    uniform_loss = float(record["uniform_loss"])
                    action_conditioned = int(record["action_conditioned"])
                    optimizer_step_before = int(record["optimizer_step_before"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Malformed Method1 visit record at {self.raw_path}:{index + 1}"
                    ) from exc
                if record_epoch != int(epoch) + 1:
                    raise RuntimeError(
                        f"Method1 visit epoch mismatch at {self.raw_path}:{index + 1}: "
                        f"expected {int(epoch) + 1}, got {record_epoch}"
                    )
                if not math.isfinite(weighted_loss) or not math.isfinite(uniform_loss):
                    raise RuntimeError(
                        f"Non-finite Method1 visit record at {self.raw_path}:{index + 1}"
                    )
                if optimizer_step_before >= resume_optimizer_step:
                    continue
                retained_records.append(record)
                self._accumulate(
                    metadata_index,
                    weighted_loss,
                    uniform_loss,
                    action_conditioned,
                    optimizer_step_before,
                )

        tmp_path = self.raw_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for record in retained_records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        os.replace(tmp_path, self.raw_path)
        return len(retained_records)

    def record_gathered(self, epoch, dataloader_step, optimizer_step_before, gathered):
        if not self.accelerator.is_main_process:
            return
        if self.raw_path is None or self.current_epoch != int(epoch):
            raise RuntimeError("Method1 sample-loss recorder received a sample outside an active epoch.")

        for metadata_index_f, weighted_loss, uniform_loss, action_conditioned_f in gathered.detach().cpu().tolist():
            metadata_index = int(metadata_index_f)
            weighted_loss = float(weighted_loss)
            uniform_loss = float(uniform_loss)
            action_conditioned = int(action_conditioned_f >= 0.5)
            if not math.isfinite(weighted_loss) or not math.isfinite(uniform_loss):
                raise RuntimeError(
                    "Non-finite Method1 sample loss for metadata index "
                    f"{metadata_index}: weighted={weighted_loss} uniform={uniform_loss}"
                )

            self.pending_lines.append(
                json.dumps(
                    {
                        "epoch": int(epoch) + 1,
                        "metadata_index": metadata_index,
                        "sample_id": str(metadata_index),
                        "weighted_loss": weighted_loss,
                        "uniform_loss": uniform_loss,
                        "action_conditioned": action_conditioned,
                        "dataloader_step": int(dataloader_step),
                        "optimizer_step_before": int(optimizer_step_before),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )

            self._accumulate(
                metadata_index,
                weighted_loss,
                uniform_loss,
                action_conditioned,
                int(optimizer_step_before),
            )

        if len(self.pending_lines) >= self.flush_every:
            self._flush_visits()

    def finalize_epoch(self, epoch, complete, optimizer_step_after):
        if not self.accelerator.is_main_process:
            return None
        if self.raw_path is None or self.current_epoch != int(epoch):
            raise RuntimeError("Method1 sample-loss recorder cannot finalize an inactive epoch.")
        self._flush_visits()

        losses = {
            metadata_index: {
                "weighted": values["weighted_sum"] / values["count"],
                "uniform": values["uniform_sum"] / values["count"],
            }
            for metadata_index, values in self.aggregates.items()
        }
        weighted_order = sorted(losses, key=lambda index: (-losses[index]["weighted"], index))
        uniform_order = sorted(losses, key=lambda index: (-losses[index]["uniform"], index))
        weighted_ranks = {metadata_index: rank for rank, metadata_index in enumerate(weighted_order, 1)}
        uniform_ranks = {metadata_index: rank for rank, metadata_index in enumerate(uniform_order, 1)}

        epoch_rows = []
        for metadata_index in range(len(self.metadata)):
            aggregate = self.aggregates.get(metadata_index)
            row = {
                "epoch": int(epoch) + 1,
                **self._metadata_row(metadata_index),
            }
            if aggregate is None:
                row.update(
                    {
                        "weighted_loss_rank_desc": "N/A",
                        "uniform_loss_rank_desc": "N/A",
                        "weighted_loss_mean": "N/A",
                        "weighted_loss_min": "N/A",
                        "weighted_loss_max": "N/A",
                        "uniform_loss_mean": "N/A",
                        "uniform_loss_min": "N/A",
                        "uniform_loss_max": "N/A",
                        "observations": 0,
                        "action_conditioned_observations": 0,
                        "first_optimizer_step_before": "N/A",
                        "last_optimizer_step_before": "N/A",
                    }
                )
            else:
                row.update(
                    {
                        "weighted_loss_rank_desc": weighted_ranks[metadata_index],
                        "uniform_loss_rank_desc": uniform_ranks[metadata_index],
                        "weighted_loss_mean": f"{losses[metadata_index]['weighted']:.17g}",
                        "weighted_loss_min": f"{aggregate['weighted_min']:.17g}",
                        "weighted_loss_max": f"{aggregate['weighted_max']:.17g}",
                        "uniform_loss_mean": f"{losses[metadata_index]['uniform']:.17g}",
                        "uniform_loss_min": f"{aggregate['uniform_min']:.17g}",
                        "uniform_loss_max": f"{aggregate['uniform_max']:.17g}",
                        "observations": aggregate["count"],
                        "action_conditioned_observations": aggregate["action_count"],
                        "first_optimizer_step_before": aggregate["first_step"],
                        "last_optimizer_step_before": aggregate["last_step"],
                    }
                )
            epoch_rows.append(row)

        epoch_path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}_sample_losses.csv")
        self._atomic_csv(epoch_path, self.CSV_FIELDS, epoch_rows)

        comparison_path = None
        if self.previous_losses is not None:
            previous_epoch = int(self.previous_epoch)
            comparison_path = os.path.join(
                self.output_dir,
                f"epoch_{previous_epoch + 1:03d}_to_{epoch + 1:03d}_loss_changes.csv",
            )
            comparison_fields = (
                "metadata_index",
                "sample_id",
                "episode_id",
                "task",
                "start_frame",
                "file_path",
                "previous_epoch",
                "current_epoch",
                "previous_weighted_loss",
                "current_weighted_loss",
                "weighted_loss_drop",
                "previous_uniform_loss",
                "current_uniform_loss",
                "uniform_loss_drop",
            )
            comparison_rows = []
            for metadata_index in range(len(self.metadata)):
                previous = self.previous_losses.get(metadata_index)
                current = losses.get(metadata_index)
                weighted_drop = None if previous is None or current is None else previous["weighted"] - current["weighted"]
                uniform_drop = None if previous is None or current is None else previous["uniform"] - current["uniform"]
                metadata_row = self._metadata_row(metadata_index)
                comparison_rows.append(
                    {
                        **metadata_row,
                        "previous_epoch": previous_epoch + 1,
                        "current_epoch": int(epoch) + 1,
                        "previous_weighted_loss": self._display(None if previous is None else f"{previous['weighted']:.17g}"),
                        "current_weighted_loss": self._display(None if current is None else f"{current['weighted']:.17g}"),
                        "weighted_loss_drop": self._display(None if weighted_drop is None else f"{weighted_drop:.17g}"),
                        "previous_uniform_loss": self._display(None if previous is None else f"{previous['uniform']:.17g}"),
                        "current_uniform_loss": self._display(None if current is None else f"{current['uniform']:.17g}"),
                        "uniform_loss_drop": self._display(None if uniform_drop is None else f"{uniform_drop:.17g}"),
                    }
                )
            self._atomic_csv(comparison_path, comparison_fields, comparison_rows)

        summary = {
            "epoch": int(epoch) + 1,
            "complete": bool(complete),
            "optimizer_step_after": int(optimizer_step_after),
            "observations": sum(values["count"] for values in self.aggregates.values()),
            "unique_samples": len(losses),
            "missing_metadata_candidates": max(len(self.metadata) - len(losses), 0),
            "complete_metadata_coverage": len(losses) == len(self.metadata),
            "sample_losses_csv": epoch_path,
            "visits_jsonl": self.raw_path,
            "comparison_csv": comparison_path,
            "weighted_loss_definition": "per-sample Method1 focused loss used for backward",
            "uniform_loss_definition": "per-sample unweighted MSE over the same future latent tokens",
        }
        summary_path = os.path.join(self.output_dir, f"epoch_{epoch + 1:03d}_summary.json")
        self._atomic_json(summary_path, summary)

        self.previous_epoch = int(epoch)
        self.previous_losses = losses
        self.current_epoch = None
        self.aggregates = {}
        self.raw_path = None
        self.pending_lines = []
        return summary
