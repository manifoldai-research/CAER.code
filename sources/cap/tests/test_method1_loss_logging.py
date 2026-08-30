import csv
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

from videox_fun.training.method1_focused_loss import method1_focused_flow_loss
from videox_fun.training.realtime_metrics import (
    append_jsonl,
    prepare_step_metrics_jsonl,
    write_json_atomic,
)
from videox_fun.training.sample_loss_recorder import Method1SampleLossRecorder


class _MainProcessAccelerator:
    is_main_process = True


class Method1LossLoggingTest(unittest.TestCase):
    def test_realtime_tee_updates_log_before_producer_exits(self):
        repo_root = Path(__file__).parents[1]
        tee_script = repo_root / "scripts/wan2.2_fun/realtime_tee.py"
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "console.log"
            logger = subprocess.Popen(
                [
                    sys.executable,
                    str(tee_script),
                    str(log_path),
                    "--flush-seconds",
                    "0.1",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
            )
            try:
                logger.stdin.write(b"weighted=1.25 uniform=1.00\r")
                logger.stdin.flush()
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if log_path.exists() and "weighted=1.25" in log_path.read_text():
                        break
                    if logger.poll() is not None:
                        self.fail(f"realtime tee exited early with code {logger.returncode}")
                    time.sleep(0.05)
                self.assertIsNone(logger.poll())
                self.assertTrue(log_path.exists())
                self.assertIn("weighted=1.25", log_path.read_text())

                logger.stdin.write(b"weighted=1.10 uniform=0.95\n")
                logger.stdin.close()
                self.assertEqual(logger.wait(timeout=5), 0)
            finally:
                if logger.poll() is None:
                    logger.kill()
                    logger.wait(timeout=5)
            final_log = log_path.read_text()
            self.assertIn("weighted=1.10", final_log)
            self.assertIn("uniform=0.95", final_log)

    def test_loss_returns_per_sample_uniform_values(self):
        prediction = torch.tensor(
            [
                [[[[0.0]], [[1.0]], [[2.0]]]],
                [[[[0.0]], [[3.0]], [[5.0]]]],
            ]
        )
        target = torch.zeros_like(prediction)
        weighted, _, uniform, per_sample_weighted, per_sample_uniform = (
            method1_focused_flow_loss(
                prediction,
                target,
                None,
                loss_variant="uniform",
            )
        )
        expected = torch.tensor([2.5, 17.0])
        torch.testing.assert_close(per_sample_uniform, expected)
        torch.testing.assert_close(per_sample_weighted, expected)
        torch.testing.assert_close(uniform, expected.mean())
        torch.testing.assert_close(weighted, expected.mean())

    def test_sample_recorder_writes_dual_loss_visits_and_epoch_csv(self):
        metadata = [
            {"episode_id": "ep-a", "task": "pick", "start_frame": 3, "file_path": "a.mp4"},
            {"episode_id": "ep-b", "task": "place", "start_frame": 7, "file_path": "b.mp4"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Method1SampleLossRecorder(
                temp_dir,
                metadata,
                _MainProcessAccelerator(),
                flush_every=1,
            )
            recorder.start_epoch(0)
            recorder.record_gathered(
                epoch=0,
                dataloader_step=4,
                optimizer_step_before=1,
                gathered=torch.tensor(
                    [
                        [1.0, 9.0, 4.0, 1.0],
                        [0.0, 2.0, 3.0, 0.0],
                        [1.0, 5.0, 2.0, 1.0],
                    ],
                    dtype=torch.float64,
                ),
            )

            visits_path = Path(temp_dir) / "epoch_001_visits.jsonl"
            visits = [json.loads(line) for line in visits_path.read_text().splitlines()]
            self.assertEqual(len(visits), 3)
            self.assertEqual(visits[0]["metadata_index"], 1)
            self.assertEqual(visits[0]["weighted_loss"], 9.0)
            self.assertEqual(visits[0]["uniform_loss"], 4.0)

            summary = recorder.finalize_epoch(0, complete=True, optimizer_step_after=2)
            self.assertEqual(summary["observations"], 3)
            self.assertEqual(summary["unique_samples"], 2)
            with open(summary["sample_losses_csv"], newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["metadata_index"] for row in rows], ["0", "1"])
            self.assertEqual(float(rows[0]["weighted_loss_mean"]), 2.0)
            self.assertEqual(float(rows[0]["uniform_loss_mean"]), 3.0)
            self.assertEqual(float(rows[1]["weighted_loss_mean"]), 7.0)
            self.assertEqual(float(rows[1]["uniform_loss_mean"]), 3.0)
            self.assertEqual(rows[1]["observations"], "2")
            self.assertEqual(rows[1]["action_conditioned_observations"], "2")

    def test_realtime_metrics_close_files_after_each_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "train_metrics.jsonl"
            latest_path = Path(temp_dir) / "latest_metrics.json"
            first = {"global_step": 1, "method1_weighted_loss": 2.0, "method1_uniform_loss": 3.0}
            second = {"global_step": 2, "method1_weighted_loss": 1.0, "method1_uniform_loss": 2.0}
            append_jsonl(jsonl_path, first)
            write_json_atomic(latest_path, first)
            append_jsonl(jsonl_path, second)
            write_json_atomic(latest_path, second)
            records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
            self.assertEqual(records, [first, second])
            self.assertEqual(json.loads(latest_path.read_text()), second)

    def test_realtime_metrics_resume_keeps_checkpoint_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            jsonl_path = Path(temp_dir) / "train_metrics.jsonl"
            append_jsonl(jsonl_path, {"global_step": 1, "loss": 3.0})
            append_jsonl(jsonl_path, {"global_step": 2, "loss": 2.0})
            append_jsonl(jsonl_path, {"global_step": 3, "loss": 1.0})

            latest = prepare_step_metrics_jsonl(jsonl_path, resume_global_step=2)

            self.assertEqual(latest, {"global_step": 2, "loss": 2.0})
            records = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
            self.assertEqual(
                records,
                [
                    {"global_step": 1, "loss": 3.0},
                    {"global_step": 2, "loss": 2.0},
                ],
            )

    def test_sample_recorder_resume_keeps_only_precheckpoint_visits(self):
        metadata = [{"file_path": "a.mp4"}, {"file_path": "b.mp4"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Method1SampleLossRecorder(
                temp_dir,
                metadata,
                _MainProcessAccelerator(),
                flush_every=1,
            )
            first.start_epoch(0)
            first.record_gathered(
                epoch=0,
                dataloader_step=4,
                optimizer_step_before=4,
                gathered=torch.tensor([[0.0, 8.0, 4.0, 1.0]], dtype=torch.float64),
            )
            first.record_gathered(
                epoch=0,
                dataloader_step=5,
                optimizer_step_before=5,
                gathered=torch.tensor([[1.0, 99.0, 99.0, 0.0]], dtype=torch.float64),
            )

            resumed = Method1SampleLossRecorder(
                temp_dir,
                metadata,
                _MainProcessAccelerator(),
                flush_every=1,
            )
            retained = resumed.start_epoch(0, resume_optimizer_step=5)
            self.assertEqual(retained, 1)
            resumed.record_gathered(
                epoch=0,
                dataloader_step=5,
                optimizer_step_before=5,
                gathered=torch.tensor([[1.0, 6.0, 3.0, 1.0]], dtype=torch.float64),
            )
            summary = resumed.finalize_epoch(0, complete=True, optimizer_step_after=6)

            visits = [
                json.loads(line)
                for line in (Path(temp_dir) / "epoch_001_visits.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual([visit["weighted_loss"] for visit in visits], [8.0, 6.0])
            self.assertEqual(summary["observations"], 2)
            self.assertEqual(summary["unique_samples"], 2)

    def test_sample_recorder_resume_loads_previous_epoch_for_comparison(self):
        metadata = [{"file_path": "a.mp4"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Method1SampleLossRecorder(
                temp_dir,
                metadata,
                _MainProcessAccelerator(),
                flush_every=1,
            )
            first.start_epoch(0)
            first.record_gathered(
                epoch=0,
                dataloader_step=0,
                optimizer_step_before=0,
                gathered=torch.tensor([[0.0, 8.0, 4.0, 1.0]], dtype=torch.float64),
            )
            first.finalize_epoch(0, complete=True, optimizer_step_after=1)

            resumed = Method1SampleLossRecorder(
                temp_dir,
                metadata,
                _MainProcessAccelerator(),
                flush_every=1,
            )
            previous_path = resumed.load_previous_epoch(0)
            self.assertTrue(Path(previous_path).is_file())
            resumed.start_epoch(1, resume_optimizer_step=1)
            resumed.record_gathered(
                epoch=1,
                dataloader_step=0,
                optimizer_step_before=1,
                gathered=torch.tensor([[0.0, 6.0, 3.0, 1.0]], dtype=torch.float64),
            )
            summary = resumed.finalize_epoch(1, complete=True, optimizer_step_after=2)

            self.assertTrue(Path(summary["comparison_csv"]).is_file())
            with open(summary["comparison_csv"], newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(float(row["weighted_loss_drop"]), 2.0)
            self.assertEqual(float(row["uniform_loss_drop"]), 1.0)


if __name__ == "__main__":
    unittest.main()
