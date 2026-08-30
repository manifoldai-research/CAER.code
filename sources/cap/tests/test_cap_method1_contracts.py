import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from videox_fun.training.cap_conditioning import (
    build_action_map_control_latents,
    build_poseanything_condition_latents,
    expand_patch_embedding_weight,
    pack_camera_condition,
)
from videox_fun.training.cap_gradient_audit import local_shard_max_abs
from videox_fun.training.method1_focused_loss import method1_focused_flow_loss
from videox_fun.training.realtime_metrics import append_jsonl, write_json_atomic
from videox_fun.training.sample_loss_recorder import Method1SampleLossRecorder
from videox_fun.training.sample_loss_recorder import padded_epoch_sample_count
from videox_fun.models.wan_camera_adapter import SimpleAdapter


class CAPConditioningContractTest(unittest.TestCase):
    def test_fsdp_local_shards_select_only_requested_input_channels(self):
        full_shape = (2, 5, 1, 1, 2)
        full_grad = torch.arange(20, dtype=torch.float32).reshape(full_shape)
        expected = full_grad[:, 3:5].abs().max()
        shard_sizes = (6, 5, 4, 5)
        offset = 0
        local_values = []
        for shard_size in shard_sizes:
            local_values.append(
                local_shard_max_abs(
                    full_grad.reshape(-1)[offset : offset + shard_size],
                    full_shape=full_shape,
                    shard_offset=offset,
                    channel_slice=slice(3, 5),
                )
            )
            offset += shard_size
        torch.testing.assert_close(torch.stack(local_values).max(), expected)

    def test_fsdp_local_shard_audit_rejects_invalid_coverage(self):
        with self.assertRaisesRegex(ValueError, "outside parameter"):
            local_shard_max_abs(
                torch.ones(3),
                full_shape=(2, 2),
                shard_offset=2,
            )

    def test_camera_adapter_zero_output_keeps_first_step_gradient(self):
        adapter = SimpleAdapter(
            in_dim=2,
            out_dim=4,
            kernel_size=1,
            stride=1,
            downscale_factor=2,
            zero_init_output=True,
        )
        camera = torch.randn(1, 2, 3, 4, 4)
        output = adapter(camera)
        torch.testing.assert_close(output, torch.zeros_like(output))
        output.sum().backward()
        self.assertIsNotNone(adapter.output_conv.weight.grad)
        self.assertGreater(float(adapter.output_conv.weight.grad.abs().max()), 0.0)
        self.assertTrue(bool(torch.isfinite(adapter.output_conv.weight.grad).all()))

    def test_patch_expansion_preserves_video_and_zeros_condition_channels(self):
        source = torch.arange(3 * 2 * 1 * 2 * 2, dtype=torch.float32).reshape(3, 2, 1, 2, 2)
        target = torch.empty(3, 5, 1, 2, 2)
        expanded = expand_patch_embedding_weight(source, target)
        torch.testing.assert_close(expanded[:, :2], source)
        self.assertEqual(float(expanded[:, 2:].abs().max()), 0.0)

    def test_action_map_replaces_only_future_video_control_channels(self):
        control = torch.zeros(2, 7, 3, 1, 1)
        control[:, :4] = 7
        control[:, 4:] = 2
        action = torch.full((2, 3, 3, 1, 1), 9.0)
        conditioned, null = build_action_map_control_latents(
            control, action, latent_channels=3, action_map_mask=torch.tensor([1, 0])
        )
        torch.testing.assert_close(conditioned[:, :4], control[:, :4])
        torch.testing.assert_close(conditioned[0, 4:, 0], control[0, 4:, 0])
        torch.testing.assert_close(conditioned[0, 4:, 1:], action[0, :, 1:])
        torch.testing.assert_close(conditioned[1], control[1])
        self.assertIs(null, control)

    def test_camera_pack_order_matches_vae_time_layout(self):
        camera = torch.zeros(1, 5, 6, 1, 1)
        for frame in range(5):
            for channel in range(6):
                camera[0, frame, channel, 0, 0] = frame * 10 + channel
        packed = pack_camera_condition(camera)
        self.assertEqual(tuple(packed.shape), (1, 24, 2, 1, 1))
        expected_first = torch.tensor(
            [channel * 0 + channel for channel in range(6) for _ in range(4)],
            dtype=packed.dtype,
        )
        expected_second = torch.tensor(
            [frame * 10 + channel for channel in range(6) for frame in range(1, 5)],
            dtype=packed.dtype,
        )
        torch.testing.assert_close(packed[0, :, 0, 0, 0], expected_first)
        torch.testing.assert_close(packed[0, :, 1, 0, 0], expected_second)

    def test_poseanything_uses_black_vae_latent_as_null(self):
        video = torch.zeros(2, 3, 2, 1, 1)
        skeleton = torch.full_like(video, 4)
        black = torch.full_like(video, -3)
        conditioned, null = build_poseanything_condition_latents(
            video, skeleton, black, skeleton_mask=torch.tensor([1, 0])
        )
        torch.testing.assert_close(conditioned[0], skeleton[0])
        torch.testing.assert_close(conditioned[1], black[1])
        torch.testing.assert_close(null, black)

    def test_method1_loss_is_finite_and_backpropagates_only_prediction(self):
        prediction = torch.randn(2, 3, 4, 2, 2, requires_grad=True)
        target = torch.randn_like(prediction)
        effect = torch.rand(2, 1, 4, 2, 2, requires_grad=True)
        loss, rho, uniform, per_sample, per_sample_uniform = method1_focused_flow_loss(
            prediction,
            target,
            effect,
            active_mask=torch.tensor([1, 0]).view(2, 1, 1, 1, 1),
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(bool(torch.isfinite(rho).all()))
        self.assertEqual(tuple(per_sample.shape), (2,))
        self.assertEqual(tuple(per_sample_uniform.shape), (2,))
        torch.testing.assert_close(uniform, per_sample_uniform.mean())
        torch.testing.assert_close(rho[1], torch.ones_like(rho[1]))
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(effect.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertTrue(bool(torch.isfinite(uniform)))

    def test_method1_perfect_prediction_uses_uniform_weights(self):
        for variant in ("uniform", "e_only", "s_only", "s_max1", "current"):
            prediction = torch.zeros(2, 3, 4, 2, 2, requires_grad=True)
            target = torch.zeros_like(prediction)
            effect = torch.zeros(2, 1, 4, 2, 2)
            loss, rho, uniform, per_sample, per_sample_uniform = method1_focused_flow_loss(
                prediction,
                target,
                effect,
                active_mask=torch.tensor([1, 0]).view(2, 1, 1, 1, 1),
                loss_variant=variant,
            )
            torch.testing.assert_close(rho, torch.ones_like(rho))
            torch.testing.assert_close(loss, torch.zeros_like(loss))
            torch.testing.assert_close(uniform, torch.zeros_like(uniform))
            torch.testing.assert_close(per_sample, torch.zeros_like(per_sample))
            torch.testing.assert_close(
                per_sample_uniform, torch.zeros_like(per_sample_uniform)
            )
            loss.backward()
            torch.testing.assert_close(
                prediction.grad, torch.zeros_like(prediction.grad)
            )

    def test_method1_ablation_variants_match_exact_formulas(self):
        prediction = torch.tensor(
            [[[[[7.0]]], [[[1.0]]], [[[3.0]]]]]
        ).permute(0, 2, 1, 3, 4)
        target = torch.zeros_like(prediction)
        effect = torch.tensor([[[[[9.0]], [[2.0]], [[4.0]]]]])
        expected_rho = {
            "uniform": torch.tensor([1.0, 1.0]),
            "e_only": torch.tensor([0.5, 1.5]),
            "s_only": torch.tensor([2.0 / 3.0, 4.0 / 3.0]),
            "s_max1": torch.tensor([1.0, 4.0 / 3.0]),
            "current": torch.tensor([0.4, 1.6]),
        }
        expected_loss = {
            "uniform": 5.0,
            "e_only": 7.0,
            "s_only": 19.0 / 3.0,
            "s_max1": 39.0 / 7.0,
            "current": 7.4,
        }
        for variant, expected in expected_rho.items():
            loss, rho, uniform, _, _ = method1_focused_flow_loss(
                prediction,
                target,
                effect if variant in {"s_only", "s_max1", "current"} else None,
                loss_variant=variant,
            )
            torch.testing.assert_close(rho.flatten(), expected)
            torch.testing.assert_close(loss, torch.tensor(expected_loss[variant]))
            torch.testing.assert_close(uniform, torch.tensor(5.0))

    def test_effect_map_is_required_only_for_s_variants(self):
        prediction = torch.randn(1, 2, 3, 1, 1)
        target = torch.randn_like(prediction)
        for variant in ("uniform", "e_only"):
            method1_focused_flow_loss(
                prediction, target, None, loss_variant=variant
            )
        for variant in ("s_only", "s_max1", "current"):
            with self.assertRaisesRegex(ValueError, "effect_map is required"):
                method1_focused_flow_loss(
                    prediction, target, None, loss_variant=variant
                )

    def test_method1_returns_per_sample_uniform_losses(self):
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


class _MainProcessAccelerator:
    is_main_process = True


class CAPLossRecordingContractTest(unittest.TestCase):
    def test_epoch_padding_covers_all_metadata_with_full_effective_batches(self):
        scheduled = padded_epoch_sample_count(365831, 8, 4, 1)
        self.assertEqual(scheduled, 365856)
        self.assertEqual(scheduled % 32, 0)

        generator = torch.Generator().manual_seed(42)
        sampled = list(
            torch.utils.data.RandomSampler(
                range(13),
                replacement=False,
                num_samples=padded_epoch_sample_count(13, 2, 4, 1),
                generator=generator,
            )
        )
        self.assertEqual(len(sampled), 16)
        self.assertEqual(set(sampled), set(range(13)))

        metadata = [{"file_path": f"sample-{index}.mp4"} for index in range(13)]
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = Method1SampleLossRecorder(
                temp_dir,
                metadata,
                _MainProcessAccelerator(),
            )
            recorder.start_epoch(0)
            records = torch.tensor(
                [
                    [float(index), float(index + 1), float(index + 2), 1.0]
                    for index in sampled
                ],
                dtype=torch.float64,
            )
            recorder.record_gathered(0, 0, 0, records)
            summary = recorder.finalize_epoch(0, complete=True, optimizer_step_after=2)
            self.assertEqual(summary["observations"], 16)
            self.assertEqual(summary["unique_samples"], 13)
            self.assertEqual(summary["missing_metadata_candidates"], 0)
            self.assertTrue(summary["complete_metadata_coverage"])

    def test_cap_launcher_uses_validated_runtime_and_formal_defaults(self):
        repo_root = Path(__file__).parents[1]
        launcher = (
            repo_root / "scripts/wan2.2_fun/run_cap_ablation_volc.sh"
        ).read_text(encoding="utf-8")
        train_launcher = (
            repo_root / "scripts/wan2.2_fun/run_cap_train.sh"
        ).read_text(encoding="utf-8")
        training_script = (
            repo_root / "scripts/wan2.2_fun/train_control_camera_arm_actionmap_method1.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("/dev/shm/cap-runtime/bin/python", launcher)
        self.assertNotIn("/dev/shm/cap-runtime/bin/python", train_launcher)
        self.assertIn('CAP_STAGE_LOCAL_SITE=1', launcher)
        self.assertIn('export DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"', launcher)
        self.assertIn('export LOW_VRAM="${LOW_VRAM:-1}"', launcher)
        self.assertIn('LOW_VRAM="${LOW_VRAM:-1}"', train_launcher)
        self.assertIn('uniform|e_only|s_only|s_max1|current', launcher)
        self.assertIn('uniform|e_only|s_only|s_max1|current', train_launcher)
        self.assertIn('"s_max1",', training_script)
        self.assertIn('CONTROLLED_ABLATION_VARIABLES=(', launcher)
        self.assertIn('unset "$variable_name"', launcher)
        self.assertIn('CAP_ABLATION_RESUME_CHECKPOINT', launcher)
        self.assertIn('CAP_ABLATION_TRAIN_SAMPLES', launcher)
        self.assertIn('export MAX_TRAIN_SAMPLES="$CAP_ALIGNED_TRAIN_SAMPLES"', launcher)
        self.assertIn('(CAP_REQUESTED_TRAIN_SAMPLES + 31) / 32 * 32', launcher)
        self.assertIn('--resume_with_new_dataset', launcher)
        self.assertIn('"--resume_with_new_dataset"', training_script)
        self.assertIn('CAP new-dataset resume audit:', training_script)
        self.assertIn('expected_target_step = (', training_script)
        self.assertIn('New-dataset resume target mismatch:', training_script)
        self.assertIn('export RESUME_FROM_CHECKPOINT', launcher)
        self.assertIn('CAP resume:', launcher)
        self.assertIn('RESUMING" != "1"', launcher)
        self.assertIn('export CONTROL_MODEL="${CONTROL_MODEL:?', launcher)
        self.assertNotIn("/ML-vePFS/", launcher)
        self.assertNotIn("/manifold-obs/", launcher)
        self.assertIn("Complete Method1 sample-loss epoch missed metadata samples", training_script)
        self.assertIn(
            "first_epoch = global_step // num_update_steps_per_epoch",
            training_script,
        )

        self.assertIn(
            "camera)\n        DEFAULT_HEIGHT=704\n        DEFAULT_WIDTH=1280",
            train_launcher,
        )
        self.assertIn("export HEIGHT=704\n    export WIDTH=1280", launcher)
        self.assertNotIn("DEFAULT_WIDTH=704", train_launcher)
        self.assertNotIn("export WIDTH=704", launcher)

    def test_cap_launcher_isolates_mutable_runtime_paths_per_run(self):
        launcher = (
            Path(__file__).parents[1]
            / "scripts/wan2.2_fun/run_cap_ablation_volc.sh"
        ).read_text(encoding="utf-8")
        run_scope = "${UID}-${MODALITY}-${LOSS_VARIANT}-${CAP_ABLATION_RUN_ID}"
        self.assertIn(run_scope, launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/pycache', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/tmp', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/triton', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/torch_extensions', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/torchinductor', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/cuda', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/numba', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/external_cache', launcher)
        self.assertIn('$CAP_NODE_CACHE_ROOT/per_rank', launcher)
        self.assertIn('export PYTHONDONTWRITEBYTECODE=1', launcher)
        self.assertIn('VIDEOX_METHOD1_ISOLATE_RUNTIME_CACHE=1', launcher)
        self.assertIn('$MODALITY/$LOSS_VARIANT/$CAP_ABLATION_RUN_ID', launcher)

    def test_cap_training_keeps_distributed_epoch_remainder(self):
        training_script = (
            Path(__file__).parents[1]
            / "scripts/wan2.2_fun/train_control_camera_arm_actionmap_method1.py"
        ).read_text(encoding="utf-8")
        self.assertIn("drop_last=False", training_script)
        self.assertNotIn("class Method1SampleLossRecorder:", training_script)

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


if __name__ == "__main__":
    unittest.main()
