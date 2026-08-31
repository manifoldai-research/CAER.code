import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


SCRIPT_DIR = Path(__file__).parents[1] / "scripts" / "wan2.2_fun"
sys.path.insert(0, str(SCRIPT_DIR))

import arm_mse_heatmap as weight_viz  # noqa: E402


def load_visualization_module():
    single = types.ModuleType("infer_cap_arm_sample")
    single.VARIANTS = ("CAER",)
    batch = types.ModuleType("infer_cap_arm_worldarena_batch")
    with mock.patch.dict(
        sys.modules,
        {
            "infer_cap_arm_sample": single,
            "infer_cap_arm_worldarena_batch": batch,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "visualize_cap_arm_weights_test",
            SCRIPT_DIR / "visualize_cap_arm_weights.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class WeightVisualizationRenderingTest(unittest.TestCase):
    def test_episode_min_and_latent_positive_p99_are_shared_by_all_frames(self):
        latent = np.asarray(
            [
                [[999.0, 999.0]],  # fixed first frame must not affect vmax
                [[1.0, 2.0]],
                [[3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        vmax = weight_viz.positive_percentile_vmax(latent, percentile=99.0)
        self.assertAlmostEqual(vmax, float(np.percentile([1, 2, 3, 4], 99.0)))

        upsampled = np.asarray(
            [
                [[0.0, 0.5], [1.0, 1.5]],
                [[1.0, 2.0], [3.0, 4.0]],
            ],
            dtype=np.float32,
        )
        vmin = weight_viz.episode_response_vmin(upsampled, vmax)
        response = weight_viz.normalize_weight_response(
            upsampled, vmax, vmin=vmin
        )
        expected = np.clip(
            (upsampled / vmax - vmin) / (1.0 - vmin), 0.0, 1.0
        )
        np.testing.assert_allclose(response, expected)
        self.assertAlmostEqual(vmin, 1.0 / vmax)
        self.assertEqual(float(response[1, 0, 0]), 0.0)
        self.assertAlmostEqual(float(response[1, 1, 1]), 1.0)

    def test_empty_positive_latent_values_use_unit_vmax(self):
        latent = np.asarray([[[0.0]], [[-2.0]], [[np.nan]]], dtype=np.float32)
        self.assertEqual(weight_viz.positive_percentile_vmax(latent), 1.0)

    def test_single_blur_is_not_rescaled(self):
        response = np.zeros((65, 65), dtype=np.float32)
        response[28:37, 28:37] = 1.0
        smooth = weight_viz.smooth_weight_response(response, blur_radius=12.0)
        self.assertGreater(float(smooth.max()), 0.0)
        self.assertLess(float(smooth.max()), 1.0)
        self.assertGreater(float(smooth[32, 24]), 0.0)

    def test_latent_smoothing_is_spatial_only_and_preserves_constant_maps(self):
        latent = np.zeros((2, 9, 9), dtype=np.float32)
        latent[0, 4, 4] = 1.0
        latent[1] = 0.25
        smooth = weight_viz.smooth_latent_spatially(latent, sigma=1.5)
        self.assertGreater(float(smooth[0, 4, 4]), 0.0)
        self.assertLess(float(smooth[0, 4, 4]), 1.0)
        self.assertEqual(float(smooth[1].min()), 0.25)
        self.assertEqual(float(smooth[1].max()), 0.25)

    def test_latent_product_is_normalized_per_chunk(self):
        s_chunks = [
            np.asarray([[[1.0, 2.0]]], dtype=np.float32),
            np.asarray([[[10.0, 20.0]]], dtype=np.float32),
        ]
        e_chunks = [
            np.asarray([[[4.0, 2.0]]], dtype=np.float32),
            np.asarray([[[0.4, 0.2]]], dtype=np.float32),
        ]
        result = weight_viz.normalize_latent_product_chunks(s_chunks, e_chunks)
        for chunk in result:
            np.testing.assert_allclose(chunk, np.ones_like(chunk))
            self.assertAlmostEqual(float(chunk.mean()), 1.0)

        e = [np.asarray([[[1.0, 2.0, 3.0]]], dtype=np.float32)]
        squared = weight_viz.normalize_latent_product_chunks(e, e)[0]
        np.testing.assert_allclose(squared, np.asarray([[[1.0, 4.0, 9.0]]]) / (14.0 / 3.0))
        self.assertAlmostEqual(float(squared.mean()), 1.0)

        squared_result = weight_viz.normalize_latent_product_chunks(
            s_chunks, [chunk * chunk for chunk in e_chunks]
        )
        for chunk in squared_result:
            np.testing.assert_allclose(
                chunk, np.asarray([[[4.0 / 3.0, 2.0 / 3.0]]], dtype=np.float32)
            )
            self.assertAlmostEqual(float(chunk.mean()), 1.0)

    def test_normalized_sigmoid_matches_endpoints_and_midpoint(self):
        response = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        curved = weight_viz.normalized_sigmoid(response, k=12.0)
        np.testing.assert_allclose(curved[[0, 2, 4]], [0.0, 0.5, 1.0], atol=1e-6)
        self.assertLess(float(curved[1]), 0.25)
        self.assertGreater(float(curved[3]), 0.75)

    def test_reference_fixed_overlay_matches_formula_without_blur(self):
        response = np.asarray(
            [[0.0, 0.5, 1.0]], dtype=np.float32
        )
        rgb = np.full((1, 3, 3), 100, dtype=np.uint8)
        stops = np.asarray(
            [
                [0, 0, 128],
                [127.5, 255, 127.5],
                [128, 0, 0],
            ],
            dtype=np.float32,
        )
        expected = np.uint8(np.clip(rgb.astype(np.float32) * 0.55 + stops * 0.65, 0, 255))
        actual = weight_viz.overlay_weight_response(rgb, response, blur_radius=0.0)
        np.testing.assert_array_equal(actual, expected)

    def test_export_keeps_png_npz_layout_and_records_one_vmax_per_mode(self):
        visualization = load_visualization_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video_dir = root / "videos"
            output_dir = root / "weight_visualization" / "episode1"
            video_dir.mkdir()
            (video_dir / "episode1.mp4").touch()
            args = SimpleNamespace(
                video_dir=video_dir,
                frame_stride=1,
                height=8,
                width=8,
                blur_radius=12.0,
            )
            weights = {
                "CAER": np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8),
                "MSE": np.linspace(0.0, 4.0, 3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8),
            }
            original = {mode: values.copy() for mode, values in weights.items()}
            latent_vmax = {"CAER": 7.5, "MSE": 2.25}
            frames = {index: np.full((8, 8, 3), 80, dtype=np.uint8) for index in (1, 2)}
            with mock.patch.object(
                visualization, "read_selected_video_frames", return_value=frames
            ):
                report = visualization.export_overlays(
                    args,
                    {"episode": "episode1"},
                    weights,
                    latent_vmax,
                    output_dir,
                )

            for mode, mode_name in (("CAER", "CAER"), ("MSE", "MSE")):
                spec = report["weights"][mode]
                self.assertEqual(spec["normalization"]["vmax"], latent_vmax[mode])
                expected_vmin = weight_viz.episode_response_vmin(
                    weights[mode], latent_vmax[mode]
                )
                self.assertEqual(spec["normalization"]["vmin"], expected_vmin)
                self.assertEqual(len(spec["pngs"]), 2)
                self.assertTrue(all(Path(path).suffix == ".png" for path in spec["pngs"]))
                self.assertTrue(all(Path(path).is_file() for path in spec["pngs"]))
                saved = np.load(output_dir / f"{mode_name}_weights.npz")["weights"]
                np.testing.assert_array_equal(saved, original[mode])
            self.assertEqual(report["normalization"], visualization.RENDERING_CONFIG["normalization"])
            self.assertEqual(
                report["vmin"],
                "episode_interpolated_min_excluding_first_frame",
            )
            self.assertEqual(report["percentile"], 99.0)
            self.assertEqual(report["blur"], "single_gaussian_radius_12")
            self.assertFalse(report["response_rescaled_after_blur"])
            self.assertEqual(
                report["color_response_curve"],
                "normalized_sigmoid_k12_after_blur",
            )
            self.assertEqual(
                report["colormap"],
                "six_stop_blue_cyan_yellow_red_reference",
            )
            self.assertEqual(report["overlay"], "0.55_rgb_plus_0.65_heat")
            self.assertEqual(report["output"], "png")
            self.assertFalse(list(output_dir.rglob("*.gif")))

            manifest = output_dir / "manifest.json"
            manifest.write_text(json.dumps(report), encoding="utf-8")
            self.assertIsNotNone(
                visualization.load_complete_episode_report(
                    output_dir.parent, "episode1"
                )
            )
            stale = dict(report)
            stale["normalization"] = "episode_positive_p2_p99.5"
            manifest.write_text(json.dumps(stale), encoding="utf-8")
            self.assertIsNone(
                visualization.load_complete_episode_report(
                    output_dir.parent, "episode1"
                )
            )


if __name__ == "__main__":
    unittest.main()
