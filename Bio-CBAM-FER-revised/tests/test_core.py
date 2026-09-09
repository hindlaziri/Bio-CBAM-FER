"""Software tests only; synthetic arrays are never scientific results."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from dataset_scripts.dataset_loader import (
    FER2013Dataset,
    ImageRecord,
    RecordDataset,
    SSIMFilterConfig,
    filter_training_records_ssim,
    validate_subject_disjointness,
)
from experiments.analyze_runs import analyze, holm_adjust
from models import BioCBAM, BioCBAMConfig, ResNetBaseline, create_model
from priors.tps import ThinPlateSpline2D, warp_image_tps
from priors.variants import gaussian_center_prior, random_matched_prior, spectral_residual_saliency


class TestTPS(unittest.TestCase):
    def test_identity_control_points(self):
        points = np.asarray([[0, 0], [9, 0], [0, 9], [9, 9]], dtype=float)
        transform = ThinPlateSpline2D.fit(points, points, regularization=1e-6)
        query = np.asarray([[2.0, 3.0], [7.0, 8.0]])
        np.testing.assert_allclose(transform.transform(query), query, atol=1e-5)

    def test_identity_image_warp(self):
        image = np.arange(100, dtype=np.float32).reshape(10, 10)
        points = np.asarray([[0, 0], [9, 0], [0, 9], [9, 9], [4.5, 4.5]])
        warped = warp_image_tps(image, points, points, (10, 10), regularization=1e-6)
        np.testing.assert_allclose(warped, image, atol=1e-3)


class TestPriors(unittest.TestCase):
    def test_random_matched_preserves_distribution(self):
        source = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
        shuffled = random_matched_prior(source, seed=42)
        np.testing.assert_allclose(np.sort(source.ravel()), np.sort(shuffled.ravel()))
        self.assertFalse(np.array_equal(source, shuffled))

    def test_gaussian_and_saliency_ranges(self):
        gaussian = gaussian_center_prior(32, 32, 0.2)
        saliency = spectral_residual_saliency(np.eye(32, dtype=np.uint8) * 255, (32, 32))
        for prior in (gaussian, saliency):
            self.assertEqual(prior.shape, (32, 32))
            self.assertGreaterEqual(float(prior.min()), 0.0)
            self.assertLessEqual(float(prior.max()), 1.0)


class TestModel(unittest.TestCase):
    def test_multiscale_forward_and_lambdas(self):
        config = BioCBAMConfig(num_classes=4, backbone="resnet18", pretrained=False, lambda_init=0.25)
        prior_bank = torch.rand(2, 32, 32)
        model = BioCBAM(config, prior_bank=prior_bank).eval()
        with torch.inference_mode():
            logits, diagnostics = model(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 4))
        self.assertEqual(len(diagnostics["stages"]), 4)
        self.assertEqual(tuple(diagnostics["lambda_values"].shape), (4,))
        for stage in diagnostics["stages"]:
            self.assertEqual(stage["spatial_attention"].shape[0], 2)
            self.assertEqual(stage["spatial_attention"].shape[1], 1)

    def test_checkpoint_roundtrip(self):
        config = BioCBAMConfig(num_classes=4, backbone="resnet18", pretrained=False)
        prior_bank = torch.rand(1, 16, 16)
        first = BioCBAM(config, prior_bank=prior_bank)
        state = first.state_dict()
        second = BioCBAM(config, prior_bank=prior_bank.clone())
        second.load_state_dict(state, strict=True)

    def test_resnet_and_cbam_baselines_are_distinct(self):
        resnet = create_model(BioCBAMConfig(architecture="resnet", num_classes=4, backbone="resnet18"))
        cbam = create_model(BioCBAMConfig(architecture="cbam", num_classes=4, backbone="resnet18"))
        self.assertIsInstance(resnet, ResNetBaseline)
        self.assertIsInstance(cbam, BioCBAM)
        self.assertEqual(cbam.prior_mixer, None)
        self.assertEqual(cbam.lambda_values().numel(), 0)
        self.assertTrue(all(not hasattr(block, "gate") for block in cbam.bio_blocks))
        with torch.inference_mode():
            resnet_logits, resnet_info = resnet(torch.randn(2, 3, 64, 64))
            cbam_logits, cbam_info = cbam(torch.randn(2, 3, 64, 64))
        self.assertEqual(tuple(resnet_logits.shape), (2, 4))
        self.assertEqual(tuple(cbam_logits.shape), (2, 4))
        self.assertEqual(len(resnet_info["stages"]), 0)
        self.assertEqual(len(cbam_info["stages"]), 4)


class TestDatasets(unittest.TestCase):
    @staticmethod
    def pixels(value: int) -> str:
        return " ".join([str(value)] * 2304)

    def test_fer_official_splits_and_four_class_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fer.csv"
            rows = [
                {"emotion": 0, "pixels": self.pixels(0), "Usage": "Training"},
                {"emotion": 3, "pixels": self.pixels(64), "Usage": "PublicTest"},
                {"emotion": 6, "pixels": self.pixels(128), "Usage": "PrivateTest"},
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            self.assertEqual(len(FER2013Dataset(str(path), "train")), 1)
            self.assertEqual(len(FER2013Dataset(str(path), "val")), 1)
            self.assertEqual(len(FER2013Dataset(str(path), "test")), 1)
            with self.assertRaises(ValueError):
                FER2013Dataset(str(path), "train", class_names=("angry", "happy", "sad", "confusion"))

    def test_fer_directory_preserves_official_usage_prefixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train" / "angry").mkdir(parents=True)
            (root / "test" / "angry").mkdir(parents=True)
            image = Image.fromarray(np.full((48, 48), 64, dtype=np.uint8))
            image.save(root / "train" / "angry" / "Training_1.jpg")
            image.save(root / "test" / "angry" / "PublicTest_1.jpg")
            image.save(root / "test" / "angry" / "PrivateTest_1.jpg")
            classes = ("angry",)
            self.assertEqual(len(FER2013Dataset(str(root), "train", class_names=classes)), 1)
            self.assertEqual(len(FER2013Dataset(str(root), "val", class_names=classes)), 1)
            self.assertEqual(len(FER2013Dataset(str(root), "test", class_names=classes)), 1)

    def test_subject_leakage_detection(self):
        frame = pd.DataFrame({"subject_id": ["S1", "S1"], "split": ["train", "test"]})
        with self.assertRaises(ValueError):
            validate_subject_disjointness(frame)

    def test_ssim_filter_training_only(self):
        records = [
            ImageRecord("a", 0, "angry", "train", pixels=" ".join(["128"] * 2304)),
            ImageRecord("b", 0, "angry", "train", pixels=" ".join(["128"] * 2304)),
        ]
        kept = filter_training_records_ssim(records, RecordDataset.read_record, SSIMFilterConfig(threshold=0.99))
        self.assertEqual([record.identifier for record in kept], ["a"])


class TestStatistics(unittest.TestCase):
    def test_holm_is_monotone_in_sorted_order(self):
        raw = [0.01, 0.04, 0.03]
        adjusted = holm_adjust(raw)
        self.assertTrue(all(0 <= value <= 1 for value in adjusted))
        ordered = [adjusted[index] for index in np.argsort(raw)]
        self.assertEqual(ordered, sorted(ordered))

    def test_run_discovery_and_paired_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant, values in {"fmri": [0.8, 0.82], "gaussian": [0.7, 0.71]}.items():
                for seed, value in zip((42, 123), values):
                    run = root / variant / f"seed_{seed}"
                    run.mkdir(parents=True)
                    (run / "run_summary.json").write_text(json.dumps({"test_metrics": {"accuracy": value}}))
                    (run / "configuration.json").write_text(json.dumps({"run_config": {"seed": seed}}))
            report = analyze(root, "test_metrics.accuracy", "fmri")
            self.assertEqual(report["comparisons"][0]["test"], "two-sided paired t-test")
            self.assertEqual(report["summaries"]["fmri"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
