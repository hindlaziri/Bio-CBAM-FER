"""End-to-end software smoke test on temporary synthetic data only."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def pixels(label: int, replicate: int) -> str:
    base_intensity = {0: 25, 3: 75, 4: 125, 6: 175}[label]
    image = np.full((48, 48), base_intensity, dtype=np.uint8)
    image[8 + replicate : 16 + replicate, 8 + label * 4 : 16 + label * 4] = 220
    return " ".join(str(value) for value in image.ravel())


def run(command: list[str], project: Path) -> None:
    subprocess.run(command, cwd=project, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="biocbam_smoke_") as directory:
        root = Path(directory)
        rows = []
        labels = (0, 3, 4, 6)  # angry, happy, sad, neutral
        for usage, replicates in (("Training", 2), ("PublicTest", 1), ("PrivateTest", 1)):
            for label in labels:
                for replicate in range(replicates):
                    rows.append({"emotion": label, "pixels": pixels(label, replicate), "Usage": usage})
        csv_path = root / "fer_synthetic.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        prior_path = root / "prior.npy"
        np.save(prior_path, np.linspace(0, 1, 64 * 64, dtype=np.float32).reshape(64, 64), allow_pickle=False)
        output = root / "run"

        run([
            sys.executable, str(project / "train.py"),
            "--dataset", "fer2013", "--data-path", str(csv_path),
            "--output-dir", str(output), "--num-classes", "4",
            "--four-classes", "angry,happy,sad,neutral",
            "--architecture", "biocbam", "--prior", str(prior_path), "--require-prior",
            "--backbone", "resnet18", "--image-size", "64", "--epochs", "1",
            "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
            "--seed", "7", "--early-stopping-patience", "0",
        ], project)

        checkpoint_path = output / "best_checkpoint.pt"
        evaluation_dir = root / "evaluation"
        run([
            sys.executable, str(project / "eval.py"), "--checkpoint", str(checkpoint_path),
            "--data-path", str(csv_path), "--batch-size", "4", "--num-workers", "0",
            "--device", "cpu", "--output-dir", str(evaluation_dir),
        ], project)

        profile_path = root / "profile.json"
        run([
            sys.executable, str(project / "experiments" / "profile_model.py"),
            "--checkpoint", str(checkpoint_path), "--batch-size", "1", "--height", "64",
            "--width", "64", "--warmup", "1", "--repeats", "2", "--device", "cpu",
            "--output", str(profile_path),
        ], project)

        conditions_path = root / "conditions.json"
        conditions_path.write_text(json.dumps([
            {"name": "clean", "severity": 0.0},
            {"name": "gaussian_noise", "severity": 0.03},
        ]), encoding="utf-8")
        robustness_path = root / "robustness.json"
        run([
            sys.executable, str(project / "experiments" / "evaluate_robustness.py"),
            "--checkpoint", str(checkpoint_path), "--data-path", str(csv_path),
            "--conditions-json", str(conditions_path), "--batch-size", "4", "--num-workers", "0",
            "--device", "cpu", "--output", str(robustness_path),
        ], project)

        attention_dir = root / "attention"
        run([
            sys.executable, str(project / "experiments" / "evaluate_attention.py"),
            "--checkpoint", str(checkpoint_path), "--data-path", str(csv_path),
            "--stage", "4", "--max-images", "1", "--batch-size", "4", "--num-workers", "0",
            "--device", "cpu", "--output-dir", str(attention_dir),
        ], project)

        latex_path = root / "generated_results.tex"
        run([
            sys.executable, str(project / "experiments" / "export_latex_tables.py"),
            "--profile-json", str(profile_path), "--robustness-json", str(robustness_path),
            "--output", str(latex_path),
        ], project)

        expected = [
            checkpoint_path, output / "last_checkpoint.pt", output / "history.json",
            output / "run_summary.json", output / "test" / "metrics.json",
            output / "test" / "predictions.json", evaluation_dir / "metrics.json",
            evaluation_dir / "predictions.json", profile_path, robustness_path,
            attention_dir / "attention_report.json", latex_path,
        ]
        missing = [str(path) for path in expected if not path.exists()]
        if missing:
            raise AssertionError(f"Smoke workflow did not create: {missing}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "prior_bank" not in checkpoint or checkpoint.get("format_version") != 2:
            raise AssertionError("Checkpoint is not self-contained format_version=2")
        summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
        if summary["epochs_completed"] != 1:
            raise AssertionError("Smoke training did not complete exactly one epoch")
    print("Synthetic end-to-end workflow passed; all temporary outputs were deleted.")


if __name__ == "__main__":
    main()
