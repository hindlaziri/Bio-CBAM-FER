"""Compute calibration metrics from archived predictions without rerunning models."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    targets = np.asarray(payload["targets"], dtype=np.int64)
    predictions = np.asarray(payload["predictions"], dtype=np.int64)
    probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError(f"Invalid target/prediction shape in {path}")
    if probabilities.shape != (targets.size, int(probabilities.shape[1])):
        raise ValueError(f"Invalid probability shape in {path}")
    if np.any(targets < 0) or np.any(targets >= probabilities.shape[1]):
        raise ValueError(f"Target outside probability columns in {path}")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(f"Probabilities do not sum to one in {path}")
    return targets, predictions, probabilities


def _fixed_ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> tuple[float, float, List[Dict[str, float]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    contributions: List[Dict[str, float]] = []
    ece = 0.0
    maximum = 0.0
    for index in range(bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        include = (confidence >= lower) & (confidence < upper if index < bins - 1 else confidence <= upper)
        count = int(include.sum())
        if count == 0:
            contributions.append({"lower": lower, "upper": upper, "count": 0, "accuracy": 0.0, "confidence": 0.0, "gap": 0.0})
            continue
        bin_accuracy = float(correct[include].mean())
        bin_confidence = float(confidence[include].mean())
        gap = abs(bin_accuracy - bin_confidence)
        ece += count / confidence.size * gap
        maximum = max(maximum, gap)
        contributions.append({"lower": lower, "upper": upper, "count": count, "accuracy": bin_accuracy, "confidence": bin_confidence, "gap": gap})
    return float(ece), float(maximum), contributions


def _adaptive_ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    order = np.argsort(confidence)
    chunks = np.array_split(order, bins)
    value = 0.0
    for indices in chunks:
        if indices.size == 0:
            continue
        gap = abs(float(correct[indices].mean()) - float(confidence[indices].mean()))
        value += indices.size / confidence.size * gap
    return float(value)


def calibration_metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, bins: int) -> Dict[str, object]:
    epsilon = np.finfo(np.float64).eps
    confidence = probabilities.max(axis=1)
    correct = predictions == targets
    true_probability = probabilities[np.arange(targets.size), targets]
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    ece, mce, reliability = _fixed_ece(confidence, correct, bins)
    return {
        "support": int(targets.size),
        "class_count": int(probabilities.shape[1]),
        "accuracy": float(correct.mean()),
        "mean_confidence": float(confidence.mean()),
        "confidence_minus_accuracy": float(confidence.mean() - correct.mean()),
        "nll": float(-np.log(np.clip(true_probability, epsilon, 1.0)).mean()),
        "brier_multiclass": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece_equal_width": ece,
        "mce_equal_width": mce,
        "ece_equal_count": _adaptive_ece(confidence, correct, bins),
        "reliability_bins": reliability,
    }


def _parse_run(path: Path, runs_root: Path) -> tuple[str, str, str]:
    relative = path.relative_to(runs_root)
    parts = relative.parts
    if parts[0] == "fer4":
        return "FER-2013 4-class", parts[1], parts[2]
    if parts[0] == "fer4_random_control":
        return "FER-2013 4-class", parts[1], parts[2]
    if parts[0] == "fer7":
        return "FER-2013 7-class", parts[1], parts[2]
    if parts[0] == "fer7_crossmodal":
        return "FER-2013 7-class", parts[1], parts[2]
    if parts[0] == "jaffe":
        return "JAFFE subject-disjoint", parts[2], parts[1]
    raise ValueError(f"Unrecognized run path: {relative}")


def _discover(runs_root: Path, cross_root: Path | None) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(runs_root.glob("**/test/predictions.json")):
        protocol, variant, replicate = _parse_run(path, runs_root)
        records.append({"protocol": protocol, "variant": variant, "replicate": replicate, "path": path})
    if cross_root is not None:
        for path in sorted(cross_root.glob("*/seed_*/predictions.json")):
            records.append(
                {
                    "protocol": "FER-2013 to JAFFE, no adaptation",
                    "variant": path.parent.parent.name,
                    "replicate": path.parent.name,
                    "path": path,
                }
            )
    return records


def _confidence_interval(values: Sequence[float], confidence: float = 0.95) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    mean = float(data.mean())
    if data.size < 2:
        return mean, mean
    half = float(t.ppf((1.0 + confidence) / 2.0, df=data.size - 1) * data.std(ddof=1) / math.sqrt(data.size))
    return mean - half, mean + half


def _aggregate(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    grouped: Dict[tuple[str, str], List[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["protocol"]), str(record["variant"]))].append(record)
    output: Dict[str, object] = defaultdict(dict)
    metric_names = (
        "accuracy",
        "mean_confidence",
        "confidence_minus_accuracy",
        "nll",
        "brier_multiclass",
        "ece_equal_width",
        "mce_equal_width",
        "ece_equal_count",
    )
    for (protocol, variant), entries in sorted(grouped.items()):
        summary: Dict[str, object] = {"n_replicates": len(entries), "replicates": [str(item["replicate"]) for item in entries]}
        for metric in metric_names:
            values = [float(item["metrics"][metric]) for item in entries]
            low, high = _confidence_interval(values)
            summary[metric] = {
                "mean": float(statistics.mean(values)),
                "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
                "ci95_low": low,
                "ci95_high": high,
                "values": values,
            }
        output[protocol][variant] = summary
    return dict(output)


def analyze(args: argparse.Namespace) -> Dict[str, object]:
    runs_root = Path(args.runs_root).expanduser().resolve()
    cross_root = Path(args.cross_dataset_root).expanduser().resolve() if args.cross_dataset_root else None
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    discovered = _discover(runs_root, cross_root)
    if len(discovered) != 73:
        raise ValueError(f"Expected 73 prediction sets (58 primary + 15 cross-dataset), found {len(discovered)}")

    run_records: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    for item in discovered:
        path = Path(item["path"])
        targets, predictions, probabilities = _load_predictions(path)
        metrics = calibration_metrics(targets, predictions, probabilities, args.bins)
        record = {**item, "path": str(path), "metrics": metrics}
        run_records.append(record)
        rows.append(
            {
                "protocol": item["protocol"],
                "variant": item["variant"],
                "replicate": item["replicate"],
                **{key: value for key, value in metrics.items() if key != "reliability_bins"},
            }
        )
    payload = {
        "design": {
            "bins": args.bins,
            "ece_equal_width": "sum_b n_b/N * |accuracy_b - confidence_b|",
            "ece_equal_count": "same gap with approximately equal-count bins",
            "brier_multiclass": "mean_i sum_k (p_ik - 1[y_i=k])^2",
            "confidence_interval": "Student-t interval across seeds or subject-disjoint folds",
            "run_count": len(run_records),
        },
        "aggregate": _aggregate(run_records),
        "runs": run_records,
    }
    (destination / "calibration_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(destination / "calibration_runs.csv", index=False)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--cross-dataset-root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bins", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    payload = analyze(parse_args())
    compact = {
        protocol: {
            variant: {
                metric: round(float(values[metric]["mean"]), 6)
                for metric in ("accuracy", "nll", "brier_multiclass", "ece_equal_width")
            }
            for variant, values in variants.items()
        }
        for protocol, variants in payload["aggregate"].items()
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
