"""Aggregate real experiment runs and perform pre-specified statistical tests.

The script reads ``run_summary.json`` files produced by train.py. It never
creates or imputes scores. Runs are paired only when variants share the same
seed; otherwise Welch's independent-samples t-test is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class RunResult:
    variant: str
    seed: int
    value: float
    summary_path: str


def nested_value(payload: dict, dotted_path: str) -> float:
    current = payload
    for part in dotted_path.split("."):
        current = current[part]
    return float(current)


def discover_runs(root: Path, metric: str) -> List[RunResult]:
    results: List[RunResult] = []
    for summary_path in sorted(root.rglob("run_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config_path = summary_path.parent / "configuration.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing configuration.json next to {summary_path}")
        configuration = json.loads(config_path.read_text(encoding="utf-8"))
        seed = int(configuration["run_config"]["seed"])
        relative = summary_path.parent.relative_to(root)
        variant = relative.parts[0] if len(relative.parts) >= 1 else "default"
        results.append(RunResult(variant, seed, nested_value(summary, metric), str(summary_path)))
    if not results:
        raise ValueError(f"No run_summary.json found under {root}")
    duplicates = {(r.variant, r.seed) for r in results}
    if len(duplicates) != len(results):
        raise ValueError("Duplicate variant/seed combinations detected")
    return results


def t_interval(values: Sequence[float], confidence: float = 0.95) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float("nan"), float("nan")
    mean = float(np.mean(array))
    standard_error = float(stats.sem(array))
    critical = float(stats.t.ppf((1 + confidence) / 2, df=array.size - 1))
    return mean - critical * standard_error, mean + critical * standard_error


def summarize(values: Sequence[float]) -> Dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    low, high = t_interval(array)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std_sample": float(np.std(array, ddof=1)) if array.size > 1 else float("nan"),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "ci95_low": low,
        "ci95_high": high,
    }


def paired_effect_size(reference: np.ndarray, comparison: np.ndarray) -> float:
    differences = reference - comparison
    denominator = np.std(differences, ddof=1)
    return float(np.mean(differences) / denominator) if denominator > 0 else float("inf")


def hedges_g(reference: np.ndarray, comparison: np.ndarray) -> float:
    n1, n2 = reference.size, comparison.size
    pooled_variance = ((n1 - 1) * np.var(reference, ddof=1) + (n2 - 1) * np.var(comparison, ddof=1)) / (n1 + n2 - 2)
    if pooled_variance <= 0:
        return float("inf")
    cohen_d = (np.mean(reference) - np.mean(comparison)) / math.sqrt(pooled_variance)
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float(correction * cohen_d)


def compare_variants(reference_runs: Sequence[RunResult], other_runs: Sequence[RunResult]) -> Dict[str, object]:
    reference_by_seed = {run.seed: run.value for run in reference_runs}
    other_by_seed = {run.seed: run.value for run in other_runs}
    shared_seeds = sorted(set(reference_by_seed) & set(other_by_seed))
    exactly_paired = shared_seeds and len(shared_seeds) == len(reference_runs) == len(other_runs)

    if exactly_paired:
        reference = np.asarray([reference_by_seed[seed] for seed in shared_seeds], dtype=float)
        other = np.asarray([other_by_seed[seed] for seed in shared_seeds], dtype=float)
        if reference.size < 2:
            raise ValueError("At least two paired runs are required")
        statistic, p_value = stats.ttest_rel(reference, other)
        effect = paired_effect_size(reference, other)
        test_name = "two-sided paired t-test"
        effect_name = "Cohen dz"
        pairing = shared_seeds
    else:
        reference = np.asarray([run.value for run in reference_runs], dtype=float)
        other = np.asarray([run.value for run in other_runs], dtype=float)
        if min(reference.size, other.size) < 2:
            raise ValueError("At least two runs per variant are required for Welch's test")
        statistic, p_value = stats.ttest_ind(reference, other, equal_var=False)
        effect = hedges_g(reference, other)
        test_name = "two-sided Welch t-test"
        effect_name = "Hedges g"
        pairing = []

    return {
        "test": test_name,
        "statistic": float(statistic),
        "p_raw": float(p_value),
        "mean_difference_reference_minus_comparison": float(np.mean(reference) - np.mean(other)),
        "effect_size_name": effect_name,
        "effect_size": effect,
        "paired_seeds": pairing,
    }


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Family-wise error correction by Holm's step-down procedure."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def analyze(root: Path, metric: str, reference_variant: str) -> Dict[str, object]:
    runs = discover_runs(root, metric)
    groups: Dict[str, List[RunResult]] = {}
    for run in runs:
        groups.setdefault(run.variant, []).append(run)
    if reference_variant not in groups:
        raise ValueError(f"Reference variant '{reference_variant}' was not found")
    summaries = {name: summarize([run.value for run in values]) for name, values in groups.items()}

    comparisons = []
    for name in sorted(groups):
        if name == reference_variant:
            continue
        result = compare_variants(groups[reference_variant], groups[name])
        result.update({"reference": reference_variant, "comparison": name})
        comparisons.append(result)
    if comparisons:
        adjusted = holm_adjust([item["p_raw"] for item in comparisons])
        for item, p_adjusted in zip(comparisons, adjusted):
            item["p_holm"] = p_adjusted
            item["significant_at_0.05"] = bool(p_adjusted < 0.05)

    return {
        "metric": metric,
        "reference_variant": reference_variant,
        "summaries": summaries,
        "comparisons": comparisons,
        "runs": [run.__dict__ for run in runs],
        "multiplicity_control": "Holm family-wise correction across comparisons to the reference",
    }


def write_csv(report: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["reference", "comparison", "test", "statistic", "p_raw", "p_holm", "mean_difference_reference_minus_comparison", "effect_size_name", "effect_size", "significant_at_0.05"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["comparisons"]:
            writer.writerow({key: row.get(key) for key in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--metric", default="test_metrics.accuracy")
    parser.add_argument("--reference", default="fmri")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = analyze(Path(args.root).expanduser().resolve(), args.metric, args.reference)
    (output / "statistics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csv(report, output / "comparisons.csv")
    print(json.dumps(report["summaries"], indent=2))


if __name__ == "__main__":
    main()
