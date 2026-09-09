"""Compare classifiers on the same test examples using exact McNemar tests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import binomtest

try:
    from .analyze_runs import holm_adjust
except ImportError:  # Direct script execution.
    from analyze_runs import holm_adjust


def load_predictions(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"targets", "predictions"}
    if not required.issubset(payload):
        raise ValueError(f"{path} is missing {sorted(required - set(payload))}")
    return payload


def compare(reference: Dict[str, object], comparison: Dict[str, object]) -> Dict[str, object]:
    targets = np.asarray(reference["targets"], dtype=int)
    ref_pred = np.asarray(reference["predictions"], dtype=int)
    other_targets = np.asarray(comparison["targets"], dtype=int)
    other_pred = np.asarray(comparison["predictions"], dtype=int)
    if not (targets.shape == ref_pred.shape == other_targets.shape == other_pred.shape):
        raise ValueError("Prediction files have incompatible lengths")
    if not np.array_equal(targets, other_targets):
        raise ValueError("Prediction files do not contain the same ordered targets")
    if "identifiers" in reference or "identifiers" in comparison:
        if reference.get("identifiers") != comparison.get("identifiers"):
            raise ValueError("Example identifiers are not aligned")

    reference_correct = ref_pred == targets
    comparison_correct = other_pred == targets
    ref_only = int(np.sum(reference_correct & ~comparison_correct))
    comparison_only = int(np.sum(~reference_correct & comparison_correct))
    discordant = ref_only + comparison_only
    p_value = 1.0 if discordant == 0 else float(binomtest(min(ref_only, comparison_only), discordant, 0.5, alternative="two-sided").pvalue)
    odds_ratio = float((ref_only + 0.5) / (comparison_only + 0.5))
    return {
        "n_examples": int(targets.size),
        "reference_accuracy": float(np.mean(reference_correct)),
        "comparison_accuracy": float(np.mean(comparison_correct)),
        "accuracy_difference_reference_minus_comparison": float(np.mean(reference_correct) - np.mean(comparison_correct)),
        "reference_correct_comparison_wrong": ref_only,
        "reference_wrong_comparison_correct": comparison_only,
        "discordant_pairs": discordant,
        "test": "two-sided exact McNemar test",
        "p_raw": p_value,
        "matched_odds_ratio_haldane_anscombe": odds_ratio,
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/predictions.json")
    name, path = value.split("=", 1)
    return name, Path(path).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=parse_named_path)
    parser.add_argument("--comparison", required=True, action="append", type=parse_named_path)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_name, reference_path = args.reference
    reference = load_predictions(reference_path)
    rows: List[Dict[str, object]] = []
    for name, path in args.comparison:
        row = compare(reference, load_predictions(path))
        row.update({"reference": reference_name, "comparison": name, "reference_file": str(reference_path), "comparison_file": str(path)})
        rows.append(row)
    adjusted = holm_adjust([row["p_raw"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
        row["significant_at_0.05"] = bool(value < 0.05)
    output = {"comparisons": rows, "multiplicity_control": "Holm correction"}
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
