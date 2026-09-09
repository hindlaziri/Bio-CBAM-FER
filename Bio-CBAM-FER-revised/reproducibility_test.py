"""Compatibility entry point for the revised reproducibility workflow.

The former script evaluated freshly initialized, untrained models and therefore
did not measure reproducibility. This replacement prints the two required
commands: run matched training jobs, then aggregate their real outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Ablation JSON specification")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--reference", default="fmri")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = Path(__file__).resolve().parent
    commands = {
        "step_1_train_real_runs": [
            "python",
            str(project / "experiments" / "run_ablation.py"),
            "--spec",
            str(Path(args.spec).expanduser().resolve()),
            "--output-root",
            str(Path(args.output_root).expanduser().resolve()),
        ],
        "step_2_analyze_completed_runs": [
            "python",
            str(project / "experiments" / "analyze_runs.py"),
            "--root",
            str(Path(args.output_root).expanduser().resolve()),
            "--reference",
            args.reference,
            "--output-dir",
            str(Path(args.output_root).expanduser().resolve() / "statistics"),
        ],
    }
    print(json.dumps(commands, indent=2))
    print("Execute step 1 to train every run; execute step 2 only after all runs finish.")


if __name__ == "__main__":
    main()
