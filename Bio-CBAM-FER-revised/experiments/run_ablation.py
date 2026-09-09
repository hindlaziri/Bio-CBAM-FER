"""Run matched Bio-CBAM ablations from a JSON experiment specification.

Example specification:
{
  "seeds": [42, 123, 456, 789, 999],
  "base_args": ["--dataset", "fer2013", "--data-path", "/data/fer2013.csv",
                "--num-classes", "7", "--epochs", "100", "--require-prior"],
  "variants": {
    "fmri": ["/priors/fmri.npy"],
    "random_matched": ["/priors/random_seed42.npy"],
    "gaussian_center": ["/priors/gaussian.npy"],
    "generic_saliency": ["/priors/spectral_residual.npy"],
    "cbam": [],
    "resnet": []
  }
}
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


FORBIDDEN_BASE_ARGS = {"--seed", "--output-dir", "--prior", "--architecture", "--require-prior"}


def validate_base_args(arguments: List[str]) -> None:
    overlap = FORBIDDEN_BASE_ARGS.intersection(arguments)
    if overlap:
        raise ValueError(f"base_args must not contain orchestrated flags: {sorted(overlap)}")


def build_commands(specification: Dict[str, object], output_root: Path, train_script: Path) -> List[Dict[str, object]]:
    seeds = [int(seed) for seed in specification["seeds"]]
    base_args = [str(item) for item in specification["base_args"]]
    variants = specification["variants"]
    if not isinstance(variants, dict) or not variants:
        raise ValueError("variants must be a non-empty object")
    validate_base_args(base_args)
    commands: List[Dict[str, object]] = []
    for variant, prior_paths in variants.items():
        if not isinstance(prior_paths, list):
            raise ValueError(f"Variant {variant} must map to a list of prior paths")
        if variant == "resnet":
            architecture = "resnet"
        elif variant in {"cbam", "no_prior"}:
            architecture = "cbam"
        else:
            architecture = "biocbam"
        if architecture == "biocbam" and not prior_paths:
            raise ValueError(f"Bio-CBAM variant {variant} requires one or more prior paths")
        if architecture != "biocbam" and prior_paths:
            raise ValueError(f"Baseline variant {variant} must not define prior paths")
        for seed in seeds:
            destination = output_root / str(variant) / f"seed_{seed}"
            command = [sys.executable, str(train_script), *base_args, "--architecture", architecture, "--seed", str(seed), "--output-dir", str(destination)]
            for path in prior_paths:
                seed_path = str(path).format(seed=seed)
                command.extend(["--prior", str(Path(seed_path).expanduser().resolve())])
            if prior_paths:
                command.append("--require-prior")
            commands.append({"variant": variant, "seed": seed, "output_dir": str(destination), "command": command})
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec).expanduser().resolve()
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    project_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    commands = build_commands(specification, output_root, project_root / "train.py")
    (output_root / "ablation_plan.json").write_text(json.dumps(commands, indent=2), encoding="utf-8")

    results = []
    for item in commands:
        printable = shlex.join(item["command"])
        print(printable, flush=True)
        if args.dry_run:
            result = {**item, "returncode": None, "status": "dry_run"}
        else:
            process = subprocess.run(item["command"], cwd=project_root, check=False)
            result = {**item, "returncode": process.returncode, "status": "ok" if process.returncode == 0 else "failed"}
            if process.returncode != 0 and not args.continue_on_error:
                results.append(result)
                (output_root / "ablation_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
                raise SystemExit(process.returncode)
        results.append(result)
        (output_root / "ablation_status.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
