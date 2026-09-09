"""Export manuscript-ready LaTeX tables from computed experiment JSON files."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def escape(text: object) -> str:
    value = str(text)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        value = value.replace(old, new)
    return value


def number(value: object, digits: int = 4) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "--"
    if numeric != 0 and abs(numeric) < 10 ** (-digits):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def statistics_table(path: Path) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Multi-run performance summary. Values are computed from the archived run outputs.}",
        r"\label{tab:computed-multirun}",
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Variant & $n$ & Mean & SD & 95\% CI low & 95\% CI high & Median \\",
        r"\hline",
    ]
    for variant, summary in sorted(report["summaries"].items()):
        lines.append(
            f"{escape(variant)} & {summary['n']} & {number(summary['mean'])} & {number(summary['std_sample'])} & "
            f"{number(summary['ci95_low'])} & {number(summary['ci95_high'])} & {number(summary['median'])} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])

    if report.get("comparisons"):
        lines.extend([
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\caption{Pre-specified comparisons with Holm correction for multiplicity.}",
            r"\label{tab:computed-significance}",
            r"\begin{tabular}{llllrrr}",
            r"\hline",
            r"Reference & Comparison & Test & Effect & Difference & $p$ & $p_{\mathrm{Holm}}$ \\",
            r"\hline",
        ])
        for row in report["comparisons"]:
            effect = f"{escape(row['effect_size_name'])}={number(row['effect_size'], 3)}"
            lines.append(
                f"{escape(row['reference'])} & {escape(row['comparison'])} & {escape(row['test'])} & {effect} & "
                f"{number(row['mean_difference_reference_minus_comparison'])} & {number(row['p_raw'])} & {number(row['p_holm'])} \\\\"
            )
        lines.extend([r"\hline", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def profile_table(path: Path) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    runtime = report["runtime"]
    return "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Measured computational profile under the recorded environment.}",
        r"\label{tab:computed-profile}",
        r"\begin{tabular}{lr}",
        r"\hline",
        f"Trainable parameters & {int(report['trainable_parameters']):,} \\\\ ",
        f"FLOPs per batch & {number(report.get('flops_per_batch'), 0)} \\\\ ",
        f"Mean batch latency (s) & {number(runtime['mean_batch_seconds'], 6)} \\\\ ",
        f"Throughput (images/s) & {number(runtime['images_per_second'], 2)} \\\\ ",
        f"Peak CUDA memory (bytes) & {number(runtime.get('peak_cuda_memory_bytes'), 0)} \\\\ ",
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ])


def robustness_table(path: Path) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Robustness under controlled perturbations.}",
        r"\label{tab:computed-robustness}",
        r"\begin{tabular}{lrr}",
        r"\hline",
        r"Condition & Accuracy & Absolute drop \\",
        r"\hline",
    ]
    for row in report["conditions"]:
        label = f"{row['condition']} ({number(row['severity'], 2)})"
        lines.append(f"{escape(label)} & {number(row['metrics']['accuracy'])} & {number(row['absolute_accuracy_drop'])} \\\\ ")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics-json")
    parser.add_argument("--profile-json")
    parser.add_argument("--robustness-json")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections = [
        "% AUTO-GENERATED FROM EXPERIMENT OUTPUTS. DO NOT EDIT NUMERIC VALUES MANUALLY.",
        "% Direct SOTA comparison remains indicative when evaluation protocols differ.",
    ]
    if args.statistics_json:
        sections.append(statistics_table(Path(args.statistics_json)))
    if args.profile_json:
        sections.append(profile_table(Path(args.profile_json)))
    if args.robustness_json:
        sections.append(robustness_table(Path(args.robustness_json)))
    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
