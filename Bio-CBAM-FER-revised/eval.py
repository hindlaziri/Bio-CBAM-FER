"""Evaluate a trained Bio-CBAM checkpoint without changing its architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from torch import nn

from dataset_scripts import create_dataloaders
from utils.checkpoint import load_checkpoint, model_from_checkpoint
from utils.evaluation import evaluate_model, save_evaluation
from utils.runtime import environment_report, resolve_device, save_json, set_global_seed


def evaluate_checkpoint(args: argparse.Namespace):
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model, model_config = model_from_checkpoint(checkpoint)
    model.to(device)

    run = checkpoint.get("run_config", {})
    dataset = args.dataset or run.get("dataset")
    data_path = args.data_path or run.get("data_path")
    if not dataset or not data_path:
        raise ValueError("Dataset and data path must come from CLI or checkpoint")
    seed = int(args.seed if args.seed is not None else run.get("seed", 42))
    set_global_seed(seed, deterministic=True)
    four_classes = str(run.get("four_classes", "angry,happy,sad,neutral"))

    loaders = create_dataloaders(
        dataset_name=dataset,
        data_path=data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=model_config.num_classes,
        four_class_names=[item.strip() for item in four_classes.split(",")],
        image_size=int(run.get("image_size", 224)),
        seed=seed,
        use_ssim_filtering=False,
        return_metadata=True,
    )
    split = args.split
    class_names = checkpoint["class_names"]
    criterion = nn.CrossEntropyLoss(label_smoothing=float(run.get("label_smoothing", 0.0)))
    metrics, outputs = evaluate_model(model, loaders[split], device, class_names, criterion)
    destination = Path(args.output_dir).expanduser().resolve()
    save_evaluation(metrics, outputs, destination)
    provenance = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "split": split,
        "dataset": dataset,
        "data_path": str(Path(data_path).expanduser().resolve()),
        "class_names": class_names,
        "model_config": checkpoint["model_config"],
        "training_environment": checkpoint.get("environment"),
        "evaluation_environment": environment_report(Path(__file__).resolve().parent),
    }
    save_json(provenance, destination / "provenance.json")
    return {"metrics": metrics, "provenance": provenance}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("fer2013", "ckplus", "jaffe"))
    parser.add_argument("--data-path")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    result = evaluate_checkpoint(parse_args())
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
