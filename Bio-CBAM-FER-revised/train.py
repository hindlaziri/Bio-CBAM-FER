"""Train Bio-CBAM with auditable configuration and real multi-stage priors."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataset_scripts import DEFAULT_FOUR_CLASSES, create_dataloaders
from models import BioCBAMConfig, create_model
from utils.evaluation import evaluate_model, save_evaluation
from utils.runtime import environment_report, load_prior_bank, resolve_device, save_json, set_global_seed


def class_names_from_dataset(dataset) -> List[str]:
    if hasattr(dataset, "class_names"):
        return list(dataset.class_names)
    if hasattr(dataset, "label_map"):
        return [name for name, _ in sorted(dataset.label_map.items(), key=lambda item: item[1])]
    raise ValueError("Dataset does not expose class names")


def unpack_batch(batch):
    if len(batch) == 2:
        return batch
    if len(batch) == 3:
        return batch[0], batch[1]
    raise ValueError("Expected batch=(images, labels[, metadata])")


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
    grad_clip: Optional[float],
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()
    progress = tqdm(loader, desc="train", leave=False)
    for batch in progress:
        images, labels = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits, _ = model(images)
            classification_loss = criterion(logits, labels)
            loss = classification_loss + model.lambda_regularization_loss()
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        total_samples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        progress.set_postfix(loss=f"{loss.detach().item():.4f}")
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "seconds": time.perf_counter() - start,
    }


def checkpoint_payload(
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_metric: float,
    history: List[Dict[str, object]],
    run_config: Dict[str, object],
    class_names: Sequence[str],
    environment: Dict[str, object],
) -> Dict[str, object]:
    payload = {
        "format_version": 2,
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_validation_metric": best_metric,
        "history": history,
        "run_config": run_config,
        "model_config": asdict(model.config),
        "class_names": list(class_names),
        "environment": environment,
        "lambda_values": [float(v) for v in model.lambda_values().detach().cpu()],
    }
    if model.prior_mixer is not None:
        payload["prior_bank"] = model.prior_mixer.priors.detach().cpu()
    return payload


def fit(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.seed, deterministic=not args.allow_nondeterministic)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")

    four_classes = tuple(name.strip().lower() for name in args.four_classes.split(",") if name.strip())
    if args.num_classes == 4 and len(four_classes) != 4:
        raise ValueError("--four-classes must contain exactly four official FER-2013 names")
    dataloaders = create_dataloaders(
        dataset_name=args.dataset,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        four_class_names=four_classes,
        image_size=args.image_size,
        seed=args.seed,
        use_ssim_filtering=args.ssim_filter,
        ssim_threshold=args.ssim_threshold,
        ssim_audit_path=str(output_dir / "ssim_training_audit.json") if args.ssim_filter else None,
        return_metadata=True,
    )
    class_names = class_names_from_dataset(dataloaders["train"].dataset)
    if len(class_names) != args.num_classes:
        raise ValueError(f"Requested {args.num_classes} classes but dataset exposes {len(class_names)}")

    resume_checkpoint = (
        torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.resume else None
    )
    prior_bank = load_prior_bank(args.prior)
    if prior_bank is None and resume_checkpoint is not None:
        prior_bank = resume_checkpoint.get("prior_bank")
    if args.architecture == "biocbam" and prior_bank is None:
        raise ValueError("architecture=biocbam requires at least one --prior or a resumed checkpoint containing a prior bank")
    if args.architecture in {"resnet", "cbam"} and prior_bank is not None:
        raise ValueError(f"architecture={args.architecture} must not receive an external prior")
    if args.require_prior and prior_bank is None:
        raise ValueError("--require-prior was set but no prior is available from CLI or checkpoint")
    model_config = BioCBAMConfig(
        architecture=args.architecture,
        num_classes=args.num_classes,
        backbone=args.backbone,
        pretrained=args.pretrained,
        reduction_ratio=args.reduction_ratio,
        spatial_kernel_size=args.spatial_kernel_size,
        lambda_init=args.lambda_init,
        lambda_nonnegative=args.lambda_nonnegative,
        shared_lambda=args.shared_lambda,
        lambda_regularization=args.lambda_regularization,
        classifier_dropout=args.dropout,
    )
    model = create_model(model_config, prior_bank=prior_bank).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=args.scheduler_patience)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_metric = float("-inf")
    history: List[Dict[str, object]] = []
    best_path = output_dir / "best_checkpoint.pt"
    last_path = output_dir / "last_checkpoint.pt"
    environment = environment_report(Path(__file__).resolve().parent)
    run_config = vars(args).copy()
    run_config["device_resolved"] = str(device)
    run_config["prior_mode"] = "bank" if prior_bank is not None else "none"
    save_json({"run_config": run_config, "model_config": asdict(model_config), "environment": environment}, output_dir / "configuration.json")

    if resume_checkpoint is not None:
        saved_config = dict(resume_checkpoint["model_config"])
        requested_config = asdict(model_config)
        for key in sorted(set(saved_config) | set(requested_config)):
            if key == "pretrained":
                continue  # Initialization source is irrelevant after weights are restored.
            if saved_config.get(key) != requested_config.get(key):
                raise ValueError(
                    f"Resume configuration mismatch for {key}: "
                    f"checkpoint={saved_config.get(key)!r}, requested={requested_config.get(key)!r}"
                )
        if list(resume_checkpoint.get("class_names", [])) != list(class_names):
            raise ValueError("Resume checkpoint class order differs from the current dataset")
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        scaler.load_state_dict(resume_checkpoint.get("scaler_state", {}))
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_metric = float(resume_checkpoint["best_validation_metric"])
        history = list(resume_checkpoint.get("history", []))

    stale_epochs = 0
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_epoch(
            model, dataloaders["train"], optimizer, criterion, device, scaler,
            amp_enabled, args.grad_clip,
        )
        validation_metrics, _ = evaluate_model(
            model, dataloaders["val"], device, class_names, criterion
        )
        monitored = float(validation_metrics[args.monitor])
        scheduler.step(monitored)
        record: Dict[str, object] = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "validation": validation_metrics,
            "lambda_values": [float(v) for v in model.lambda_values().detach().cpu()],
        }
        history.append(record)
        save_json(history, output_dir / "history.json")

        improved = monitored > best_metric + args.min_delta
        if improved:
            best_metric = monitored
            stale_epochs = 0
            torch.save(
                checkpoint_payload(
                    model, optimizer, scheduler, scaler, epoch, best_metric,
                    history, run_config, class_names, environment,
                ),
                best_path,
            )
        else:
            stale_epochs += 1
        torch.save(
            checkpoint_payload(
                model, optimizer, scheduler, scaler, epoch, best_metric,
                history, run_config, class_names, environment,
            ),
            last_path,
        )
        print(json.dumps({"epoch": epoch, "train": train_metrics, "validation": validation_metrics, "best": best_metric}))
        if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
            break

    if not best_path.exists():
        raise RuntimeError("Training did not produce a best checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state"])
    test_metrics, test_outputs = evaluate_model(
        model, dataloaders["test"], device, class_names, criterion
    )
    save_evaluation(test_metrics, test_outputs, output_dir / "test")
    summary = {
        "best_checkpoint": str(best_path),
        "best_validation_metric": best_metric,
        "test_metrics": test_metrics,
        "epochs_completed": len(history),
        "class_names": class_names,
        "note": "Test set evaluated once after validation-based checkpoint selection.",
    }
    save_json(summary, output_dir / "run_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fer2013", "ckplus", "jaffe"), required=True)
    parser.add_argument("--data-path", required=True, help="FER CSV or subject-aware manifest CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, choices=(4, 7, 8), default=7)
    parser.add_argument("--four-classes", default=",".join(DEFAULT_FOUR_CLASSES))
    parser.add_argument("--architecture", choices=("resnet", "cbam", "biocbam"), default="biocbam")
    parser.add_argument("--prior", action="append", default=[], help="Repeat for each 2D .npy prior")
    parser.add_argument("--require-prior", action="store_true")
    parser.add_argument("--backbone", choices=("resnet18", "resnet50"), default="resnet50")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--reduction-ratio", type=int, default=16)
    parser.add_argument("--spatial-kernel-size", type=int, choices=(3, 7), default=7)
    parser.add_argument("--lambda-init", type=float, default=0.0)
    parser.add_argument("--lambda-nonnegative", action="store_true")
    parser.add_argument("--shared-lambda", action="store_true")
    parser.add_argument("--lambda-regularization", type=float, default=0.0)
    parser.add_argument("--ssim-filter", action="store_true")
    parser.add_argument("--ssim-threshold", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--allow-nondeterministic", action="store_true")
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--monitor", choices=("accuracy", "balanced_accuracy", "f1_macro"), default="f1_macro")
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    summary = fit(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
