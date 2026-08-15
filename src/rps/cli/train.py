from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from rps.checkpoint import save_checkpoint
from rps.constants import DATA_DIR, DEFAULT_CHECKPOINT_PATH, REPORTS_DIR
from rps.data import (
    LandmarkFrameDataset,
    ReviewManifestError,
    dataset_fingerprint,
    load_reviewed_trajectories,
    participant_split,
    save_split_manifest,
    trajectories_for_participants,
)
from rps.device import benchmark_training_devices, mps_available
from rps.metrics import checkpoint_passes_promotion, evaluate_trajectories
from rps.model import GestureMLP, calculate_activation_scales


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if mps_available():
        torch.mps.manual_seed(seed)


def _resolve_training_device(requested: str, model: GestureMLP) -> tuple[str, dict[str, Any]]:
    if requested == "mps":
        if not mps_available():
            raise RuntimeError("MPS was requested but is not available")
        return "mps", {"selected": "mps", "reason": "explicitly requested"}
    if requested == "cpu":
        return "cpu", {"selected": "cpu", "reason": "explicitly requested"}
    benchmark = benchmark_training_devices(model)
    details = {
        "selected": benchmark.selected,
        "reason": benchmark.reason,
        "timings": {
            name: {"median_ms": timing.median_ms, "p95_ms": timing.p95_ms}
            for name, timing in benchmark.timings.items()
        },
    }
    return benchmark.selected, details


@torch.inference_mode()
def _validation_logits(
    model: GestureMLP, dataset: LandmarkFrameDataset, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.as_tensor(dataset.features, dtype=torch.float32, device=device)
    logits = model(features).logits.detach().cpu()
    labels = torch.as_tensor(dataset.labels, dtype=torch.long)
    return logits, labels


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit a scalar temperature by deterministic validation NLL grid search."""

    criterion = nn.CrossEntropyLoss()
    candidates = np.linspace(0.5, 3.0, 251)
    losses = [float(criterion(logits / float(value), labels)) for value in candidates]
    return float(candidates[int(np.argmin(losses))])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the mid-gesture MLP")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--report-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--frame-sampling", choices=("fixed", "all"), default="fixed")
    parser.add_argument("--balanced-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument(
        "--refit-train-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After model selection, refit from scratch on train plus validation for best_epoch",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seeds(args.seed)
    review_manifest_path = args.review_manifest or args.data / "review-manifest.json"
    try:
        trajectories, review_summary = load_reviewed_trajectories(
            args.data, review_manifest_path
        )
    except ReviewManifestError as error:
        raise SystemExit(str(error)) from error
    if not trajectories:
        raise SystemExit(f"No trajectories found under {args.data}; run rps-capture first")
    print(
        "Data review: "
        f"{review_summary.included}/{review_summary.total} included, "
        f"{review_summary.relabeled} relabeled, "
        f"{review_summary.excluded} excluded, "
        f"{review_summary.unreviewed} unreviewed"
    )

    split = participant_split(trajectories, args.split_seed)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.report_dir / "split-manifest.json"
    save_split_manifest(manifest_path, split, args.split_seed)
    train_trajectories = trajectories_for_participants(trajectories, split["train"])
    validation_trajectories = trajectories_for_participants(trajectories, split["validation"])
    test_trajectories = trajectories_for_participants(trajectories, split["test"])

    training_dataset = LandmarkFrameDataset(
        train_trajectories,
        augment=True,
        seed=args.seed,
        sampling=args.frame_sampling,
    )
    calibration_dataset = LandmarkFrameDataset(
        train_trajectories,
        augment=False,
        seed=args.seed,
        sampling=args.frame_sampling,
    )
    validation_dataset = LandmarkFrameDataset(
        validation_trajectories, augment=False, seed=args.seed
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.balanced_sampling:
        sampler = WeightedRandomSampler(
            training_dataset.balanced_sample_weights(),
            num_samples=len(training_dataset),
            replacement=True,
            generator=generator,
        )
    loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )

    model = GestureMLP()
    try:
        device_name, device_benchmark = _resolve_training_device(args.device, model)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    device = torch.device(device_name)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.02,
        )
    criterion = nn.CrossEntropyLoss()

    best_state: dict[str, torch.Tensor] | None = None
    best_score = (-1.0, -1.0)
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features).logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_metrics = evaluate_trajectories(
            model, validation_trajectories, temperature=1.0, device=device
        )
        score = (
            float(validation_metrics["forced_lock"]["accuracy"]),
            float(validation_metrics["final_pose"]["macro_f1"]),
        )
        loss_value = float(np.mean(losses)) if losses else 0.0
        learning_rate = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": epoch,
                "loss": loss_value,
                "learning_rate": learning_rate,
                "validation_lock_accuracy": score[0],
                "validation_final_macro_f1": score[1],
            }
        )
        print(
            f"epoch {epoch:03d} loss={loss_value:.4f} "
            f"lock_acc={score[0]:.3f} final_f1={score[1]:.3f} lr={learning_rate:.2e}"
        )
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epoch >= args.min_epochs and epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break
        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise SystemExit("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    logits, labels = _validation_logits(model, validation_dataset, device)
    temperature = fit_temperature(logits, labels)
    validation_metrics = evaluate_trajectories(
        model, validation_trajectories, temperature=temperature, device=device
    )

    refit_details: dict[str, Any] = {"enabled": False}
    if args.refit_train_validation:
        refit_participants = split["train"] + split["validation"]
        refit_trajectories = trajectories_for_participants(trajectories, refit_participants)
        set_seeds(args.seed)
        refit_dataset = LandmarkFrameDataset(
            refit_trajectories,
            augment=True,
            seed=args.seed,
            sampling=args.frame_sampling,
        )
        calibration_dataset = LandmarkFrameDataset(
            refit_trajectories,
            augment=False,
            seed=args.seed,
            sampling=args.frame_sampling,
        )
        refit_generator = torch.Generator().manual_seed(args.seed)
        refit_sampler = None
        if args.balanced_sampling:
            refit_sampler = WeightedRandomSampler(
                refit_dataset.balanced_sample_weights(),
                num_samples=len(refit_dataset),
                replacement=True,
                generator=refit_generator,
            )
        refit_loader = DataLoader(
            refit_dataset,
            batch_size=args.batch_size,
            shuffle=refit_sampler is None,
            sampler=refit_sampler,
            num_workers=0,
            generator=refit_generator,
        )
        model = GestureMLP().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = None
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs,
                eta_min=args.learning_rate * 0.02,
            )
        for refit_epoch in range(1, best_epoch + 1):
            model.train()
            refit_losses: list[float] = []
            for features, refit_labels in refit_loader:
                features = features.to(device)
                refit_labels = refit_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(features).logits, refit_labels)
                loss.backward()
                optimizer.step()
                refit_losses.append(float(loss.detach().cpu()))
            if scheduler is not None:
                scheduler.step()
            if refit_epoch == 1 or refit_epoch % 10 == 0 or refit_epoch == best_epoch:
                print(
                    f"refit epoch {refit_epoch:03d}/{best_epoch:03d} "
                    f"loss={float(np.mean(refit_losses)):.4f}"
                )
        model.eval()
        refit_details = {
            "enabled": True,
            "epochs": best_epoch,
            "participants": sorted(refit_participants),
            "samples": len(refit_dataset),
            "temperature_source": "selection validation split",
        }

    activation_scales = calculate_activation_scales(model, calibration_dataset.features, device)
    test_metrics = evaluate_trajectories(
        model, test_trajectories, temperature=temperature, device=device
    )
    promoted, failures = checkpoint_passes_promotion(test_metrics)
    source_paths = sorted(args.data.rglob("*.npz"))
    fingerprint = dataset_fingerprint(source_paths, review_manifest_path)
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "best_epoch": best_epoch,
        "temperature": temperature,
        "device_benchmark": device_benchmark,
        "split": split,
        "data_fingerprint": fingerprint,
        "data_review": {
            "manifest": str(review_manifest_path),
            **review_summary.as_dict(),
        },
        "training_config": {
            "frame_sampling": args.frame_sampling,
            "balanced_sampling": args.balanced_sampling,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "min_epochs": args.min_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "training_samples": len(training_dataset),
        },
        "refit": refit_details,
        "history": history,
        "validation": validation_metrics,
        "test": test_metrics,
        "promoted": promoted,
        "promotion_failures": failures,
    }
    report_path = args.report_dir / f"training-{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint_path = args.checkpoint
    if not promoted:
        checkpoint_path = args.checkpoint.with_name(f"{args.checkpoint.stem}-candidate.pt")
    save_checkpoint(
        checkpoint_path,
        model,
        temperature=temperature,
        activation_scales=activation_scales,
        data_fingerprint=fingerprint,
        metrics={
            "validation": validation_metrics,
            "test": test_metrics,
            "refit": refit_details,
        },
    )
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved report: {report_path}")
    if promoted:
        print("Checkpoint meets all booth promotion thresholds")
    else:
        print("Candidate was not promoted:")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
