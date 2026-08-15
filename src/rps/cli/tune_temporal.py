from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from rps.checkpoint import CheckpointError, load_checkpoint
from rps.cli.train import set_seeds
from rps.constants import (
    DATA_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_TEMPORAL_POLICY_PATH,
    REPORTS_DIR,
    TEMPORAL_POLICY_CANDIDATE_PATH,
)
from rps.data import (
    LandmarkFrameDataset,
    ReviewManifestError,
    Trajectory,
    load_reviewed_trajectories,
    trajectories_for_participants,
)
from rps.metrics import trajectory_observations
from rps.model import GestureMLP
from rps.temporal import (
    TemporalPolicyArtifact,
    TemporalPolicyArtifactError,
    TemporalPolicyConfig,
    TemporalPolicyKind,
    checkpoint_sha256,
    load_temporal_policy,
    save_temporal_policy,
)
from rps.temporal_tuning import (
    TemporalTrace,
    benchmark_policy_overhead,
    confirmation_failures,
    score_policy,
    select_temporal_policy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune and confirm protocol-aware temporal locking"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune = subparsers.add_parser("tune", help="Generate out-of-fold traces and a candidate")
    _add_common_arguments(tune)
    tune.add_argument("--candidate", type=Path, default=TEMPORAL_POLICY_CANDIDATE_PATH)
    tune.add_argument("--report", type=Path)
    tune.add_argument("--fold-epochs", type=int, default=32)
    tune.add_argument("--seed", type=int, default=42)

    confirm = subparsers.add_parser(
        "confirm", help="Confirm a frozen candidate on fully reviewed fresh participants"
    )
    _add_common_arguments(confirm)
    confirm.add_argument("--candidate", type=Path, default=TEMPORAL_POLICY_CANDIDATE_PATH)
    confirm.add_argument("--participants", nargs="+", required=True)
    confirm.add_argument("--promoted", type=Path, default=DEFAULT_TEMPORAL_POLICY_PATH)
    confirm.add_argument("--report", type=Path)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)


def _write_report(path: Path | None, prefix: str, payload: dict[str, Any]) -> Path:
    report_path = path or REPORTS_DIR / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report_path


def _train_fold_model(
    trajectories: list[Trajectory],
    *,
    epochs: int,
    seed: int,
) -> GestureMLP:
    set_seeds(seed)
    dataset = LandmarkFrameDataset(trajectories, augment=True, seed=seed, sampling="fixed")
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        dataset.balanced_sample_weights(),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    model = GestureMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=160, eta_min=2e-5
    )
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features).logits, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return model.eval()


def _traces_for_model(
    model: GestureMLP,
    trajectories: list[Trajectory],
    *,
    temperature: float,
) -> list[TemporalTrace]:
    traces: list[TemporalTrace] = []
    for trajectory in trajectories:
        timestamps, probabilities, features = trajectory_observations(
            model, trajectory, temperature=temperature, device="cpu"
        )
        if len(timestamps):
            traces.append(
                TemporalTrace(
                    participant=trajectory.participant,
                    label=trajectory.label,
                    trajectory_id=trajectory.trajectory_id,
                    timestamps_ms=timestamps,
                    probabilities=probabilities,
                    features=features,
                )
            )
    return traces


def _load_data(args: argparse.Namespace) -> tuple[list[Trajectory], Path, dict[str, int]]:
    review_manifest = args.review_manifest or args.data / "review-manifest.json"
    try:
        trajectories, summary = load_reviewed_trajectories(args.data, review_manifest)
    except ReviewManifestError as error:
        raise SystemExit(str(error)) from error
    if not trajectories:
        raise SystemExit(f"No included trajectories found under {args.data}")
    return trajectories, review_manifest, summary.as_dict()


def _load_checkpoint(args: argparse.Namespace):
    try:
        return load_checkpoint(args.checkpoint, "cpu", allow_untrained=False)
    except CheckpointError as error:
        raise SystemExit(str(error)) from error


def _run_tune(args: argparse.Namespace) -> None:
    if args.fold_epochs < 1:
        raise SystemExit("--fold-epochs must be positive")
    trajectories, review_manifest, review_summary = _load_data(args)
    loaded = _load_checkpoint(args)
    participants = sorted({trajectory.participant for trajectory in trajectories})
    if len(participants) < 3:
        raise SystemExit("Temporal tuning requires at least three participants")

    traces: list[TemporalTrace] = []
    for fold_index, held_out in enumerate(participants, start=1):
        print(f"OOF fold {fold_index}/{len(participants)}: hold out {held_out}")
        training = [
            trajectory for trajectory in trajectories if trajectory.participant != held_out
        ]
        held_out_trajectories = trajectories_for_participants(trajectories, [held_out])
        model = _train_fold_model(training, epochs=args.fold_epochs, seed=args.seed)
        traces.extend(
            _traces_for_model(
                model,
                held_out_trajectories,
                temperature=loaded.temperature,
            )
        )

    baseline, winner, summaries = select_temporal_policy(traces)
    ranked = sorted(
        summaries,
        key=lambda item: (
            not item["eligible"],
            item["median_lock_time_ms"],
            item["p90_lock_time_ms"],
        ),
    )
    diagnostic_leaders = {
        "best_accuracy": max(summaries, key=lambda item: item["accuracy"]),
        "fastest_nonbaseline": min(
            (
                item
                for item in summaries
                if item["config"]["kind"] != TemporalPolicyKind.BASELINE.value
            ),
            key=lambda item: (item["median_lock_time_ms"], item["p90_lock_time_ms"]),
        ),
        "best_stability_gate": max(
            (
                item
                for item in summaries
                if item["config"]["kind"]
                == TemporalPolicyKind.STABILITY_GATE.value
            ),
            key=lambda item: item["accuracy"],
        ),
        "best_hmm": max(
            (
                item
                for item in summaries
                if item["config"]["kind"] == TemporalPolicyKind.HMM.value
            ),
            key=lambda item: item["accuracy"],
        ),
    }
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "participant-disjoint-oof-tuning",
        "participants": participants,
        "trajectory_count": len(traces),
        "fold_epochs": args.fold_epochs,
        "seed": args.seed,
        "temperature": loaded.temperature,
        "data_review": {"manifest": str(review_manifest), **review_summary},
        "baseline": baseline.as_dict(),
        "winner": winner.as_dict(),
        "searched_config_count": len(summaries),
        "eligible_nonbaseline_count": sum(
            item["eligible"]
            and item["config"]["kind"] != TemporalPolicyKind.BASELINE.value
            for item in summaries
        ),
        "diagnostic_leaders": diagnostic_leaders,
        "top_results": ranked[:20],
    }
    artifact = TemporalPolicyArtifact(
        config=winner.config,
        model_data_fingerprint=loaded.data_fingerprint,
        checkpoint_sha256=checkpoint_sha256(args.checkpoint),
        selection_metrics={
            "baseline": baseline.as_dict(),
            "winner": winner.as_dict(),
            "trajectory_count": len(traces),
            "participants": participants,
        },
        status=(
            "candidate"
            if winner.config.kind != TemporalPolicyKind.BASELINE
            else "baseline_retained"
        ),
    )
    save_temporal_policy(args.candidate, artifact)
    report_path = _write_report(args.report, "temporal-tuning", report)
    print(f"Policy result: {args.candidate}")
    print(f"Report: {report_path}")
    print(
        f"Selected {winner.config.kind.value}: accuracy={winner.accuracy:.3f}, "
        f"median={winner.median_lock_time_ms:.1f} ms, p90={winner.p90_lock_time_ms:.1f} ms"
    )
    if winner.config.kind == TemporalPolicyKind.BASELINE:
        print("No temporal candidate met every constraint; baseline retained")


def _require_fully_reviewed(
    participants: list[str], data_root: Path, review_manifest: Path
) -> None:
    payload = json.loads(review_manifest.read_text(encoding="utf-8"))
    reviewed_paths = set(payload.get("reviews", {}))
    participant_paths = [
        path
        for participant in participants
        for path in sorted((data_root / participant).rglob("*.npz"))
    ]
    missing: list[str] = []
    for path in participant_paths:
        relative = path.relative_to(data_root).as_posix()
        if relative not in reviewed_paths:
            missing.append(relative)
    if missing:
        preview = ", ".join(missing[:3])
        raise SystemExit(
            f"Fresh confirmation requires every selected trajectory to be reviewed; "
            f"{len(missing)} missing (for example {preview})"
        )


def _run_confirm(args: argparse.Namespace) -> None:
    if len(set(args.participants)) < 2:
        raise SystemExit("Confirmation requires at least two distinct fresh participants")
    trajectories, review_manifest, review_summary = _load_data(args)
    selected = trajectories_for_participants(trajectories, args.participants)
    present = {trajectory.participant for trajectory in selected}
    missing_participants = sorted(set(args.participants) - present)
    if missing_participants:
        raise SystemExit(f"No included captures for: {', '.join(missing_participants)}")
    _require_fully_reviewed(args.participants, args.data, review_manifest)
    class_counts = {
        participant: Counter(
            trajectory.label
            for trajectory in selected
            if trajectory.participant == participant
        )
        for participant in args.participants
    }
    insufficient = [
        f"{participant}/class-{label}={counts[label]}"
        for participant, counts in class_counts.items()
        for label in range(3)
        if counts[label] < 20
    ]
    if insufficient:
        raise SystemExit(
            "Fresh confirmation requires at least 20 included trajectories per class and "
            f"participant; insufficient: {', '.join(insufficient)}"
        )
    loaded = _load_checkpoint(args)
    try:
        artifact = load_temporal_policy(
            args.candidate,
            model_data_fingerprint=loaded.data_fingerprint,
            checkpoint_path=args.checkpoint,
        )
    except TemporalPolicyArtifactError as error:
        raise SystemExit(str(error)) from error
    if artifact.status != "candidate":
        raise SystemExit("Confirmation expects a frozen candidate artifact")

    traces = _traces_for_model(
        loaded.model, selected, temperature=loaded.temperature
    )
    baseline = score_policy(traces, TemporalPolicyConfig.baseline())
    candidate = score_policy(traces, artifact.config)
    overhead = benchmark_policy_overhead(traces, baseline.config, candidate.config)
    failures = confirmation_failures(
        baseline, candidate, runtime_overhead_p95_ms=overhead
    )
    confirmation = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": "fresh-confirmation",
        "participants": sorted(present),
        "trajectory_count": len(traces),
        "data_review": {"manifest": str(review_manifest), **review_summary},
        "baseline": baseline.as_dict(),
        "candidate": candidate.as_dict(),
        "runtime_overhead_p95_ms": overhead,
        "passed": not failures,
        "failures": failures,
    }
    report_path = _write_report(args.report, "temporal-confirmation", confirmation)
    print(f"Report: {report_path}")
    if failures:
        print("Temporal policy was not promoted:")
        for failure in failures:
            print(f"- {failure}")
        return
    promoted = TemporalPolicyArtifact(
        config=artifact.config,
        model_data_fingerprint=artifact.model_data_fingerprint,
        checkpoint_sha256=artifact.checkpoint_sha256,
        selection_metrics={
            **artifact.selection_metrics,
            "fresh_confirmation": confirmation,
        },
        status="promoted",
    )
    save_temporal_policy(args.promoted, promoted)
    print(f"Promoted temporal policy: {args.promoted}")


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "tune":
        _run_tune(args)
    else:
        _run_confirm(args)


if __name__ == "__main__":
    main()
