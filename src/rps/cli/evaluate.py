from __future__ import annotations

import argparse
import json
from pathlib import Path

from rps.checkpoint import CheckpointError, load_checkpoint
from rps.constants import DATA_DIR, DEFAULT_CHECKPOINT_PATH, REPORTS_DIR
from rps.data import (
    ReviewManifestError,
    load_reviewed_trajectories,
    trajectories_for_participants,
)
from rps.device import resolve_device
from rps.metrics import checkpoint_passes_promotion, evaluate_trajectories
from rps.temporal import TemporalPolicyArtifactError, load_temporal_policy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint by participant")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument(
        "--temporal-policy",
        type=Path,
        help="Evaluate a matching candidate or promoted temporal policy artifact",
    )
    parser.add_argument("--manifest", type=Path, default=REPORTS_DIR / "split-manifest.json")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        cpu_loaded = load_checkpoint(args.checkpoint, "cpu", allow_untrained=False)
        device = resolve_device(args.device, cpu_loaded.model)
        loaded = load_checkpoint(args.checkpoint, device, allow_untrained=False)
    except (CheckpointError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    if not args.manifest.exists():
        raise SystemExit(f"Split manifest not found: {args.manifest}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    participants = manifest["participants"][args.split]
    review_manifest_path = args.review_manifest or args.data / "review-manifest.json"
    try:
        reviewed, review_summary = load_reviewed_trajectories(
            args.data, review_manifest_path
        )
    except ReviewManifestError as error:
        raise SystemExit(str(error)) from error
    trajectories = trajectories_for_participants(reviewed, participants)
    temporal_config = None
    temporal_policy_details = {"kind": "baseline", "artifact": None}
    if args.temporal_policy is not None:
        try:
            artifact = load_temporal_policy(
                args.temporal_policy,
                model_data_fingerprint=loaded.data_fingerprint,
                checkpoint_path=args.checkpoint,
            )
        except TemporalPolicyArtifactError as error:
            raise SystemExit(str(error)) from error
        temporal_config = artifact.config
        temporal_policy_details = {
            "kind": artifact.config.kind.value,
            "artifact": str(args.temporal_policy),
            "status": artifact.status,
        }
    metrics = evaluate_trajectories(
        loaded.model,
        trajectories,
        temperature=loaded.temperature,
        device=loaded.device,
        temporal_config=temporal_config,
    )
    promoted, failures = checkpoint_passes_promotion(metrics)
    result = {
        "split": args.split,
        "participants": participants,
        "device": str(loaded.device),
        "temporal_policy": temporal_policy_details,
        "data_review": {
            "manifest": str(review_manifest_path),
            **review_summary.as_dict(),
        },
        "metrics": metrics,
        "meets_promotion_thresholds": promoted,
        "promotion_failures": failures,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
