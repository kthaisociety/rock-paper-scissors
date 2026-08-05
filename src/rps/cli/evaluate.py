from __future__ import annotations

import argparse
import json
from pathlib import Path

from rps.checkpoint import CheckpointError, load_checkpoint
from rps.constants import DATA_DIR, DEFAULT_CHECKPOINT_PATH, REPORTS_DIR
from rps.data import discover_trajectories, trajectories_for_participants
from rps.device import resolve_device
from rps.metrics import checkpoint_passes_promotion, evaluate_trajectories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint by participant")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
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
    trajectories = trajectories_for_participants(discover_trajectories(args.data), participants)
    metrics = evaluate_trajectories(
        loaded.model,
        trajectories,
        temperature=loaded.temperature,
        device=loaded.device,
    )
    promoted, failures = checkpoint_passes_promotion(metrics)
    result = {
        "split": args.split,
        "participants": participants,
        "device": str(loaded.device),
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
