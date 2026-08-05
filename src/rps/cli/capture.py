from __future__ import annotations

import argparse
import random
import time
import uuid
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from rps.constants import DATA_DIR
from rps.data import Trajectory, save_trajectory
from rps.model import CLASS_NAMES
from rps.setup_assets import AssetError, ensure_hand_landmarker_asset
from rps.tracking import AsyncHandTracker

EARLY_WINDOW = (150, 450)
FINAL_WINDOW = (650, 950)


def generate_prompts(repetitions: int, seed: int) -> list[int]:
    remaining = Counter({index: repetitions for index in range(len(CLASS_NAMES))})
    prompts: list[int] = []
    rng = random.Random(seed)
    while sum(remaining.values()):
        choices = [label for label, count in remaining.items() if count > 0]
        if len(prompts) >= 2 and prompts[-1] == prompts[-2]:
            choices = [label for label in choices if label != prompts[-1]]
        rng.shuffle(choices)
        selected = max(choices, key=lambda label: remaining[label])
        prompts.append(selected)
        remaining[selected] -= 1
    return prompts


def _draw_capture_ui(
    frame: np.ndarray,
    *,
    label: str,
    trial: int,
    total: int,
    phase_text: str,
    frame_count: int,
) -> np.ndarray:
    display = cv2.flip(frame, 1)
    cv2.rectangle(display, (20, 20), (display.shape[1] - 20, 145), (12, 22, 30), -1)
    cv2.putText(
        display,
        f"TRAINING CAPTURE {trial}/{total}",
        (45, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (120, 230, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"MAKE: {label}",
        (45, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        f"{phase_text}   captured frames: {frame_count}",
        (45, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 210, 220),
        1,
        cv2.LINE_AA,
    )
    return display


def capture_trial(
    camera: cv2.VideoCapture,
    tracker: AsyncHandTracker,
    *,
    label: int,
    trial_number: int,
    total_trials: int,
    last_tracker_timestamp: int,
) -> tuple[list[np.ndarray], list[int], list[str], int, bool]:
    started_ms = int(time.monotonic_ns() / 1_000_000)
    go_ms = started_ms + 3500
    end_ms = go_ms + 1050
    landmarks: list[np.ndarray] = []
    timestamps: list[int] = []
    handedness: list[str] = []
    aborted = False

    while int(time.monotonic_ns() / 1_000_000) <= end_ms:
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Camera stopped returning frames")
        now_ms = int(time.monotonic_ns() / 1_000_000)
        tracker.submit(frame, now_ms)
        result = tracker.latest(after_timestamp_ms=last_tracker_timestamp)
        if result is not None:
            last_tracker_timestamp = result.timestamp_ms
            elapsed = result.timestamp_ms - go_ms
            if result.observation is not None and 0 <= elapsed <= 950:
                landmarks.append(result.observation.landmarks)
                timestamps.append(elapsed)
                handedness.append(result.observation.handedness)

        if now_ms < go_ms - 3000:
            phase = "START WITH A CLOSED FIST"
        elif now_ms < go_ms:
            phase = str(max(1, int(np.ceil((go_ms - now_ms) / 1000))))
        elif now_ms <= go_ms + 950:
            phase = "GO - DEPLOY AND HOLD"
        else:
            phase = "DONE"
        display = _draw_capture_ui(
            frame,
            label=CLASS_NAMES[label],
            trial=trial_number,
            total=total_trials,
            phase_text=phase,
            frame_count=len(timestamps),
        )
        cv2.imshow("RPS Training Capture", display)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), 27}:
            aborted = True
            break
    return landmarks, timestamps, handedness, last_tracker_timestamp, aborted


def _window_count(timestamps: list[int], window: tuple[int, int]) -> int:
    return sum(window[0] <= timestamp <= window[1] for timestamp in timestamps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture landmark-only gesture trajectories")
    parser.add_argument("--participant", required=True, help="Pseudonymous participant alias")
    parser.add_argument("--repetitions", type=int, default=20, help="Trials per gesture")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    participant = args.participant.strip()
    if not participant or any(character in participant for character in "/\\"):
        raise SystemExit("--participant must be a non-empty filename-safe alias")
    try:
        asset_path = ensure_hand_landmarker_asset()
    except AssetError as error:
        raise SystemExit(str(error)) from error

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    session_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output / participant / session_id
    prompts = generate_prompts(args.repetitions, args.seed)
    completed = 0
    last_tracker_timestamp = -1
    try:
        with AsyncHandTracker(asset_path) as tracker:
            while completed < len(prompts):
                label = prompts[completed]
                result = capture_trial(
                    camera,
                    tracker,
                    label=label,
                    trial_number=completed + 1,
                    total_trials=len(prompts),
                    last_tracker_timestamp=last_tracker_timestamp,
                )
                landmarks, timestamps, handedness, last_tracker_timestamp, aborted = result
                if aborted:
                    break
                if (
                    _window_count(timestamps, EARLY_WINDOW) < 4
                    or _window_count(timestamps, FINAL_WINDOW) < 4
                ):
                    print(
                        f"Repeating {CLASS_NAMES[label]}: insufficient early/final landmark frames"
                    )
                    continue

                trajectory_id = (
                    f"{completed:04d}-{CLASS_NAMES[label].lower()}-{uuid.uuid4().hex[:8]}"
                )
                trajectory = Trajectory(
                    landmarks=np.stack(landmarks).astype(np.float32),
                    timestamps_ms=np.asarray(timestamps, dtype=np.int64),
                    handedness=np.asarray(handedness, dtype="U16"),
                    label=label,
                    participant=participant,
                    session_id=session_id,
                    trajectory_id=trajectory_id,
                    metadata={
                        "camera_index": args.camera,
                        "protocol_version": 1,
                        "early_window_ms": list(EARLY_WINDOW),
                        "final_window_ms": list(FINAL_WINDOW),
                    },
                )
                save_trajectory(output_dir / f"{trajectory_id}.npz", trajectory)
                completed += 1
                print(f"Saved {completed}/{len(prompts)}: {CLASS_NAMES[label]}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Capture complete: {completed} trajectories in {output_dir}")


if __name__ == "__main__":
    main()
