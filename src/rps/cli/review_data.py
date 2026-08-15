from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rps.checkpoint import CheckpointError, LoadedModel, load_checkpoint
from rps.constants import DATA_DIR, DEFAULT_CHECKPOINT_PATH
from rps.data import (
    ReviewManifestError,
    Trajectory,
    load_review_manifest,
    load_reviewed_trajectories,
    load_trajectory,
    save_review_manifest,
)
from rps.landmark_drawing import HAND_CONNECTIONS
from rps.metrics import trajectory_probabilities
from rps.model import CLASS_NAMES

WINDOW_NAME = "RPS Landmark Review"
CANVAS_SIZE = (720, 1280)
LEFT_KEYS = {2, 81, 63234, 2424832}
RIGHT_KEYS = {3, 83, 63235, 2555904}


@dataclass(frozen=True, slots=True)
class ModelReviewAnalysis:
    timestamps_ms: np.ndarray
    probabilities: np.ndarray
    final_probabilities: np.ndarray
    final_prediction: int | None
    final_confidence: float
    final_consistency: float


@dataclass(frozen=True, slots=True)
class ReviewItem:
    relative_path: str
    trajectory: Trajectory
    suspicion_score: float = 0.0
    suspicion_reasons: tuple[str, ...] = ()
    model_analysis: ModelReviewAnalysis | None = None


def _put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (220, 230, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _decision_text(decision: dict[str, Any] | None) -> str:
    if decision is None:
        return "UNREVIEWED"
    action = str(decision["action"])
    if action == "relabel":
        return f"RELABEL -> {decision['observed_label']}"
    return action.upper()


def _decision_conflict(item: ReviewItem, decision: dict[str, Any] | None) -> str | None:
    analysis = item.model_analysis
    if analysis is None or decision is None or decision["action"] == "exclude":
        return None
    if analysis.final_prediction is None or analysis.final_confidence < 0.80:
        return None
    expected_name = (
        str(decision["observed_label"])
        if decision["action"] == "relabel"
        else str(decision["prompt_label"])
    )
    predicted_name = CLASS_NAMES[analysis.final_prediction]
    if predicted_name == expected_name:
        return None
    return (
        f"CHECK REVIEW: model says {predicted_name} "
        f"({analysis.final_confidence:.0%}), review says {expected_name}"
    )


def _draw_probability_bars(
    canvas: np.ndarray, probabilities: np.ndarray, *, top: int
) -> None:
    left, width = 28, 350
    for index, name in enumerate(CLASS_NAMES):
        y = top + index * 27
        probability = float(probabilities[index])
        _put_text(canvas, name, (left, y + 15), scale=0.38, color=(175, 195, 205))
        cv2.rectangle(canvas, (left + 82, y), (left + 82 + width, y + 16), (42, 53, 60), -1)
        cv2.rectangle(
            canvas,
            (left + 82, y),
            (left + 82 + int(width * probability), y + 16),
            (75, 205, 145),
            -1,
        )
        _put_text(canvas, f"{probability:.0%}", (left + 442, y + 15), scale=0.38)


def _project_trajectory(trajectory: Trajectory) -> np.ndarray:
    points = trajectory.landmarks[:, :, :2].astype(np.float32, copy=True)
    points[:, :, 0] = 1.0 - points[:, :, 0]
    minimum = points.min(axis=(0, 1))
    maximum = points.max(axis=(0, 1))
    extent = np.maximum(maximum - minimum, 1e-4)
    area_left, area_top, area_width, area_height = 500, 120, 700, 500
    scale = min(area_width / float(extent[0]), area_height / float(extent[1]))
    center = (minimum + maximum) / 2.0
    target = np.asarray(
        [area_left + area_width / 2.0, area_top + area_height / 2.0], dtype=np.float32
    )
    return (points - center) * scale + target


def render_review_frame(
    item: ReviewItem,
    frame_index: int,
    decision: dict[str, Any] | None,
    *,
    position: int,
    total: int,
    paused: bool = False,
) -> np.ndarray:
    """Render one landmark-only review frame without opening a window."""

    canvas = np.full((*CANVAS_SIZE, 3), (10, 17, 23), dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (1280, 88), (16, 32, 42), -1)
    _put_text(
        canvas,
        "LANDMARK TRAJECTORY REVIEW",
        (28, 40),
        scale=0.8,
        color=(110, 235, 225),
        thickness=2,
    )
    _put_text(canvas, f"{position}/{total}", (1110, 40), scale=0.65)

    trajectory = item.trajectory
    timestamp = int(trajectory.timestamps_ms[frame_index])
    prompt = CLASS_NAMES[trajectory.label]
    _put_text(canvas, f"Participant: {trajectory.participant}", (28, 125))
    _put_text(canvas, f"Session: {trajectory.session_id}", (28, 153), scale=0.48)
    _put_text(canvas, item.relative_path, (28, 181), scale=0.39, color=(145, 165, 175))
    _put_text(canvas, f"PROMPT: {prompt}", (28, 235), scale=0.9, thickness=2)
    _put_text(
        canvas,
        f"DECISION: {_decision_text(decision)}",
        (28, 278),
        scale=0.65,
        color=(100, 230, 145) if decision is not None else (100, 190, 255),
        thickness=2,
    )
    window_name = "FINAL REVIEW WINDOW" if 650 <= timestamp <= 950 else "DEPLOYMENT"
    _put_text(canvas, f"{timestamp:>4} ms  {window_name}", (28, 330), scale=0.58)
    if paused:
        _put_text(canvas, "PAUSED", (28, 363), color=(100, 190, 255), thickness=2)

    analysis = item.model_analysis
    if analysis is not None and analysis.final_prediction is not None:
        nearest = int(np.argmin(np.abs(analysis.timestamps_ms - timestamp)))
        frame_probabilities = analysis.probabilities[nearest]
        frame_prediction = int(np.argmax(frame_probabilities))
        _put_text(
            canvas,
            f"MODEL FRAME: {CLASS_NAMES[frame_prediction]} "
            f"{float(frame_probabilities[frame_prediction]):.0%}",
            (28, 390),
            scale=0.48,
            color=(120, 220, 190),
            thickness=2,
        )
        _put_text(
            canvas,
            f"FINAL AVG: {CLASS_NAMES[analysis.final_prediction]} "
            f"{analysis.final_confidence:.0%}  stability {analysis.final_consistency:.0%}",
            (28, 418),
            scale=0.45,
            color=(120, 220, 190),
        )
        _draw_probability_bars(canvas, analysis.final_probabilities, top=432)
    else:
        _put_text(canvas, "MODEL ASSISTANCE UNAVAILABLE", (28, 405), color=(110, 160, 190))

    conflict = _decision_conflict(item, decision)
    if conflict is not None:
        _put_text(canvas, conflict, (28, 535), scale=0.43, color=(80, 120, 255), thickness=2)
        _put_text(
            canvas,
            "Review saved; press RIGHT to accept or choose another label",
            (28, 558),
            scale=0.38,
            color=(80, 150, 255),
        )
    elif item.suspicion_reasons:
        _put_text(
            canvas,
            f"CHECK: {item.suspicion_reasons[0]}",
            (28, 535),
            scale=0.43,
            color=(90, 180, 255),
            thickness=2,
        )

    _put_text(canvas, "R / P / S   observed pose", (28, 575), scale=0.5)
    _put_text(canvas, "K keep prompt    X exclude", (28, 605), scale=0.5)
    _put_text(canvas, "U clear review   arrows navigate", (28, 635), scale=0.5)
    _put_text(canvas, "SPACE pause      Q quit", (28, 665), scale=0.5)

    projected = _project_trajectory(trajectory)
    frame_points = projected[frame_index].astype(np.int32)
    in_final_window = 650 <= timestamp <= 950
    line_color = (80, 210, 140) if in_final_window else (80, 175, 220)
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            canvas,
            tuple(frame_points[start]),
            tuple(frame_points[end]),
            line_color,
            4,
            cv2.LINE_AA,
        )
    for point in frame_points:
        cv2.circle(canvas, tuple(point), 7, (220, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(point), 9, line_color, 2, cv2.LINE_AA)

    timeline_left, timeline_right, timeline_y = 510, 1190, 670
    cv2.line(canvas, (timeline_left, timeline_y), (timeline_right, timeline_y), (70, 80, 90), 8)
    final_left = timeline_left + int((timeline_right - timeline_left) * 650 / 950)
    cv2.line(canvas, (final_left, timeline_y), (timeline_right, timeline_y), (45, 120, 75), 8)
    marker_x = timeline_left + int(
        (timeline_right - timeline_left) * np.clip(timestamp, 0, 950) / 950
    )
    cv2.circle(canvas, (marker_x, timeline_y), 10, (235, 245, 245), -1, cv2.LINE_AA)
    _put_text(canvas, "GO", (timeline_left - 12, 702), scale=0.4)
    _put_text(canvas, "650", (final_left - 18, 702), scale=0.4)
    _put_text(canvas, "950 ms", (timeline_right - 45, 702), scale=0.4)
    return canvas


def _review_entry(trajectory: Trajectory, observed_label: str | None) -> dict[str, str]:
    prompt_label = CLASS_NAMES[trajectory.label]
    if observed_label is None or observed_label == prompt_label:
        return {"action": "keep", "prompt_label": prompt_label}
    return {
        "action": "relabel",
        "prompt_label": prompt_label,
        "observed_label": observed_label,
    }


def _suspicion(item: ReviewItem, loaded: LoadedModel) -> ReviewItem:
    trajectory = item.trajectory
    timestamps, probabilities = trajectory_probabilities(
        loaded.model,
        trajectory,
        temperature=loaded.temperature,
        device=loaded.device,
    )
    reasons: list[str] = []
    score = 0.0
    final_mask = (timestamps >= 650) & (timestamps <= 950)
    final_mean = np.zeros(len(CLASS_NAMES), dtype=np.float32)
    predicted: int | None = None
    confidence = 0.0
    consistency = 0.0
    if np.any(final_mask):
        final_probabilities = probabilities[final_mask]
        final_mean = np.mean(final_probabilities, axis=0)
        predicted = int(np.argmax(final_mean))
        confidence = float(final_mean[predicted])
        consistency = float(np.mean(np.argmax(final_probabilities, axis=1) == predicted))
        if predicted != trajectory.label and confidence >= 0.80:
            score += 2.0 + confidence
            reasons.append(f"strong final-pose disagreement ({confidence:.0%})")
        if consistency < 0.75:
            score += 1.0 - consistency
            reasons.append(f"unstable final predictions ({consistency:.0%} agreement)")
    transitions = sum(
        left != right
        for left, right in zip(
            trajectory.handedness[:-1], trajectory.handedness[1:], strict=True
        )
    )
    if transitions >= 2:
        score += min(transitions / 10.0, 0.5)
        reasons.append(f"{transitions} handedness changes")
    analysis = ModelReviewAnalysis(
        timestamps_ms=timestamps,
        probabilities=probabilities,
        final_probabilities=final_mean,
        final_prediction=predicted,
        final_confidence=confidence,
        final_consistency=consistency,
    )
    return ReviewItem(
        relative_path=item.relative_path,
        trajectory=trajectory,
        suspicion_score=score,
        suspicion_reasons=tuple(reasons),
        model_analysis=analysis,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review landmark-only gesture trajectories")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--participant")
    parser.add_argument("--session")
    parser.add_argument("--all", action="store_true", help="Include already reviewed items")
    parser.add_argument("--suspicious-first", action="store_true")
    parser.add_argument("--suspicious-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--no-model", action="store_true", help="Disable model assistance")
    parser.add_argument("--speed", type=float, default=0.75, help="Playback speed multiplier")
    return parser


def _load_items(args: argparse.Namespace, reviews: dict[str, Any]) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for path in sorted(args.data.rglob("*.npz")):
        relative_path = path.relative_to(args.data).as_posix()
        trajectory = load_trajectory(path)
        if args.participant and trajectory.participant != args.participant:
            continue
        if args.session and trajectory.session_id != args.session:
            continue
        if not args.all and relative_path in reviews:
            continue
        items.append(ReviewItem(relative_path, trajectory))

    if args.no_model and (args.suspicious_first or args.suspicious_only):
        raise SystemExit("--suspicious-first/--suspicious-only require model assistance")
    if not args.no_model:
        try:
            loaded = load_checkpoint(args.checkpoint, "cpu", allow_untrained=False)
        except CheckpointError as error:
            if args.suspicious_first or args.suspicious_only:
                raise SystemExit(str(error)) from error
            print(f"Model assistance disabled: {error}")
        else:
            print(
                f"Model assistance: {args.checkpoint} "
                f"(temperature {loaded.temperature:.2f})"
            )
            items = [_suspicion(item, loaded) for item in items]
        if args.suspicious_only:
            items = [item for item in items if item.suspicion_score > 0]
    if args.suspicious_first or args.suspicious_only:
        items.sort(key=lambda item: (-item.suspicion_score, item.relative_path))
    return items


def main() -> None:
    args = build_parser().parse_args()
    if args.speed <= 0:
        raise SystemExit("--speed must be positive")
    manifest_path = args.manifest or args.data / "review-manifest.json"
    try:
        manifest = load_review_manifest(manifest_path)
        load_reviewed_trajectories(args.data, manifest_path)
    except ReviewManifestError as error:
        raise SystemExit(str(error)) from error
    reviews: dict[str, Any] = manifest["reviews"]
    items = _load_items(args, reviews)
    if not items:
        print("No matching trajectories need review")
        return

    index = 0
    paused = False
    playback_origin_ms = time.monotonic_ns() / 1_000_000
    paused_timestamp = 0
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)
        while 0 <= index < len(items):
            item = items[index]
            timestamps = item.trajectory.timestamps_ms
            if paused:
                playback_ms = paused_timestamp
            else:
                elapsed = (time.monotonic_ns() / 1_000_000 - playback_origin_ms) * args.speed
                duration = max(951, int(timestamps[-1]) + 1)
                playback_ms = int(elapsed % duration)
            frame_index = int(np.searchsorted(timestamps, playback_ms, side="right") - 1)
            frame_index = int(np.clip(frame_index, 0, len(timestamps) - 1))
            decision = reviews.get(item.relative_path)
            canvas = render_review_frame(
                item,
                frame_index,
                decision,
                position=index + 1,
                total=len(items),
                paused=paused,
            )
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKeyEx(20)
            if key < 0:
                continue
            low_key = key & 0xFF
            if low_key in {ord("q"), 27}:
                break
            if low_key == ord(" "):
                paused_timestamp = playback_ms
                paused = not paused
                if not paused:
                    playback_origin_ms = (
                        time.monotonic_ns() / 1_000_000 - paused_timestamp / args.speed
                    )
                continue
            if key in LEFT_KEYS:
                index = max(0, index - 1)
                playback_origin_ms = time.monotonic_ns() / 1_000_000
                continue
            if key in RIGHT_KEYS:
                index = min(len(items) - 1, index + 1)
                playback_origin_ms = time.monotonic_ns() / 1_000_000
                continue
            if low_key == ord("u"):
                reviews.pop(item.relative_path, None)
                save_review_manifest(manifest_path, manifest)
                continue

            observed = {ord("r"): "ROCK", ord("p"): "PAPER", ord("s"): "SCISSORS"}.get(
                low_key
            )
            if observed is not None or low_key in {ord("k"), ord("x")}:
                if low_key == ord("x"):
                    reviews[item.relative_path] = {
                        "action": "exclude",
                        "prompt_label": CLASS_NAMES[item.trajectory.label],
                    }
                else:
                    reviews[item.relative_path] = _review_entry(item.trajectory, observed)
                save_review_manifest(manifest_path, manifest)
                print(f"{item.relative_path}: {_decision_text(reviews[item.relative_path])}")
                conflict = _decision_conflict(item, reviews[item.relative_path])
                if conflict is not None:
                    print(f"  {conflict}")
                    paused = True
                    paused_timestamp = playback_ms
                    continue
                if index == len(items) - 1:
                    break
                index += 1
                playback_origin_ms = time.monotonic_ns() / 1_000_000
    except cv2.error as error:
        raise SystemExit(f"Could not open the OpenCV review window: {error}") from error
    finally:
        cv2.destroyAllWindows()
    print(f"Review manifest: {manifest_path} ({len(reviews)} decisions)")


if __name__ == "__main__":
    main()
