from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from rps.data import TARGET_TIMES_MS, Trajectory
from rps.features import preprocess_landmarks
from rps.game import Gesture, counter_move, lock_from_probability_trace, score_round
from rps.model import CLASS_NAMES, GestureMLP, calibrated_probabilities


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for expected, predicted in zip(labels, predictions, strict=True):
        matrix[int(expected), int(predicted)] += 1
    return matrix


def classification_report(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    matrix = confusion_matrix(labels, predictions)
    classes: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(CLASS_NAMES):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        classes[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[index, :].sum()),
        }
    accuracy = float(np.trace(matrix) / max(matrix.sum(), 1))
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
    }


@torch.inference_mode()
def predict_features(
    model: GestureMLP,
    features: np.ndarray,
    *,
    temperature: float,
    device: torch.device | str,
) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0, 3), dtype=np.float32)
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    output = model(tensor)
    return calibrated_probabilities(output.logits, temperature).cpu().numpy()


def trajectory_probabilities(
    model: GestureMLP,
    trajectory: Trajectory,
    *,
    temperature: float,
    device: torch.device | str,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    timestamps: list[int] = []
    for landmarks, handedness, timestamp in zip(
        trajectory.landmarks,
        trajectory.handedness,
        trajectory.timestamps_ms,
        strict=True,
    ):
        try:
            features.append(preprocess_landmarks(landmarks, str(handedness)))
            timestamps.append(int(timestamp))
        except ValueError:
            continue
    if not features:
        return np.empty(0, dtype=np.int64), np.empty((0, 3), dtype=np.float32)
    probabilities = predict_features(
        model,
        np.stack(features),
        temperature=temperature,
        device=device,
    )
    return np.asarray(timestamps, dtype=np.int64), probabilities


def evaluate_trajectories(
    model: GestureMLP,
    trajectories: list[Trajectory],
    *,
    temperature: float,
    device: torch.device | str,
) -> dict[str, Any]:
    final_labels: list[int] = []
    final_predictions: list[int] = []
    lock_labels: list[int] = []
    lock_predictions: list[int] = []
    lock_times: list[int] = []
    outcomes: defaultdict[str, int] = defaultdict(int)
    timestamp_correct: dict[int, list[bool]] = {target: [] for target in TARGET_TIMES_MS}
    participant_correct: defaultdict[str, list[bool]] = defaultdict(list)

    for trajectory in trajectories:
        timestamps, probabilities = trajectory_probabilities(
            model, trajectory, temperature=temperature, device=device
        )
        if not len(timestamps):
            continue
        for target in TARGET_TIMES_MS:
            index = int(np.argmin(np.abs(timestamps - target)))
            timestamp_correct[target].append(
                int(np.argmax(probabilities[index])) == trajectory.label
            )

        final_mask = (timestamps >= 650) & (timestamps <= 950)
        if np.any(final_mask):
            final_prediction = int(np.argmax(np.mean(probabilities[final_mask], axis=0)))
            final_labels.append(trajectory.label)
            final_predictions.append(final_prediction)

        decision = lock_from_probability_trace(timestamps, probabilities)
        if decision.gesture is not None and decision.lock_time_ms is not None:
            prediction = int(decision.gesture)
            correct = prediction == trajectory.label
            lock_labels.append(trajectory.label)
            lock_predictions.append(prediction)
            lock_times.append(decision.lock_time_ms)
            participant_correct[trajectory.participant].append(correct)
            outcome = score_round(counter_move(Gesture(prediction)), Gesture(trajectory.label))
            outcomes[outcome.value] += 1

    final_report = classification_report(
        np.asarray(final_labels, dtype=np.int64), np.asarray(final_predictions, dtype=np.int64)
    )
    lock_report = classification_report(
        np.asarray(lock_labels, dtype=np.int64), np.asarray(lock_predictions, dtype=np.int64)
    )
    return {
        "final_pose": final_report,
        "forced_lock": lock_report,
        "accuracy_by_timestamp_ms": {
            str(target): float(np.mean(values)) if values else 0.0
            for target, values in timestamp_correct.items()
        },
        "median_lock_time_ms": float(np.median(lock_times)) if lock_times else None,
        "median_prediction_lead_ms": (float(950 - np.median(lock_times)) if lock_times else None),
        "outcomes": dict(outcomes),
        "per_participant_lock_accuracy": {
            participant: float(np.mean(values))
            for participant, values in participant_correct.items()
        },
        "trajectory_count": len(trajectories),
    }


def checkpoint_passes_promotion(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    final = metrics["final_pose"]
    lock = metrics["forced_lock"]
    if final["macro_f1"] < 0.95:
        failures.append("final macro F1 is below 0.95")
    for name, values in final["classes"].items():
        if values["recall"] < 0.90:
            failures.append(f"{name} final recall is below 0.90")
    if lock["accuracy"] < 0.85:
        failures.append("forced-lock accuracy is below 0.85")
    median_lock = metrics["median_lock_time_ms"]
    if median_lock is None or median_lock > 450:
        failures.append("median lock time is later than 450 ms")
    for participant, accuracy in metrics["per_participant_lock_accuracy"].items():
        if accuracy < 0.70:
            failures.append(f"participant {participant} lock accuracy is below 0.70")
    return not failures, failures
