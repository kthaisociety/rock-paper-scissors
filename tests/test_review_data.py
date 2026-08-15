from __future__ import annotations

import numpy as np
import torch

from rps.checkpoint import LoadedModel
from rps.cli.review_data import (
    ReviewItem,
    _decision_conflict,
    _review_entry,
    _suspicion,
    render_review_frame,
)
from rps.data import Trajectory
from rps.model import GestureMLP


def _trajectory(hand_landmarks: np.ndarray) -> Trajectory:
    timestamps = np.asarray([150, 450, 700, 900], dtype=np.int64)
    landmarks = np.stack([hand_landmarks.copy() for _ in timestamps])
    landmarks[:, :, 0] += 0.5
    landmarks[:, :, 1] += 0.5
    return Trajectory(
        landmarks=landmarks,
        timestamps_ms=timestamps,
        handedness=np.asarray(["Right"] * len(timestamps)),
        label=0,
        participant="P01",
        session_id="session",
        trajectory_id="trajectory",
        metadata={"camera_index": 0},
    )


def test_review_label_keeps_matching_prompt_and_relabels_mismatch(
    hand_landmarks: np.ndarray,
) -> None:
    trajectory = _trajectory(hand_landmarks)

    assert _review_entry(trajectory, "ROCK") == {
        "action": "keep",
        "prompt_label": "ROCK",
    }
    assert _review_entry(trajectory, "PAPER") == {
        "action": "relabel",
        "prompt_label": "ROCK",
        "observed_label": "PAPER",
    }


def test_review_renderer_is_landmark_only_and_marks_decision(
    hand_landmarks: np.ndarray,
) -> None:
    trajectory = _trajectory(hand_landmarks)
    item = ReviewItem("P01/session/trajectory.npz", trajectory)
    decision = {"action": "exclude", "prompt_label": "ROCK"}

    frame = render_review_frame(item, 2, decision, position=1, total=1)

    assert frame.shape == (720, 1280, 3)
    assert frame.dtype == np.uint8
    assert np.any(frame != frame[0, 0])


def test_suspicion_queue_flags_strong_final_disagreement(
    hand_landmarks: np.ndarray,
) -> None:
    trajectory = _trajectory(hand_landmarks)
    model = GestureMLP().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.fc3.bias[1] = 8.0
    loaded = LoadedModel(model=model, device=torch.device("cpu"), trained=True)

    flagged = _suspicion(ReviewItem("trajectory.npz", trajectory), loaded)

    assert flagged.suspicion_score > 2.0
    assert any("disagreement" in reason for reason in flagged.suspicion_reasons)
    assert flagged.model_analysis is not None
    assert flagged.model_analysis.final_prediction == 1
    conflict = _decision_conflict(
        flagged,
        {"action": "keep", "prompt_label": "ROCK"},
    )
    assert conflict is not None
    assert "model says PAPER" in conflict


def test_exclusion_never_conflicts_with_model(hand_landmarks: np.ndarray) -> None:
    trajectory = _trajectory(hand_landmarks)
    item = ReviewItem("trajectory.npz", trajectory)

    assert _decision_conflict(
        item,
        {"action": "exclude", "prompt_label": "ROCK"},
    ) is None
