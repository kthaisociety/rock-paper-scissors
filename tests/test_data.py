from __future__ import annotations

from pathlib import Path

import numpy as np

from rps.data import (
    LandmarkFrameDataset,
    Trajectory,
    load_trajectory,
    participant_split,
    save_trajectory,
    trajectory_samples,
)


def make_trajectory(
    hand_landmarks: np.ndarray,
    participant: str,
    label: int = 0,
) -> Trajectory:
    timestamps = np.asarray([250, 350, 450, 700, 800, 900], dtype=np.int64)
    landmarks = np.stack([hand_landmarks.copy() for _ in timestamps])
    if label == 1:
        landmarks[:, 4:9, 1] -= 0.3
    elif label == 2:
        landmarks[:, 8:13, 0] += 0.2
    return Trajectory(
        landmarks=landmarks,
        timestamps_ms=timestamps,
        handedness=np.asarray(["Right"] * len(timestamps)),
        label=label,
        participant=participant,
        session_id="session",
        trajectory_id=f"{participant}-{label}",
        metadata={"camera_index": 0},
    )


def test_trajectory_round_trip_contains_landmarks_only(
    tmp_path: Path, hand_landmarks: np.ndarray
) -> None:
    path = tmp_path / "trajectory.npz"
    expected = make_trajectory(hand_landmarks, "P01")
    save_trajectory(path, expected)
    actual = load_trajectory(path)
    np.testing.assert_allclose(actual.landmarks, expected.landmarks)
    assert actual.participant == "P01"
    assert not list(tmp_path.glob("*.jpg"))
    assert not list(tmp_path.glob("*.mp4"))


def test_participant_split_is_disjoint(hand_landmarks: np.ndarray) -> None:
    trajectories = [make_trajectory(hand_landmarks, f"P{index:02d}") for index in range(10)]
    split = participant_split(trajectories)
    train = set(split["train"])
    validation = set(split["validation"])
    test = set(split["test"])
    assert train and validation and test
    assert train.isdisjoint(validation | test)
    assert validation.isdisjoint(test)
    assert train | validation | test == {f"P{index:02d}" for index in range(10)}


def test_each_trajectory_contributes_six_balanced_samples(
    hand_landmarks: np.ndarray,
) -> None:
    trajectory = make_trajectory(hand_landmarks, "P01", label=2)
    features, labels, times = trajectory_samples(trajectory)
    assert features.shape == (6, 63)
    assert labels.tolist() == [2] * 6
    assert times.tolist() == [250, 350, 450, 700, 800, 900]
    dataset = LandmarkFrameDataset([trajectory], augment=False)
    assert len(dataset) == 6
