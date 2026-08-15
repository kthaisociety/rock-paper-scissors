from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rps.data import (
    LandmarkFrameDataset,
    ReviewManifestError,
    Trajectory,
    dataset_fingerprint,
    empty_review_manifest,
    load_reviewed_trajectories,
    load_trajectory,
    participant_split,
    save_review_manifest,
    save_trajectory,
    trajectory_samples,
    trajectory_window_samples,
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


def test_all_frame_sampling_uses_training_windows_and_balances_groups(
    hand_landmarks: np.ndarray,
) -> None:
    first = make_trajectory(hand_landmarks, "P01", label=0)
    first.timestamps_ms = np.asarray([100, 150, 300, 450, 550, 650], dtype=np.int64)
    first.landmarks = np.stack([hand_landmarks.copy() for _ in first.timestamps_ms])
    first.handedness = np.asarray(["Right"] * len(first.timestamps_ms))
    second = make_trajectory(hand_landmarks, "P02", label=1)

    features, labels, times = trajectory_window_samples(first)
    assert features.shape == (4, 63)
    assert labels.tolist() == [0] * 4
    assert times.tolist() == [150, 300, 450, 650]

    dataset = LandmarkFrameDataset([first, second], sampling="all")
    weights = dataset.balanced_sample_weights()
    group_mass: dict[tuple[str, int, int], float] = {}
    for participant, label, phase, weight in zip(
        dataset.participants,
        dataset.labels,
        dataset.phases,
        weights,
        strict=True,
    ):
        key = (str(participant), int(label), int(phase))
        group_mass[key] = group_mass.get(key, 0.0) + float(weight)
    np.testing.assert_allclose(list(group_mass.values()), list(group_mass.values())[0])


def test_review_manifest_relabels_and_excludes_without_mutating_raw_data(
    tmp_path: Path, hand_landmarks: np.ndarray
) -> None:
    rock_path = tmp_path / "P01" / "session" / "rock.npz"
    paper_path = tmp_path / "P02" / "session" / "paper.npz"
    save_trajectory(rock_path, make_trajectory(hand_landmarks, "P01", label=0))
    save_trajectory(paper_path, make_trajectory(hand_landmarks, "P02", label=1))
    manifest_path = tmp_path / "review-manifest.json"
    manifest = empty_review_manifest()
    manifest["reviews"] = {
        "P01/session/rock.npz": {
            "action": "relabel",
            "prompt_label": "ROCK",
            "observed_label": "SCISSORS",
        },
        "P02/session/paper.npz": {
            "action": "exclude",
            "prompt_label": "PAPER",
        },
    }
    save_review_manifest(manifest_path, manifest)

    trajectories, summary = load_reviewed_trajectories(tmp_path, manifest_path)

    assert len(trajectories) == 1
    assert trajectories[0].label == 2
    assert trajectories[0].metadata["prompt_label"] == "ROCK"
    assert summary.as_dict() == {
        "total": 2,
        "included": 1,
        "unreviewed": 0,
        "kept": 0,
        "relabeled": 1,
        "excluded": 1,
    }
    assert load_trajectory(rock_path).label == 0


def test_review_manifest_rejects_stale_paths(tmp_path: Path) -> None:
    manifest = empty_review_manifest()
    manifest["reviews"] = {
        "missing.npz": {"action": "keep", "prompt_label": "ROCK"}
    }
    manifest_path = tmp_path / "review-manifest.json"
    save_review_manifest(manifest_path, manifest)

    with pytest.raises(ReviewManifestError, match="missing trajectories"):
        load_reviewed_trajectories(tmp_path, manifest_path)


def test_dataset_fingerprint_changes_with_review_manifest(
    tmp_path: Path, hand_landmarks: np.ndarray
) -> None:
    path = tmp_path / "trajectory.npz"
    save_trajectory(path, make_trajectory(hand_landmarks, "P01"))
    manifest_path = tmp_path / "review-manifest.json"
    before = dataset_fingerprint([path], manifest_path)
    manifest = empty_review_manifest()
    manifest["reviews"] = {
        "trajectory.npz": {"action": "keep", "prompt_label": "ROCK"}
    }
    save_review_manifest(manifest_path, manifest)

    assert dataset_fingerprint([path], manifest_path) != before
