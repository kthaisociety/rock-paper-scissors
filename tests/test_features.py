from __future__ import annotations

import math

import numpy as np
import pytest

from rps.features import (
    InvalidLandmarksError,
    group_feature_indices,
    preprocess_landmarks,
    summarize_feature_groups,
)


def test_translation_and_scale_invariance(hand_landmarks: np.ndarray) -> None:
    expected = preprocess_landmarks(hand_landmarks, "Right")
    transformed = hand_landmarks * 3.7 + np.asarray([4.0, -2.0, 0.5], dtype=np.float32)
    actual = preprocess_landmarks(transformed, "Right")
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_rotation_invariance(hand_landmarks: np.ndarray) -> None:
    angle = math.radians(37)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float32,
    )
    rotated = hand_landmarks.copy()
    rotated[:, :2] = rotated[:, :2] @ rotation.T
    np.testing.assert_allclose(
        preprocess_landmarks(rotated, "Right"),
        preprocess_landmarks(hand_landmarks, "Right"),
        atol=1e-5,
    )


def test_handedness_canonicalization(hand_landmarks: np.ndarray) -> None:
    left = hand_landmarks.copy()
    left[:, 0] *= -1
    np.testing.assert_allclose(
        preprocess_landmarks(left, "Left"),
        preprocess_landmarks(hand_landmarks, "Right"),
        atol=1e-5,
    )


def test_degenerate_and_non_finite_landmarks_are_rejected() -> None:
    with pytest.raises(InvalidLandmarksError):
        preprocess_landmarks(np.zeros((21, 3), dtype=np.float32), "Right")
    invalid = np.zeros((21, 3), dtype=np.float32)
    invalid[9, 0] = np.nan
    with pytest.raises(InvalidLandmarksError):
        preprocess_landmarks(invalid, "Right")


def test_feature_groups_cover_valid_coordinates(hand_landmarks: np.ndarray) -> None:
    groups = group_feature_indices()
    assert list(groups) == ["PALM", "THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
    assert all(np.all((indices >= 0) & (indices < 63)) for indices in groups.values())
    summary = summarize_feature_groups(preprocess_landmarks(hand_landmarks, "Right"))
    assert summary.shape == (6,)
    assert np.isfinite(summary).all()
