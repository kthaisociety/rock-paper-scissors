from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

PREPROCESS_VERSION = 1
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9
MIN_SCALE = 1e-4

LANDMARK_GROUPS: Mapping[str, tuple[int, ...]] = {
    "PALM": (0, 5, 9, 13, 17),
    "THUMB": (1, 2, 3, 4),
    "INDEX": (5, 6, 7, 8),
    "MIDDLE": (9, 10, 11, 12),
    "RING": (13, 14, 15, 16),
    "PINKY": (17, 18, 19, 20),
}


class InvalidLandmarksError(ValueError):
    """Raised when landmarks cannot produce a safe normalized feature vector."""


def _as_landmark_array(landmarks: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(landmarks, dtype=np.float32)
    if array.shape != (21, 3):
        raise InvalidLandmarksError(f"Expected landmarks with shape (21, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise InvalidLandmarksError("Landmarks contain non-finite values")
    return array.copy()


def preprocess_landmarks(
    landmarks: np.ndarray | Sequence[Sequence[float]], handedness: str
) -> np.ndarray:
    """Return wrist-, scale-, handedness-, and rotation-normalized 63 features."""

    points = _as_landmark_array(landmarks)
    points -= points[WRIST_INDEX]

    scale = float(np.linalg.norm(points[MIDDLE_MCP_INDEX]))
    if not math.isfinite(scale) or scale < MIN_SCALE:
        raise InvalidLandmarksError(f"Degenerate wrist-to-middle-MCP scale: {scale}")
    points /= scale

    if handedness.strip().lower().startswith("left"):
        points[:, 0] *= -1.0

    middle_vector = points[MIDDLE_MCP_INDEX, :2]
    current_angle = math.atan2(float(middle_vector[1]), float(middle_vector[0]))
    rotation_angle = -math.pi / 2.0 - current_angle
    cosine = math.cos(rotation_angle)
    sine = math.sin(rotation_angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rotation.T

    return points.reshape(63).astype(np.float32, copy=False)


def group_feature_indices() -> dict[str, np.ndarray]:
    """Map the six visual feature groups to flattened coordinate indices."""

    result: dict[str, np.ndarray] = {}
    for name, landmarks in LANDMARK_GROUPS.items():
        result[name] = np.asarray(
            [
                coordinate
                for landmark in landmarks
                for coordinate in range(3 * landmark, 3 * landmark + 3)
            ],
            dtype=np.int64,
        )
    return result


def summarize_feature_groups(features: np.ndarray) -> np.ndarray:
    flat = np.asarray(features, dtype=np.float32).reshape(63)
    return np.asarray(
        [float(np.mean(np.abs(flat[indices]))) for indices in group_feature_indices().values()],
        dtype=np.float32,
    )


def augment_features(features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply small geometry-preserving perturbations to normalized landmarks."""

    points = np.asarray(features, dtype=np.float32).reshape(21, 3).copy()
    angle = math.radians(float(rng.uniform(-15.0, 15.0)))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    points[:, :2] = points[:, :2] @ rotation.T
    points[:, :2] += rng.normal(0.0, 0.01, size=(21, 2)).astype(np.float32)
    points[:, 2] += rng.normal(0.0, 0.006, size=21).astype(np.float32)
    points[WRIST_INDEX] = 0.0
    return points.reshape(63)
