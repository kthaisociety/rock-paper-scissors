from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from rps.features import augment_features, preprocess_landmarks
from rps.model import CLASS_NAMES

TARGET_TIMES_MS = (250, 350, 450, 700, 800, 900)
TRAINING_WINDOWS_MS = ((150, 450, 0), (650, 950, 1))
REVIEW_MANIFEST_FORMAT_VERSION = 1
REVIEW_ACTIONS = frozenset({"keep", "relabel", "exclude"})


class ReviewManifestError(ValueError):
    """Raised when a review manifest cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    total: int
    included: int
    unreviewed: int
    kept: int
    relabeled: int
    excluded: int

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "included": self.included,
            "unreviewed": self.unreviewed,
            "kept": self.kept,
            "relabeled": self.relabeled,
            "excluded": self.excluded,
        }


@dataclass(slots=True)
class Trajectory:
    landmarks: np.ndarray
    timestamps_ms: np.ndarray
    handedness: np.ndarray
    label: int
    participant: str
    session_id: str
    trajectory_id: str
    metadata: dict[str, Any]
    path: Path | None = None

    def validate(self) -> None:
        frame_count = len(self.timestamps_ms)
        if self.landmarks.shape != (frame_count, 21, 3):
            raise ValueError("Landmark trajectory must have shape (frames, 21, 3)")
        if self.handedness.shape != (frame_count,):
            raise ValueError("Handedness must contain one value per frame")
        if not 0 <= self.label < len(CLASS_NAMES):
            raise ValueError(f"Invalid gesture label: {self.label}")
        if frame_count and np.any(np.diff(self.timestamps_ms) < 0):
            raise ValueError("Trajectory timestamps must be monotonically increasing")
        if not self.participant.strip():
            raise ValueError("Participant alias cannot be empty")


def save_trajectory(path: Path, trajectory: Trajectory) -> None:
    trajectory.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(
            temporary_path,
            landmarks=trajectory.landmarks.astype(np.float32),
            timestamps_ms=trajectory.timestamps_ms.astype(np.int64),
            handedness=trajectory.handedness.astype("U16"),
            label=np.asarray(trajectory.label, dtype=np.int64),
            participant=np.asarray(trajectory.participant),
            session_id=np.asarray(trajectory.session_id),
            trajectory_id=np.asarray(trajectory.trajectory_id),
            metadata_json=np.asarray(json.dumps(trajectory.metadata, sort_keys=True)),
        )
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_trajectory(path: Path) -> Trajectory:
    with np.load(path, allow_pickle=False) as payload:
        trajectory = Trajectory(
            landmarks=payload["landmarks"].astype(np.float32),
            timestamps_ms=payload["timestamps_ms"].astype(np.int64),
            handedness=payload["handedness"].astype("U16"),
            label=int(payload["label"]),
            participant=str(payload["participant"]),
            session_id=str(payload["session_id"]),
            trajectory_id=str(payload["trajectory_id"]),
            metadata=json.loads(str(payload["metadata_json"])),
            path=path,
        )
    trajectory.validate()
    return trajectory


def empty_review_manifest() -> dict[str, Any]:
    return {"format_version": REVIEW_MANIFEST_FORMAT_VERSION, "reviews": {}}


def _validate_review_manifest(payload: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ReviewManifestError("Review manifest must be a JSON object")
    if payload.get("format_version") != REVIEW_MANIFEST_FORMAT_VERSION:
        raise ReviewManifestError("Unsupported review manifest format version")
    reviews = payload.get("reviews")
    if not isinstance(reviews, dict):
        raise ReviewManifestError("Review manifest 'reviews' must be an object")
    for relative_path, decision in reviews.items():
        path = Path(relative_path)
        if (
            not isinstance(relative_path, str)
            or path.is_absolute()
            or ".." in path.parts
            or path.suffix != ".npz"
        ):
            raise ReviewManifestError(
                f"Invalid trajectory path in review manifest: {relative_path}"
            )
        if not isinstance(decision, dict) or decision.get("action") not in REVIEW_ACTIONS:
            raise ReviewManifestError(f"Invalid review action for {relative_path}")
        prompt_label = decision.get("prompt_label")
        if prompt_label not in CLASS_NAMES:
            raise ReviewManifestError(f"Invalid prompt label for {relative_path}")
        observed_label = decision.get("observed_label")
        if decision["action"] == "relabel" and observed_label not in CLASS_NAMES:
            raise ReviewManifestError(f"Relabel decision lacks an observed label: {relative_path}")
    return reviews


def load_review_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_review_manifest()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewManifestError(f"Could not read review manifest {path}: {error}") from error
    _validate_review_manifest(payload)
    return payload


def save_review_manifest(path: Path, payload: dict[str, Any]) -> None:
    _validate_review_manifest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_reviewed_trajectories(
    data_dir: Path, review_manifest_path: Path | None = None
) -> tuple[list[Trajectory], ReviewSummary]:
    paths = sorted(data_dir.rglob("*.npz"))
    relative_paths = {path.relative_to(data_dir).as_posix(): path for path in paths}
    manifest_path = review_manifest_path or data_dir / "review-manifest.json"
    reviews = _validate_review_manifest(load_review_manifest(manifest_path))
    stale_paths = sorted(set(reviews) - set(relative_paths))
    if stale_paths:
        preview = ", ".join(stale_paths[:3])
        suffix = "..." if len(stale_paths) > 3 else ""
        raise ReviewManifestError(
            f"Review manifest references missing trajectories: {preview}{suffix}"
        )

    included: list[Trajectory] = []
    counts = {"unreviewed": 0, "kept": 0, "relabeled": 0, "excluded": 0}
    for relative_path, path in relative_paths.items():
        trajectory = load_trajectory(path)
        decision = reviews.get(relative_path)
        if decision is None:
            counts["unreviewed"] += 1
            included.append(trajectory)
            continue
        prompt_label = CLASS_NAMES[trajectory.label]
        if decision["prompt_label"] != prompt_label:
            raise ReviewManifestError(
                f"Prompt label mismatch for {relative_path}: data has {prompt_label}, "
                f"manifest has {decision['prompt_label']}"
            )
        action = decision["action"]
        count_key = {"keep": "kept", "relabel": "relabeled", "exclude": "excluded"}[action]
        counts[count_key] += 1
        if action == "exclude":
            continue
        if action == "relabel":
            observed_label = str(decision["observed_label"])
            metadata = {
                **trajectory.metadata,
                "prompt_label": prompt_label,
                "observed_label": observed_label,
            }
            trajectory = replace(
                trajectory,
                label=CLASS_NAMES.index(observed_label),
                metadata=metadata,
            )
        included.append(trajectory)

    summary = ReviewSummary(
        total=len(paths),
        included=len(included),
        unreviewed=counts["unreviewed"],
        kept=counts["kept"],
        relabeled=counts["relabeled"],
        excluded=counts["excluded"],
    )
    return included, summary


def discover_trajectories(data_dir: Path) -> list[Trajectory]:
    trajectories, _ = load_reviewed_trajectories(data_dir)
    return trajectories


def dataset_fingerprint(paths: list[Path], review_manifest_path: Path | None = None) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    if review_manifest_path is not None and review_manifest_path.exists():
        digest.update(b"review-manifest")
        digest.update(review_manifest_path.read_bytes())
    return digest.hexdigest()


def participant_split(trajectories: list[Trajectory], seed: int = 42) -> dict[str, list[str]]:
    participants = sorted({trajectory.participant for trajectory in trajectories})
    if len(participants) < 3:
        raise ValueError(
            "At least three participants are required for train/validation/test splits"
        )
    random.Random(seed).shuffle(participants)
    validation_count = max(1, round(len(participants) * 0.15))
    test_count = max(1, round(len(participants) * 0.15))
    train_count = len(participants) - validation_count - test_count
    if train_count < 1:
        raise ValueError("Participant split would have no training participants")
    return {
        "train": sorted(participants[:train_count]),
        "validation": sorted(participants[train_count : train_count + validation_count]),
        "test": sorted(participants[train_count + validation_count :]),
    }


def save_split_manifest(path: Path, split: dict[str, list[str]], seed: int = 42) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "participants": split}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def nearest_frame_indices(
    timestamps_ms: np.ndarray, targets: tuple[int, ...] = TARGET_TIMES_MS
) -> list[int]:
    if len(timestamps_ms) == 0:
        return []
    return [int(np.argmin(np.abs(timestamps_ms - target))) for target in targets]


def trajectory_samples(trajectory: Trajectory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    times: list[int] = []
    for index in nearest_frame_indices(trajectory.timestamps_ms):
        try:
            vector = preprocess_landmarks(
                trajectory.landmarks[index], str(trajectory.handedness[index])
            )
        except ValueError:
            continue
        features.append(vector)
        labels.append(trajectory.label)
        times.append(int(trajectory.timestamps_ms[index]))
    if not features:
        return (
            np.empty((0, 63), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
    )


def trajectory_window_samples(
    trajectory: Trajectory,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return every valid frame in the early and final training windows."""

    features: list[np.ndarray] = []
    labels: list[int] = []
    times: list[int] = []
    for landmarks, handedness, timestamp in zip(
        trajectory.landmarks,
        trajectory.handedness,
        trajectory.timestamps_ms,
        strict=True,
    ):
        elapsed = int(timestamp)
        if not any(start <= elapsed <= end for start, end, _phase in TRAINING_WINDOWS_MS):
            continue
        try:
            vector = preprocess_landmarks(landmarks, str(handedness))
        except ValueError:
            continue
        features.append(vector)
        labels.append(trajectory.label)
        times.append(elapsed)
    if not features:
        return (
            np.empty((0, 63), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(times, dtype=np.int64),
    )


class LandmarkFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        trajectories: list[Trajectory],
        *,
        augment: bool = False,
        seed: int = 42,
        sampling: Literal["fixed", "all"] = "fixed",
    ) -> None:
        if sampling not in {"fixed", "all"}:
            raise ValueError(f"Unsupported frame sampling mode: {sampling}")
        samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        for trajectory in trajectories:
            feature_builder = (
                trajectory_samples if sampling == "fixed" else trajectory_window_samples
            )
            features, labels, times = feature_builder(trajectory)
            if len(features):
                samples.append((features, labels, times, trajectory.participant))
        non_empty = [sample for sample in samples if len(sample[0])]
        if not non_empty:
            raise ValueError("No valid landmark samples were found")
        self.features = np.concatenate([sample[0] for sample in non_empty], axis=0)
        self.labels = np.concatenate([sample[1] for sample in non_empty], axis=0)
        self.times_ms = np.concatenate([sample[2] for sample in non_empty], axis=0)
        self.participants = np.concatenate(
            [np.repeat(sample[3], len(sample[1])) for sample in non_empty]
        )
        self.phases = (self.times_ms > 450).astype(np.int64)
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if self.augment:
            features = augment_features(features, self._rng)
        return torch.from_numpy(features.copy()), torch.tensor(self.labels[index], dtype=torch.long)

    def balanced_sample_weights(self) -> np.ndarray:
        """Equalize participant, class, and early/final phase mass per epoch."""

        groups = [
            (str(participant), int(label), int(phase))
            for participant, label, phase in zip(
                self.participants, self.labels, self.phases, strict=True
            )
        ]
        counts = Counter(groups)
        weights = np.asarray([1.0 / counts[group] for group in groups], dtype=np.float64)
        return weights / float(np.mean(weights))


def trajectories_for_participants(
    trajectories: list[Trajectory], participants: list[str]
) -> list[Trajectory]:
    accepted = set(participants)
    return [trajectory for trajectory in trajectories if trajectory.participant in accepted]
