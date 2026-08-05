from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rps.features import augment_features, preprocess_landmarks
from rps.model import CLASS_NAMES

TARGET_TIMES_MS = (250, 350, 450, 700, 800, 900)


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


def discover_trajectories(data_dir: Path) -> list[Trajectory]:
    return [load_trajectory(path) for path in sorted(data_dir.rglob("*.npz"))]


def dataset_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
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


class LandmarkFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        trajectories: list[Trajectory],
        *,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        samples = [trajectory_samples(trajectory)[:2] for trajectory in trajectories]
        non_empty = [(features, labels) for features, labels in samples if len(features)]
        if not non_empty:
            raise ValueError("No valid landmark samples were found")
        self.features = np.concatenate([sample[0] for sample in non_empty], axis=0)
        self.labels = np.concatenate([sample[1] for sample in non_empty], axis=0)
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if self.augment:
            features = augment_features(features, self._rng)
        return torch.from_numpy(features.copy()), torch.tensor(self.labels[index], dtype=torch.long)


def trajectories_for_participants(
    trajectories: list[Trajectory], participants: list[str]
) -> list[Trajectory]:
    accepted = set(participants)
    return [trajectory for trajectory in trajectories if trajectory.participant in accepted]
