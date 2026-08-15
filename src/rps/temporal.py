from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

TEMPORAL_POLICY_FORMAT_VERSION = 1


class TemporalPolicyKind(StrEnum):
    BASELINE = "baseline"
    STABILITY_GATE = "stability_gate"
    HMM = "hmm"


class TemporalState(StrEnum):
    READY_FIST = "READY_FIST"
    TRANSITION = "TRANSITION"
    ROCK = "ROCK"
    PAPER = "PAPER"
    SCISSORS = "SCISSORS"


class LockReason(StrEnum):
    STABLE_EVIDENCE = "stable_evidence"
    ROCK_RETURN = "rock_return"
    ROCK_DEADLINE = "rock_deadline"
    HMM_POSTERIOR = "hmm_posterior"
    FORCED_DEADLINE = "forced_deadline"


@dataclass(frozen=True, slots=True)
class TemporalObservation:
    timestamp_ms: int
    probabilities: np.ndarray
    features: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class TemporalDecision:
    gesture: int | None
    lock_time_ms: int | None
    probabilities: np.ndarray
    confidence: float = 0.0
    state: TemporalState = TemporalState.READY_FIST
    reason: LockReason | None = None


@dataclass(frozen=True, slots=True)
class TemporalPolicyConfig:
    kind: TemporalPolicyKind = TemporalPolicyKind.BASELINE
    inference_start_ms: int = 150
    early_lock_start_ms: int = 200
    force_lock_ms: int = 450

    # Immutable baseline behavior.
    baseline_ema_alpha: float = 0.4
    baseline_stable_results: int = 3
    confidence: float = 0.70
    margin: float = 0.15

    # Time-aware gate/HMM behavior.
    ema_half_life_ms: float = 50.0
    dwell_ms: int = 50
    ready_window_ms: int = 200
    ready_change_rms: float = 0.10
    ready_return_rms: float = 0.06
    stable_velocity_rms_per_ms: float = 0.002
    motion_velocity_rms_per_ms: float = 0.003
    rock_deadline_ms: int = 300

    # HMM transition and emission controls.
    hmm_ready_stay: float = 0.92
    hmm_transition_stay: float = 0.55
    hmm_gesture_stay: float = 0.98
    hmm_ready_sigma: float = 0.08
    hmm_motion_scale: float = 0.002

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TemporalPolicyKind(self.kind))
        if self.inference_start_ms < 0:
            raise ValueError("inference_start_ms must be non-negative")
        if not self.inference_start_ms <= self.early_lock_start_ms <= self.force_lock_ms:
            raise ValueError("lock timing must be ordered")
        if self.rock_deadline_ms > self.force_lock_ms:
            raise ValueError("rock_deadline_ms cannot exceed force_lock_ms")
        if self.dwell_ms < 0 or self.ready_window_ms < 1:
            raise ValueError("temporal durations must be positive")
        for name in (
            "baseline_ema_alpha",
            "confidence",
            "margin",
            "hmm_ready_stay",
            "hmm_transition_stay",
            "hmm_gesture_stay",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @classmethod
    def baseline(
        cls,
        *,
        inference_start_ms: int = 150,
        early_lock_start_ms: int = 200,
        force_lock_ms: int = 450,
        ema_alpha: float = 0.4,
        stable_results: int = 3,
        confidence: float = 0.70,
        margin: float = 0.15,
    ) -> TemporalPolicyConfig:
        return cls(
            kind=TemporalPolicyKind.BASELINE,
            inference_start_ms=inference_start_ms,
            early_lock_start_ms=early_lock_start_ms,
            force_lock_ms=force_lock_ms,
            baseline_ema_alpha=ema_alpha,
            baseline_stable_results=stable_results,
            confidence=confidence,
            margin=margin,
        )

    @classmethod
    def stability_gate(cls, **overrides: Any) -> TemporalPolicyConfig:
        return cls(kind=TemporalPolicyKind.STABILITY_GATE, **overrides)

    @classmethod
    def hmm(cls, **overrides: Any) -> TemporalPolicyConfig:
        return cls(kind=TemporalPolicyKind.HMM, **overrides)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TemporalPolicyConfig:
        return cls(**payload)


def _normalized_probabilities(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(3)
    vector = np.clip(vector, 1e-8, None)
    return (vector / vector.sum()).astype(np.float32)


def _feature_vector(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    vector = np.asarray(values, dtype=np.float32).reshape(63)
    if not np.isfinite(vector).all():
        return None
    return vector


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


class TemporalDecisionPolicy:
    """Online temporal policy shared by camera runtime and trajectory replay."""

    def __init__(self, config: TemporalPolicyConfig | None = None) -> None:
        self.config = config or TemporalPolicyConfig.baseline()
        self.reset()

    def reset(self) -> None:
        self._ready_samples: deque[tuple[int, np.ndarray]] = deque()
        self._ready_reference: np.ndarray | None = None
        self._ema: np.ndarray | None = None
        self._last_timestamp: int | None = None
        self._last_features: np.ndarray | None = None
        self._baseline_leaders: list[int] = []
        self._candidate: int | None = None
        self._candidate_since: int | None = None
        self._moved = False
        self._returned_to_rock = False
        self._decision: TemporalDecision | None = None
        self._state = TemporalState.READY_FIST
        self._posterior = np.asarray([0.98, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)

    @property
    def probabilities(self) -> np.ndarray:
        if self._ema is None:
            return np.zeros(3, dtype=np.float32)
        return self._ema.copy()

    @property
    def state(self) -> TemporalState:
        return self._state

    def record_ready(self, timestamp_ms: int, features: np.ndarray | None) -> None:
        vector = _feature_vector(features)
        if vector is None:
            return
        self._ready_samples.append((int(timestamp_ms), vector.copy()))
        cutoff = int(timestamp_ms) - self.config.ready_window_ms
        while self._ready_samples and self._ready_samples[0][0] < cutoff:
            self._ready_samples.popleft()

    def start_round(self) -> None:
        if self._ready_samples:
            self._ready_reference = np.median(
                np.stack([sample for _, sample in self._ready_samples]), axis=0
            ).astype(np.float32)
        self._last_timestamp = None
        self._last_features = None

    def _motion(self, observation: TemporalObservation) -> tuple[float, float]:
        features = _feature_vector(observation.features)
        pose_delta = math.inf
        velocity = math.inf
        if features is not None:
            if self._ready_reference is None:
                self._ready_reference = features.copy()
            pose_delta = _rms(features, self._ready_reference)
            if self._last_features is not None and self._last_timestamp is not None:
                delta_ms = max(1, observation.timestamp_ms - self._last_timestamp)
                velocity = _rms(features, self._last_features) / delta_ms
            self._last_features = features
        self._last_timestamp = observation.timestamp_ms
        if (
            pose_delta >= self.config.ready_change_rms
            or (
                math.isfinite(velocity)
                and velocity >= self.config.motion_velocity_rms_per_ms
            )
        ):
            self._moved = True
        return pose_delta, velocity

    def _update_ema(
        self,
        values: np.ndarray,
        timestamp_ms: int,
        previous_timestamp: int | None,
    ) -> None:
        if self._ema is None:
            self._ema = values.copy()
            return
        if self.config.kind == TemporalPolicyKind.BASELINE:
            alpha = self.config.baseline_ema_alpha
        else:
            delta_ms = max(1, timestamp_ms - (previous_timestamp or timestamp_ms - 1))
            alpha = 1.0 - math.exp(-math.log(2.0) * delta_ms / self.config.ema_half_life_ms)
        self._ema = (alpha * values + (1.0 - alpha) * self._ema).astype(np.float32)

    def _make_decision(
        self, gesture: int, timestamp_ms: int, reason: LockReason
    ) -> TemporalDecision:
        assert self._ema is not None
        self._state = (TemporalState.ROCK, TemporalState.PAPER, TemporalState.SCISSORS)[gesture]
        self._decision = TemporalDecision(
            gesture=gesture,
            lock_time_ms=int(timestamp_ms),
            probabilities=self._ema.copy(),
            confidence=float(self._ema[gesture]),
            state=self._state,
            reason=reason,
        )
        return self._decision

    def _baseline_update(self, timestamp_ms: int) -> TemporalDecision | None:
        assert self._ema is not None
        leader = int(np.argmax(self._ema))
        self._baseline_leaders.append(leader)
        self._baseline_leaders = self._baseline_leaders[-self.config.baseline_stable_results :]
        sorted_values = np.sort(self._ema)
        stable = (
            len(self._baseline_leaders) == self.config.baseline_stable_results
            and len(set(self._baseline_leaders)) == 1
        )
        if (
            timestamp_ms >= self.config.early_lock_start_ms
            and stable
            and float(self._ema[leader]) >= self.config.confidence
            and float(sorted_values[-1] - sorted_values[-2]) >= self.config.margin
        ):
            return self._make_decision(leader, timestamp_ms, LockReason.STABLE_EVIDENCE)
        return None

    def _stable_candidate(
        self,
        leader: int,
        timestamp_ms: int,
        *,
        eligible: bool,
        reason: LockReason,
    ) -> TemporalDecision | None:
        assert self._ema is not None
        sorted_values = np.sort(self._ema)
        strong = (
            float(self._ema[leader]) >= self.config.confidence
            and float(sorted_values[-1] - sorted_values[-2]) >= self.config.margin
        )
        if not eligible or not strong:
            self._candidate = None
            self._candidate_since = None
            return None
        if self._candidate != leader:
            self._candidate = leader
            self._candidate_since = timestamp_ms
            if self.config.dwell_ms == 0:
                return self._make_decision(leader, timestamp_ms, reason)
            return None
        assert self._candidate_since is not None
        if timestamp_ms - self._candidate_since >= self.config.dwell_ms:
            return self._make_decision(leader, timestamp_ms, reason)
        return None

    def _gate_update(
        self, timestamp_ms: int, pose_delta: float, velocity: float
    ) -> TemporalDecision | None:
        assert self._ema is not None
        leader = int(np.argmax(self._ema))
        stable_pose = velocity <= self.config.stable_velocity_rms_per_ms
        if leader == 0:
            returned = self._moved and pose_delta <= self.config.ready_return_rms
            self._returned_to_rock = self._returned_to_rock or returned
            if timestamp_ms >= self.config.rock_deadline_ms:
                return self._stable_candidate(
                    leader,
                    timestamp_ms,
                    eligible=stable_pose,
                    reason=LockReason.ROCK_DEADLINE,
                )
            return self._stable_candidate(
                leader,
                timestamp_ms,
                eligible=stable_pose and self._returned_to_rock,
                reason=LockReason.ROCK_RETURN,
            )
        self._state = TemporalState.TRANSITION if not stable_pose else self._state
        return self._stable_candidate(
            leader,
            timestamp_ms,
            eligible=self._moved and stable_pose,
            reason=LockReason.STABLE_EVIDENCE,
        )

    def _hmm_transition(self, timestamp_ms: int) -> np.ndarray:
        c = self.config
        matrix = np.zeros((5, 5), dtype=np.float64)
        direct_rock = 0.02 if timestamp_ms >= c.rock_deadline_ms else 0.0
        matrix[0] = [c.hmm_ready_stay, 1.0 - c.hmm_ready_stay - direct_rock, direct_rock, 0, 0]
        remaining = 1.0 - c.hmm_transition_stay
        matrix[1] = [0, c.hmm_transition_stay, remaining / 3, remaining / 3, remaining / 3]
        for index in range(2, 5):
            matrix[index, 1] = 1.0 - c.hmm_gesture_stay
            matrix[index, index] = c.hmm_gesture_stay
        return matrix

    def _hmm_update(
        self, timestamp_ms: int, pose_delta: float, velocity: float
    ) -> TemporalDecision | None:
        assert self._ema is not None
        c = self.config
        finite_delta = pose_delta if math.isfinite(pose_delta) else c.ready_change_rms * 2
        finite_velocity = velocity if math.isfinite(velocity) else c.hmm_motion_scale * 2
        ready = float(self._ema[0]) * math.exp(-0.5 * (finite_delta / c.hmm_ready_sigma) ** 2)
        motion = finite_velocity / (finite_velocity + c.hmm_motion_scale)
        entropy = -float(np.sum(self._ema * np.log(np.clip(self._ema, 1e-8, 1.0)))) / math.log(3)
        emissions = np.asarray(
            [ready + 1e-8, 0.5 * motion + 0.5 * entropy + 1e-8, *np.clip(self._ema, 1e-8, 1.0)],
            dtype=np.float64,
        )
        self._posterior = (self._posterior @ self._hmm_transition(timestamp_ms)) * emissions
        self._posterior /= max(float(self._posterior.sum()), 1e-12)
        hidden = int(np.argmax(self._posterior))
        self._state = (
            TemporalState.READY_FIST,
            TemporalState.TRANSITION,
            TemporalState.ROCK,
            TemporalState.PAPER,
            TemporalState.SCISSORS,
        )[hidden]
        if hidden < 2:
            self._candidate = None
            self._candidate_since = None
            return None
        gesture = hidden - 2
        stable_pose = finite_velocity <= c.stable_velocity_rms_per_ms
        protocol_eligible = self._moved or (
            gesture == 0 and timestamp_ms >= c.rock_deadline_ms
        )
        posterior_confident = float(self._posterior[hidden]) >= c.confidence
        return self._stable_candidate(
            gesture,
            timestamp_ms,
            eligible=stable_pose and protocol_eligible and posterior_confident,
            reason=LockReason.HMM_POSTERIOR,
        )

    def update(self, observation: TemporalObservation) -> TemporalDecision | None:
        if self._decision is not None:
            return self._decision
        timestamp_ms = int(observation.timestamp_ms)
        if (
            timestamp_ms < self.config.inference_start_ms
            or timestamp_ms > self.config.force_lock_ms
        ):
            return None
        values = _normalized_probabilities(observation.probabilities)
        previous_timestamp = self._last_timestamp
        pose_delta, velocity = self._motion(observation)
        self._update_ema(values, timestamp_ms, previous_timestamp)
        if self.config.kind == TemporalPolicyKind.BASELINE:
            return self._baseline_update(timestamp_ms)
        if timestamp_ms < self.config.early_lock_start_ms:
            return None
        if self.config.kind == TemporalPolicyKind.STABILITY_GATE:
            return self._gate_update(timestamp_ms, pose_delta, velocity)
        return self._hmm_update(timestamp_ms, pose_delta, velocity)

    def force_lock(self, timestamp_ms: int | None = None) -> TemporalDecision | None:
        if self._decision is not None:
            return self._decision
        if self._ema is None:
            return None
        timestamp = self.config.force_lock_ms if timestamp_ms is None else int(timestamp_ms)
        return self._make_decision(
            int(np.argmax(self._ema)), timestamp, LockReason.FORCED_DEADLINE
        )


class TemporalPolicyArtifactError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TemporalPolicyArtifact:
    config: TemporalPolicyConfig
    model_data_fingerprint: str
    checkpoint_sha256: str
    selection_metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "candidate"

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_version": TEMPORAL_POLICY_FORMAT_VERSION,
            "status": self.status,
            "model_data_fingerprint": self.model_data_fingerprint,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config": self.config.as_dict(),
            "selection_metrics": self.selection_metrics,
        }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_temporal_policy(path: Path, artifact: TemporalPolicyArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact.as_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_temporal_policy(
    path: Path,
    *,
    model_data_fingerprint: str | None = None,
    checkpoint_path: Path | None = None,
) -> TemporalPolicyArtifact:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemporalPolicyArtifactError(
            f"Could not load temporal policy {path}: {error}"
        ) from error
    if payload.get("format_version") != TEMPORAL_POLICY_FORMAT_VERSION:
        raise TemporalPolicyArtifactError("Unsupported temporal policy format version")
    if (
        model_data_fingerprint is not None
        and payload.get("model_data_fingerprint") != model_data_fingerprint
    ):
        raise TemporalPolicyArtifactError(
            "Temporal policy data fingerprint does not match checkpoint"
        )
    if checkpoint_path is not None:
        expected = str(payload.get("checkpoint_sha256", ""))
        if checkpoint_sha256(checkpoint_path) != expected:
            raise TemporalPolicyArtifactError("Temporal policy checkpoint hash does not match")
    try:
        config = TemporalPolicyConfig.from_dict(payload["config"])
    except (KeyError, TypeError, ValueError) as error:
        raise TemporalPolicyArtifactError(
            f"Invalid temporal policy configuration: {error}"
        ) from error
    return TemporalPolicyArtifact(
        config=config,
        model_data_fingerprint=str(payload.get("model_data_fingerprint", "")),
        checkpoint_sha256=str(payload.get("checkpoint_sha256", "")),
        selection_metrics=dict(payload.get("selection_metrics", {})),
        status=str(payload.get("status", "candidate")),
    )
