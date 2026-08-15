from __future__ import annotations

import json

import numpy as np
import pytest

from rps.game import (
    GameConfig,
    GameController,
    Gesture,
    HandPrediction,
    RoundPhase,
    lock_from_probability_trace,
)
from rps.temporal import (
    LockReason,
    TemporalDecisionPolicy,
    TemporalObservation,
    TemporalPolicyArtifact,
    TemporalPolicyArtifactError,
    TemporalPolicyConfig,
    checkpoint_sha256,
    load_temporal_policy,
    save_temporal_policy,
)
from rps.temporal_tuning import (
    PolicyScore,
    confirmation_failures,
    temporal_search_grid,
)

READY = np.zeros(63, dtype=np.float32)
DEPLOYED = np.full(63, 0.2, dtype=np.float32)
ROCK = np.asarray((0.9, 0.05, 0.05), dtype=np.float32)
PAPER = np.asarray((0.05, 0.9, 0.05), dtype=np.float32)
SCISSORS = np.asarray((0.05, 0.05, 0.9), dtype=np.float32)


def gate_config(**overrides: object) -> TemporalPolicyConfig:
    values = {
        "early_lock_start_ms": 150,
        "dwell_ms": 33,
        "ready_change_rms": 0.10,
        "stable_velocity_rms_per_ms": 0.002,
        "motion_velocity_rms_per_ms": 0.003,
        "rock_deadline_ms": 300,
    }
    values.update(overrides)
    return TemporalPolicyConfig.stability_gate(**values)


def primed_policy(config: TemporalPolicyConfig) -> TemporalDecisionPolicy:
    policy = TemporalDecisionPolicy(config)
    policy.record_ready(-100, READY)
    policy.record_ready(0, READY)
    policy.start_round()
    return policy


def test_gate_locks_deployed_non_rock_after_elapsed_dwell() -> None:
    policy = primed_policy(gate_config())
    assert policy.update(TemporalObservation(150, PAPER, DEPLOYED)) is None
    assert policy.update(TemporalObservation(200, PAPER, DEPLOYED)) is None
    decision = policy.update(TemporalObservation(233, PAPER, DEPLOYED))

    assert decision is not None
    assert decision.gesture == Gesture.PAPER
    assert decision.lock_time_ms == 233
    assert decision.reason == LockReason.STABLE_EVIDENCE


def test_gate_does_not_lock_unchanged_ready_fist_before_rock_deadline() -> None:
    policy = primed_policy(gate_config())
    for timestamp in (150, 200, 250, 299):
        assert policy.update(TemporalObservation(timestamp, ROCK, READY)) is None
    assert policy.update(TemporalObservation(300, ROCK, READY)) is None
    decision = policy.update(TemporalObservation(333, ROCK, READY))

    assert decision is not None
    assert decision.gesture == Gesture.ROCK
    assert decision.reason == LockReason.ROCK_DEADLINE


def test_gate_can_lock_rock_after_movement_and_return() -> None:
    policy = primed_policy(gate_config())
    assert policy.update(TemporalObservation(150, PAPER, DEPLOYED)) is None
    assert policy.update(TemporalObservation(200, ROCK, READY)) is None
    assert policy.update(TemporalObservation(233, ROCK, READY)) is None
    assert policy.update(TemporalObservation(266, ROCK, READY)) is None
    decision = policy.update(TemporalObservation(299, ROCK, READY))

    assert decision is not None
    assert decision.gesture == Gesture.ROCK
    assert decision.reason == LockReason.ROCK_RETURN


def test_gate_resets_dwell_when_confidence_flickers() -> None:
    policy = primed_policy(gate_config())
    policy.update(TemporalObservation(150, PAPER, DEPLOYED))
    assert policy.update(TemporalObservation(200, PAPER, DEPLOYED)) is None
    assert policy.update(TemporalObservation(233, SCISSORS, DEPLOYED)) is None
    assert policy.update(TemporalObservation(266, PAPER, DEPLOYED)) is None
    decision = policy.update(TemporalObservation(299, PAPER, DEPLOYED))

    assert decision is not None
    assert decision.gesture == Gesture.PAPER


def test_gate_uses_milliseconds_not_frame_count() -> None:
    policy = primed_policy(gate_config(dwell_ms=60))
    policy.update(TemporalObservation(150, PAPER, DEPLOYED))
    assert policy.update(TemporalObservation(201, PAPER, DEPLOYED)) is None
    assert policy.update(TemporalObservation(240, PAPER, DEPLOYED)) is None
    decision = policy.update(TemporalObservation(262, PAPER, DEPLOYED))
    assert decision is not None
    assert decision.lock_time_ms == 262


def test_zero_dwell_grid_candidate_can_lock_on_first_stable_observation() -> None:
    policy = primed_policy(gate_config(dwell_ms=0, stable_velocity_rms_per_ms=0.01))
    policy.update(TemporalObservation(150, PAPER, DEPLOYED))
    decision = policy.update(TemporalObservation(200, PAPER, DEPLOYED))
    assert decision is not None
    assert decision.lock_time_ms == 200


def test_force_lock_remains_at_450_ms() -> None:
    policy = primed_policy(gate_config(confidence=0.99))
    policy.update(TemporalObservation(200, np.asarray((0.36, 0.35, 0.29)), DEPLOYED))
    decision = policy.force_lock()

    assert decision is not None
    assert decision.lock_time_ms == 450
    assert decision.reason == LockReason.FORCED_DEADLINE


def test_hmm_filter_enters_transition_then_committed_gesture() -> None:
    config = TemporalPolicyConfig.hmm(
        early_lock_start_ms=150,
        dwell_ms=33,
        confidence=0.50,
        ready_change_rms=0.10,
        stable_velocity_rms_per_ms=0.002,
        motion_velocity_rms_per_ms=0.003,
        rock_deadline_ms=300,
    )
    policy = primed_policy(config)
    assert policy.update(TemporalObservation(150, PAPER, DEPLOYED)) is None
    assert policy.state.value == "TRANSITION"
    assert policy.update(TemporalObservation(183, PAPER, DEPLOYED)) is None
    decision = policy.update(TemporalObservation(216, PAPER, DEPLOYED))

    assert decision is not None
    assert decision.gesture == Gesture.PAPER
    assert decision.reason == LockReason.HMM_POSTERIOR


def test_controller_and_replay_make_identical_gate_decision() -> None:
    temporal_config = gate_config()
    controller = GameController(
        GameConfig(hand_stable_ms=0, countdown_ms=100, temporal_policy=temporal_config)
    )
    controller.update(0, HandPrediction(0, ROCK, features=READY))
    controller.update(50, HandPrediction(50, ROCK, features=READY))
    controller.update(100, HandPrediction(100, ROCK, features=READY))
    controller.update(250, HandPrediction(250, PAPER, features=DEPLOYED))
    controller.update(300, HandPrediction(300, PAPER, features=DEPLOYED))
    live = controller.update(333, HandPrediction(333, PAPER, features=DEPLOYED))

    timestamps = np.asarray((0, 50, 150, 200, 233), dtype=np.int64)
    probabilities = np.stack((ROCK, ROCK, PAPER, PAPER, PAPER))
    features = np.stack((READY, READY, DEPLOYED, DEPLOYED, DEPLOYED))
    replay = lock_from_probability_trace(
        timestamps,
        probabilities,
        features=features,
        temporal_config=temporal_config,
    )

    assert live.phase == RoundPhase.LOCKED
    assert live.locked_user == replay.gesture == Gesture.PAPER
    assert live.lock_time_ms == replay.lock_time_ms == 233
    assert live.lock_reason == replay.reason


def test_temporal_artifact_validates_checkpoint_and_data_fingerprint(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    policy_path = tmp_path / "policy.json"
    artifact = TemporalPolicyArtifact(
        config=gate_config(),
        model_data_fingerprint="fingerprint",
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        status="candidate",
    )
    save_temporal_policy(policy_path, artifact)

    loaded = load_temporal_policy(
        policy_path,
        model_data_fingerprint="fingerprint",
        checkpoint_path=checkpoint,
    )
    assert loaded.config == artifact.config

    with pytest.raises(TemporalPolicyArtifactError, match="fingerprint"):
        load_temporal_policy(policy_path, model_data_fingerprint="different")
    checkpoint.write_bytes(b"changed")
    with pytest.raises(TemporalPolicyArtifactError, match="hash"):
        load_temporal_policy(policy_path, checkpoint_path=checkpoint)


def test_temporal_artifact_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"format_version": 99}), encoding="utf-8")
    with pytest.raises(TemporalPolicyArtifactError, match="format"):
        load_temporal_policy(path)


def _score(
    config: TemporalPolicyConfig,
    *,
    accuracy: float,
    median: float,
    p90: float,
    recalls: dict[str, float] | None = None,
    correctness: tuple[bool, ...] = (True,) * 100,
) -> PolicyScore:
    return PolicyScore(
        config=config,
        accuracy=accuracy,
        class_recall=recalls or {"ROCK": accuracy, "PAPER": accuracy, "SCISSORS": accuracy},
        per_participant_accuracy={"P08": accuracy, "P09": accuracy},
        median_lock_time_ms=median,
        p90_lock_time_ms=p90,
        locked_by_ms={},
        forced_lock_rate=0.0,
        predictions=(0,) * len(correctness),
        correctness=correctness,
    )


def test_confirmation_gates_require_accuracy_speed_and_runtime() -> None:
    baseline = _score(TemporalPolicyConfig.baseline(), accuracy=0.92, median=250, p90=450)
    candidate = _score(gate_config(), accuracy=0.92, median=220, p90=440)
    assert not confirmation_failures(
        baseline, candidate, runtime_overhead_p95_ms=0.05
    )

    slow = _score(gate_config(), accuracy=0.92, median=240, p90=460)
    failures = confirmation_failures(baseline, slow, runtime_overhead_p95_ms=0.1)
    assert "median lock improvement is below 25 ms" in failures
    assert "p90 lock time is worse than baseline" in failures
    assert "p95 temporal runtime overhead is not below 0.1 ms" in failures


def test_search_grid_is_deterministic_and_includes_all_policy_kinds() -> None:
    first = temporal_search_grid()
    second = temporal_search_grid()
    assert [config.as_dict() for config in first] == [config.as_dict() for config in second]
    assert {config.kind.value for config in first} == {
        "baseline",
        "stability_gate",
        "hmm",
    }
