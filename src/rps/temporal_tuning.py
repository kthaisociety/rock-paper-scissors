from __future__ import annotations

import itertools
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rps.model import CLASS_NAMES
from rps.temporal import (
    LockReason,
    TemporalDecision,
    TemporalDecisionPolicy,
    TemporalObservation,
    TemporalPolicyConfig,
    TemporalPolicyKind,
)


@dataclass(frozen=True, slots=True)
class TemporalTrace:
    participant: str
    label: int
    trajectory_id: str
    timestamps_ms: np.ndarray
    probabilities: np.ndarray
    features: np.ndarray


@dataclass(frozen=True, slots=True)
class PolicyScore:
    config: TemporalPolicyConfig
    accuracy: float
    class_recall: dict[str, float]
    per_participant_accuracy: dict[str, float]
    median_lock_time_ms: float
    p90_lock_time_ms: float
    locked_by_ms: dict[str, float]
    forced_lock_rate: float
    predictions: tuple[int, ...]
    correctness: tuple[bool, ...]

    def as_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "config": self.config.as_dict(),
            "accuracy": self.accuracy,
            "class_recall": self.class_recall,
            "per_participant_accuracy": self.per_participant_accuracy,
            "median_lock_time_ms": self.median_lock_time_ms,
            "p90_lock_time_ms": self.p90_lock_time_ms,
            "locked_by_ms": self.locked_by_ms,
            "forced_lock_rate": self.forced_lock_rate,
        }
        if include_rows:
            result["predictions"] = list(self.predictions)
            result["correctness"] = list(self.correctness)
        return result


def replay_trace(trace: TemporalTrace, config: TemporalPolicyConfig) -> TemporalDecision:
    policy = TemporalDecisionPolicy(config)
    for timestamp, features in zip(trace.timestamps_ms, trace.features, strict=True):
        if int(timestamp) < config.inference_start_ms:
            policy.record_ready(int(timestamp), features)
    policy.start_round()
    for timestamp, probabilities, features in zip(
        trace.timestamps_ms,
        trace.probabilities,
        trace.features,
        strict=True,
    ):
        decision = policy.update(
            TemporalObservation(int(timestamp), probabilities, features)
        )
        if decision is not None:
            return decision
    decision = policy.force_lock()
    if decision is None:
        raise ValueError(f"Trace {trace.trajectory_id} contains no inference observation")
    return decision


def score_policy(traces: list[TemporalTrace], config: TemporalPolicyConfig) -> PolicyScore:
    decisions = [replay_trace(trace, config) for trace in traces]
    predictions = tuple(int(decision.gesture) for decision in decisions)
    correctness = tuple(
        prediction == trace.label for prediction, trace in zip(predictions, traces, strict=True)
    )
    lock_times = np.asarray(
        [int(decision.lock_time_ms) for decision in decisions], dtype=np.float64
    )
    class_recall: dict[str, float] = {}
    for label, name in enumerate(CLASS_NAMES):
        values = [
            correct
            for correct, trace in zip(correctness, traces, strict=True)
            if trace.label == label
        ]
        class_recall[name] = float(np.mean(values)) if values else 0.0
    participant_values: defaultdict[str, list[bool]] = defaultdict(list)
    for correct, trace in zip(correctness, traces, strict=True):
        participant_values[trace.participant].append(correct)
    forced = sum(decision.reason == LockReason.FORCED_DEADLINE for decision in decisions)
    return PolicyScore(
        config=config,
        accuracy=float(np.mean(correctness)),
        class_recall=class_recall,
        per_participant_accuracy={
            participant: float(np.mean(values))
            for participant, values in sorted(participant_values.items())
        },
        median_lock_time_ms=float(np.median(lock_times)),
        p90_lock_time_ms=float(np.percentile(lock_times, 90)),
        locked_by_ms={
            str(deadline): float(np.mean(lock_times <= deadline))
            for deadline in (200, 250, 300, 350)
        },
        forced_lock_rate=float(forced / len(decisions)),
        predictions=predictions,
        correctness=correctness,
    )


def temporal_search_grid() -> list[TemporalPolicyConfig]:
    configs = [TemporalPolicyConfig.baseline()]
    shared = itertools.product(
        (35.0, 55.0),
        (0.60, 0.75, 0.90),
        (0.10, 0.20),
        (0, 33, 67),
        (0.03, 0.08),
        (0.002, 0.005),
        (250, 325),
    )
    shared_values = list(shared)
    for kind in (TemporalPolicyKind.STABILITY_GATE, TemporalPolicyKind.HMM):
        for half_life, confidence, margin, dwell, change, velocity, rock_deadline in shared_values:
            constructor = (
                TemporalPolicyConfig.stability_gate
                if kind == TemporalPolicyKind.STABILITY_GATE
                else TemporalPolicyConfig.hmm
            )
            configs.append(
                constructor(
                    ema_half_life_ms=half_life,
                    confidence=confidence,
                    margin=margin,
                    dwell_ms=dwell,
                    ready_change_rms=change,
                    ready_return_rms=min(change, 0.06),
                    stable_velocity_rms_per_ms=velocity,
                    motion_velocity_rms_per_ms=max(velocity, 0.002),
                    rock_deadline_ms=rock_deadline,
                )
            )
    return configs


def participant_bootstrap_lower_bound(
    traces: list[TemporalTrace],
    baseline: PolicyScore,
    candidate: PolicyScore,
    *,
    seed: int = 42,
    samples: int = 5000,
) -> float:
    participants = sorted({trace.participant for trace in traces})
    if not participants:
        return -1.0
    indices = {
        participant: np.asarray(
            [index for index, trace in enumerate(traces) if trace.participant == participant],
            dtype=np.int64,
        )
        for participant in participants
    }
    baseline_correct = np.asarray(baseline.correctness, dtype=np.float64)
    candidate_correct = np.asarray(candidate.correctness, dtype=np.float64)
    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected = rng.choice(participants, size=len(participants), replace=True)
        sampled_indices = np.concatenate([indices[str(participant)] for participant in selected])
        differences[sample_index] = float(
            np.mean(candidate_correct[sampled_indices] - baseline_correct[sampled_indices])
        )
    return float(np.percentile(differences, 5))


def policy_is_eligible(
    traces: list[TemporalTrace],
    baseline: PolicyScore,
    candidate: PolicyScore,
) -> tuple[bool, dict[str, Any]]:
    lower_bound = participant_bootstrap_lower_bound(traces, baseline, candidate)
    participant_regressions = {
        participant: candidate.per_participant_accuracy[participant]
        - baseline.per_participant_accuracy[participant]
        for participant in baseline.per_participant_accuracy
    }
    failures: list[str] = []
    if candidate.accuracy < baseline.accuracy:
        failures.append("pooled accuracy is below baseline")
    if lower_bound < -0.02:
        failures.append("bootstrap lower bound is worse than -0.02")
    if min(candidate.class_recall.values(), default=0.0) < 0.90:
        failures.append("a gesture recall is below 0.90")
    if min(participant_regressions.values(), default=-1.0) < -0.02:
        failures.append("a participant regression is worse than -0.02")
    return not failures, {
        "bootstrap_accuracy_difference_lower_95": lower_bound,
        "participant_accuracy_differences": participant_regressions,
        "failures": failures,
    }


def select_temporal_policy(
    traces: list[TemporalTrace],
    configs: Iterable[TemporalPolicyConfig] | None = None,
) -> tuple[PolicyScore, PolicyScore, list[dict[str, Any]]]:
    configs = list(configs or temporal_search_grid())
    baseline_config = next(
        config for config in configs if config.kind == TemporalPolicyKind.BASELINE
    )
    baseline = score_policy(traces, baseline_config)
    eligible: list[PolicyScore] = []
    summaries: list[dict[str, Any]] = []
    for config in configs:
        score = baseline if config is baseline_config else score_policy(traces, config)
        passes, evidence = policy_is_eligible(traces, baseline, score)
        summaries.append({**score.as_dict(), "eligible": passes, **evidence})
        if passes:
            eligible.append(score)
    if not eligible:
        return baseline, baseline, summaries
    kind_priority = {
        TemporalPolicyKind.STABILITY_GATE: 0,
        TemporalPolicyKind.HMM: 1,
        TemporalPolicyKind.BASELINE: 2,
    }
    winner = min(
        eligible,
        key=lambda score: (
            score.median_lock_time_ms,
            score.p90_lock_time_ms,
            kind_priority[score.config.kind],
        ),
    )
    return baseline, winner, summaries


def benchmark_policy_overhead(
    traces: list[TemporalTrace],
    baseline_config: TemporalPolicyConfig,
    candidate_config: TemporalPolicyConfig,
    *,
    repeats: int = 5,
) -> float:
    def timings(config: TemporalPolicyConfig) -> np.ndarray:
        values: list[float] = []
        for _ in range(repeats):
            for trace in traces:
                started = time.perf_counter_ns()
                replay_trace(trace, config)
                observation_count = max(len(trace.timestamps_ms), 1)
                values.append((time.perf_counter_ns() - started) / 1_000_000 / observation_count)
        return np.asarray(values, dtype=np.float64)

    baseline_p95 = float(np.percentile(timings(baseline_config), 95))
    candidate_p95 = float(np.percentile(timings(candidate_config), 95))
    return max(0.0, candidate_p95 - baseline_p95)


def confirmation_failures(
    baseline: PolicyScore,
    candidate: PolicyScore,
    *,
    runtime_overhead_p95_ms: float,
) -> list[str]:
    failures: list[str] = []
    if candidate.accuracy < 0.914:
        failures.append("fresh accuracy is below 0.914")
    if sum(not value for value in candidate.correctness) > sum(
        not value for value in baseline.correctness
    ):
        failures.append("candidate makes more fresh errors than baseline")
    if min(candidate.class_recall.values(), default=0.0) < 0.90:
        failures.append("fresh gesture recall is below 0.90")
    if baseline.median_lock_time_ms - candidate.median_lock_time_ms < 25.0:
        failures.append("median lock improvement is below 25 ms")
    if candidate.p90_lock_time_ms > baseline.p90_lock_time_ms:
        failures.append("p90 lock time is worse than baseline")
    if runtime_overhead_p95_ms >= 0.1:
        failures.append("p95 temporal runtime overhead is not below 0.1 ms")
    return failures
