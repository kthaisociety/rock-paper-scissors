from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np

from rps.temporal import (
    LockReason,
    TemporalDecision,
    TemporalDecisionPolicy,
    TemporalObservation,
    TemporalPolicyConfig,
    TemporalState,
)


class Gesture(IntEnum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class RoundPhase(StrEnum):
    READY = "READY"
    COUNTDOWN = "COUNTDOWN"
    PREDICTING = "PREDICTING"
    LOCKED = "LOCKED"
    REVEAL = "REVEAL"
    RESULT = "RESULT"
    MATCH_OVER = "MATCH_OVER"


class Outcome(StrEnum):
    AI_WIN = "AI WINS"
    TIE = "TIE"
    USER_WIN = "YOU WIN"


class MatchWinner(StrEnum):
    USER = "YOU"
    AI = "AI"


@dataclass(frozen=True, slots=True)
class ScoreSnapshot:
    user_points: int = 0
    ai_points: int = 0
    ties: int = 0
    user_round_wins: int = 0
    ai_round_wins: int = 0
    user_streak: int = 0
    ai_streak: int = 0
    best_user_streak: int = 0
    user_matches: int = 0
    ai_matches: int = 0
    rounds_played: int = 0
    target_points: int = 3


@dataclass(frozen=True, slots=True)
class HandPrediction:
    timestamp_ms: int
    probabilities: np.ndarray
    centered: bool = True
    features: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class GameConfig:
    hand_stable_ms: int = 250
    countdown_ms: int = 1800
    inference_start_ms: int = 150
    early_lock_start_ms: int = 200
    force_lock_ms: int = 450
    final_start_ms: int = 650
    reveal_ms: int = 950
    result_ms: int = 1200
    match_result_ms: int = 2500
    clear_ms: int = 200
    ema_alpha: float = 0.4
    stable_results: int = 3
    early_confidence: float = 0.70
    early_margin: float = 0.15
    ready_rock_confidence: float = 0.55
    target_points: int = 3
    countdown_labels: tuple[str, str, str] = ("ROCK", "PAPER", "SCISSORS")
    temporal_policy: TemporalPolicyConfig | None = None


@dataclass(slots=True)
class GameViewState:
    phase: RoundPhase
    message: str
    countdown: int | None = None
    countdown_label: str | None = None
    ready_progress: float = 0.0
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    locked_user: Gesture | None = None
    ai_move: Gesture | None = None
    final_user: Gesture | None = None
    outcome: Outcome | None = None
    lock_time_ms: int | None = None
    prediction_lead_ms: int | None = None
    temporal_state: TemporalState = TemporalState.READY_FIST
    lock_reason: LockReason | None = None
    score: ScoreSnapshot = field(default_factory=ScoreSnapshot)
    match_winner: MatchWinner | None = None
    effect_event: str | None = None
    event_id: int = 0


def counter_move(gesture: Gesture) -> Gesture:
    return Gesture((int(gesture) + 1) % 3)


def score_round(ai_move: Gesture, user_move: Gesture) -> Outcome:
    if ai_move == user_move:
        return Outcome.TIE
    if ai_move == counter_move(user_move):
        return Outcome.AI_WIN
    return Outcome.USER_WIN


LockDecision = TemporalDecision


def _policy_config(config: GameConfig) -> TemporalPolicyConfig:
    return config.temporal_policy or TemporalPolicyConfig.baseline(
        inference_start_ms=config.inference_start_ms,
        early_lock_start_ms=config.early_lock_start_ms,
        force_lock_ms=config.force_lock_ms,
        ema_alpha=config.ema_alpha,
        stable_results=config.stable_results,
        confidence=config.early_confidence,
        margin=config.early_margin,
    )


def lock_from_probability_trace(
    timestamps_ms: np.ndarray,
    probabilities: np.ndarray,
    config: GameConfig | None = None,
    *,
    features: np.ndarray | None = None,
    temporal_config: TemporalPolicyConfig | None = None,
) -> LockDecision:
    config = config or GameConfig()
    policy = TemporalDecisionPolicy(temporal_config or _policy_config(config))
    feature_rows: list[np.ndarray | None]
    if features is None:
        feature_rows = [None] * len(timestamps_ms)
    else:
        feature_rows = [row for row in np.asarray(features, dtype=np.float32)]
    for timestamp, feature in zip(timestamps_ms, feature_rows, strict=True):
        if int(timestamp) < policy.config.inference_start_ms:
            policy.record_ready(int(timestamp), feature)
    policy.start_round()
    for timestamp, values, feature in zip(
        timestamps_ms, probabilities, feature_rows, strict=True
    ):
        decision = policy.update(
            TemporalObservation(int(timestamp), np.asarray(values), feature)
        )
        if decision is not None:
            return decision
    decision = policy.force_lock()
    if decision is not None:
        return decision
    return LockDecision(
        None,
        None,
        np.zeros(3, dtype=np.float32),
        state=policy.state,
    )


class GameController:
    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self._event_id = 0
        self.reset_session()

    def reset_round(self) -> None:
        self.phase = RoundPhase.READY
        self._hand_since: int | None = None
        self._countdown_started: int | None = None
        self._go_timestamp: int | None = None
        self._result_timestamp: int | None = None
        self._absent_since: int | None = None
        self._policy = TemporalDecisionPolicy(_policy_config(self.config))
        self._ema: np.ndarray | None = None
        self._final_probabilities: list[np.ndarray] = []
        self._last_prediction_timestamp = -1
        self.locked_user: Gesture | None = None
        self.ai_move: Gesture | None = None
        self.final_user: Gesture | None = None
        self.outcome: Outcome | None = None
        self.match_winner: MatchWinner | None = None
        self.lock_time_ms: int | None = None
        self.lock_reason: LockReason | None = None
        self.effect_event: str | None = None
        self.probability_trace: list[tuple[int, list[float]]] = []
        self._message = "Hold a closed fist in the frame"

    def reset_match(self) -> None:
        self.user_points = 0
        self.ai_points = 0
        self.user_streak = 0
        self.ai_streak = 0
        self.reset_round()

    def reset_session(self) -> None:
        self.ties = 0
        self.user_round_wins = 0
        self.ai_round_wins = 0
        self.best_user_streak = 0
        self.user_matches = 0
        self.ai_matches = 0
        self.rounds_played = 0
        self.reset_match()

    def reset(self) -> None:
        """Backward-compatible full reset."""

        self.reset_session()

    def _score_snapshot(self) -> ScoreSnapshot:
        return ScoreSnapshot(
            user_points=self.user_points,
            ai_points=self.ai_points,
            ties=self.ties,
            user_round_wins=self.user_round_wins,
            ai_round_wins=self.ai_round_wins,
            user_streak=self.user_streak,
            ai_streak=self.ai_streak,
            best_user_streak=self.best_user_streak,
            user_matches=self.user_matches,
            ai_matches=self.ai_matches,
            rounds_played=self.rounds_played,
            target_points=self.config.target_points,
        )

    def _record_outcome(self, outcome: Outcome) -> None:
        self.rounds_played += 1
        self.outcome = outcome
        self._event_id += 1
        if outcome == Outcome.USER_WIN:
            self.user_points += 1
            self.user_round_wins += 1
            self.user_streak += 1
            self.ai_streak = 0
            self.best_user_streak = max(self.best_user_streak, self.user_streak)
            self.effect_event = "user_win"
        elif outcome == Outcome.AI_WIN:
            self.ai_points += 1
            self.ai_round_wins += 1
            self.ai_streak += 1
            self.user_streak = 0
            self.effect_event = "ai_win"
        else:
            self.ties += 1
            self.effect_event = "tie"

        if self.user_points >= self.config.target_points:
            self.user_matches += 1
            self.match_winner = MatchWinner.USER
            self.phase = RoundPhase.MATCH_OVER
            self.effect_event = "user_match"
        elif self.ai_points >= self.config.target_points:
            self.ai_matches += 1
            self.match_winner = MatchWinner.AI
            self.phase = RoundPhase.MATCH_OVER
            self.effect_event = "ai_match"
        else:
            self.phase = RoundPhase.RESULT

    def start_countdown(self, timestamp_ms: int) -> None:
        self.phase = RoundPhase.COUNTDOWN
        self._countdown_started = timestamp_ms
        self._go_timestamp = timestamp_ms + self.config.countdown_ms
        self._message = "Get ready"

    def _lock(self, decision: TemporalDecision) -> None:
        if decision.gesture is None or decision.lock_time_ms is None:
            return
        self._ema = decision.probabilities.copy()
        self.locked_user = Gesture(decision.gesture)
        self.ai_move = counter_move(self.locked_user)
        self.lock_time_ms = decision.lock_time_ms
        self.lock_reason = decision.reason
        self.phase = RoundPhase.LOCKED
        self._message = "AI MOVE LOCKED"

    def update(self, timestamp_ms: int, hand_result: HandPrediction | None) -> GameViewState:
        hand_present = hand_result is not None

        if (
            hand_result is not None
            and hand_result.features is not None
            and (
                self.phase == RoundPhase.READY
                or (
                self.phase == RoundPhase.COUNTDOWN
                and self._go_timestamp is not None
                and hand_result.timestamp_ms <= self._go_timestamp
                )
            )
        ):
            self._policy.record_ready(hand_result.timestamp_ms, hand_result.features)

        if self.phase == RoundPhase.READY:
            ready_pose = (
                hand_result is not None
                and hand_result.centered
                and int(np.argmax(hand_result.probabilities)) == int(Gesture.ROCK)
                and float(hand_result.probabilities[Gesture.ROCK])
                >= self.config.ready_rock_confidence
            )
            if ready_pose:
                if self._hand_since is None:
                    self._hand_since = timestamp_ms
                if timestamp_ms - self._hand_since >= self.config.hand_stable_ms:
                    self.start_countdown(timestamp_ms)
            else:
                self._hand_since = None

        if self.phase == RoundPhase.COUNTDOWN:
            assert self._go_timestamp is not None
            if timestamp_ms >= self._go_timestamp:
                self.phase = RoundPhase.PREDICTING
                self._policy.start_round()
                self._message = "GO! Deploy your gesture"

        if self.phase in {RoundPhase.PREDICTING, RoundPhase.LOCKED}:
            assert self._go_timestamp is not None
            elapsed_ms = timestamp_ms - self._go_timestamp
            prediction_elapsed_ms = (
                hand_result.timestamp_ms - self._go_timestamp if hand_result is not None else -1
            )
            is_new_prediction = (
                hand_result is not None
                and hand_result.timestamp_ms > self._last_prediction_timestamp
            )
            if is_new_prediction:
                self._last_prediction_timestamp = hand_result.timestamp_ms
            if (
                hand_result is not None
                and is_new_prediction
                and prediction_elapsed_ms >= self._policy.config.inference_start_ms
                and self.phase == RoundPhase.PREDICTING
            ):
                values = np.asarray(hand_result.probabilities, dtype=np.float32).reshape(3)
                self.probability_trace.append((prediction_elapsed_ms, values.tolist()))
                decision = self._policy.update(
                    TemporalObservation(
                        prediction_elapsed_ms,
                        values,
                        hand_result.features,
                    )
                )
                self._ema = self._policy.probabilities
                if decision is not None:
                    self._lock(decision)
            if (
                self.phase == RoundPhase.PREDICTING
                and elapsed_ms >= self._policy.config.force_lock_ms
            ):
                decision = self._policy.force_lock(self._policy.config.force_lock_ms)
                if decision is None:
                    self.reset_round()
                    self._message = "No hand detected - try again"
                    return self.view(timestamp_ms)
                self._lock(decision)
            if (
                self.phase == RoundPhase.LOCKED
                and hand_result is not None
                and is_new_prediction
                and self.config.final_start_ms <= prediction_elapsed_ms <= self.config.reveal_ms
            ):
                self._final_probabilities.append(
                    np.asarray(hand_result.probabilities, dtype=np.float32).reshape(3)
                )
            if self.phase == RoundPhase.LOCKED and elapsed_ms >= self.config.reveal_ms:
                if not self._final_probabilities:
                    self.reset_round()
                    self._message = "Final gesture missing - try again"
                    return self.view(timestamp_ms)
                final_mean = np.mean(np.stack(self._final_probabilities), axis=0)
                self.final_user = Gesture(int(np.argmax(final_mean)))
                assert self.ai_move is not None
                self._record_outcome(score_round(self.ai_move, self.final_user))
                self._result_timestamp = timestamp_ms
                if self.phase == RoundPhase.MATCH_OVER:
                    assert self.match_winner is not None
                    self._message = f"{self.match_winner.value} WINS THE MATCH"
                else:
                    assert self.outcome is not None
                    self._message = self.outcome.value

        if self.phase in {RoundPhase.RESULT, RoundPhase.MATCH_OVER}:
            assert self._result_timestamp is not None
            result_duration = (
                self.config.match_result_ms
                if self.phase == RoundPhase.MATCH_OVER
                else self.config.result_ms
            )
            if timestamp_ms - self._result_timestamp >= result_duration:
                if not hand_present:
                    self._absent_since = self._absent_since or timestamp_ms
                    if timestamp_ms - self._absent_since >= self.config.clear_ms:
                        if self.phase == RoundPhase.MATCH_OVER:
                            self.reset_match()
                        else:
                            self.reset_round()
                else:
                    self._absent_since = None
                    next_label = (
                        "next match"
                        if self.phase == RoundPhase.MATCH_OVER
                        else "next round"
                    )
                    self._message = f"Remove your hand for the {next_label}"

        return self.view(timestamp_ms)

    def view(self, timestamp_ms: int) -> GameViewState:
        countdown = None
        countdown_label = None
        if self.phase == RoundPhase.COUNTDOWN and self._go_timestamp is not None:
            remaining = max(0, self._go_timestamp - timestamp_ms)
            step_ms = self.config.countdown_ms / len(self.config.countdown_labels)
            countdown = max(1, math.ceil(remaining / step_ms))
            label_index = min(
                len(self.config.countdown_labels) - 1,
                max(0, len(self.config.countdown_labels) - countdown),
            )
            countdown_label = self.config.countdown_labels[label_index]
        ready_progress = 0.0
        if self.phase == RoundPhase.READY and self._hand_since is not None:
            if self.config.hand_stable_ms == 0:
                ready_progress = 1.0
            else:
                ready_progress = float(
                    np.clip(
                        (timestamp_ms - self._hand_since) / self.config.hand_stable_ms,
                        0.0,
                        1.0,
                    )
                )
        probabilities = self._ema.copy() if self._ema is not None else np.zeros(3, dtype=np.float32)
        lead = None
        if self.lock_time_ms is not None:
            lead = self.config.reveal_ms - self.lock_time_ms
        return GameViewState(
            phase=self.phase,
            message=self._message,
            countdown=countdown,
            countdown_label=countdown_label,
            ready_progress=ready_progress,
            probabilities=probabilities,
            locked_user=self.locked_user,
            ai_move=self.ai_move,
            final_user=self.final_user,
            outcome=self.outcome,
            lock_time_ms=self.lock_time_ms,
            prediction_lead_ms=lead,
            temporal_state=self._policy.state,
            lock_reason=self.lock_reason,
            score=self._score_snapshot(),
            match_winner=self.match_winner,
            effect_event=self.effect_event,
            event_id=self._event_id,
        )
