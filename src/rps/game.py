from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

import numpy as np


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


class Outcome(StrEnum):
    AI_WIN = "AI WINS"
    TIE = "TIE"
    USER_WIN = "YOU WIN"


@dataclass(frozen=True, slots=True)
class HandPrediction:
    timestamp_ms: int
    probabilities: np.ndarray
    centered: bool = True


@dataclass(frozen=True, slots=True)
class GameConfig:
    hand_stable_ms: int = 500
    countdown_ms: int = 3000
    inference_start_ms: int = 150
    early_lock_start_ms: int = 200
    force_lock_ms: int = 450
    final_start_ms: int = 650
    reveal_ms: int = 950
    result_ms: int = 2000
    clear_ms: int = 300
    ema_alpha: float = 0.4
    stable_results: int = 3
    early_confidence: float = 0.70
    early_margin: float = 0.15
    ready_rock_confidence: float = 0.55


@dataclass(slots=True)
class GameViewState:
    phase: RoundPhase
    message: str
    countdown: int | None = None
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    locked_user: Gesture | None = None
    ai_move: Gesture | None = None
    final_user: Gesture | None = None
    outcome: Outcome | None = None
    lock_time_ms: int | None = None
    prediction_lead_ms: int | None = None


def counter_move(gesture: Gesture) -> Gesture:
    return Gesture((int(gesture) + 1) % 3)


def score_round(ai_move: Gesture, user_move: Gesture) -> Outcome:
    if ai_move == user_move:
        return Outcome.TIE
    if ai_move == counter_move(user_move):
        return Outcome.AI_WIN
    return Outcome.USER_WIN


@dataclass(frozen=True, slots=True)
class LockDecision:
    gesture: Gesture | None
    lock_time_ms: int | None
    probabilities: np.ndarray


def lock_from_probability_trace(
    timestamps_ms: np.ndarray,
    probabilities: np.ndarray,
    config: GameConfig | None = None,
) -> LockDecision:
    config = config or GameConfig()
    ema: np.ndarray | None = None
    leaders: list[int] = []
    for timestamp, values in zip(timestamps_ms, probabilities, strict=True):
        elapsed = int(timestamp)
        if elapsed < config.inference_start_ms or elapsed > config.force_lock_ms:
            continue
        vector = np.asarray(values, dtype=np.float32)
        ema = vector if ema is None else config.ema_alpha * vector + (1 - config.ema_alpha) * ema
        leader = int(np.argmax(ema))
        leaders.append(leader)
        leaders = leaders[-config.stable_results :]
        sorted_values = np.sort(ema)
        margin = float(sorted_values[-1] - sorted_values[-2])
        stable = len(leaders) == config.stable_results and len(set(leaders)) == 1
        if (
            elapsed >= config.early_lock_start_ms
            and stable
            and float(ema[leader]) >= config.early_confidence
            and margin >= config.early_margin
        ):
            return LockDecision(Gesture(leader), elapsed, ema.copy())
    if ema is None:
        return LockDecision(None, None, np.zeros(3, dtype=np.float32))
    return LockDecision(Gesture(int(np.argmax(ema))), config.force_lock_ms, ema.copy())


class GameController:
    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config or GameConfig()
        self.reset()

    def reset(self) -> None:
        self.phase = RoundPhase.READY
        self._hand_since: int | None = None
        self._countdown_started: int | None = None
        self._go_timestamp: int | None = None
        self._result_timestamp: int | None = None
        self._absent_since: int | None = None
        self._ema: np.ndarray | None = None
        self._leaders: list[int] = []
        self._final_probabilities: list[np.ndarray] = []
        self._last_prediction_timestamp = -1
        self.locked_user: Gesture | None = None
        self.ai_move: Gesture | None = None
        self.final_user: Gesture | None = None
        self.outcome: Outcome | None = None
        self.lock_time_ms: int | None = None
        self.probability_trace: list[tuple[int, list[float]]] = []
        self._message = "Hold a closed fist in the frame"

    def start_countdown(self, timestamp_ms: int) -> None:
        self.phase = RoundPhase.COUNTDOWN
        self._countdown_started = timestamp_ms
        self._go_timestamp = timestamp_ms + self.config.countdown_ms
        self._message = "Get ready"

    def _lock(self, elapsed_ms: int) -> None:
        if self._ema is None:
            return
        self.locked_user = Gesture(int(np.argmax(self._ema)))
        self.ai_move = counter_move(self.locked_user)
        self.lock_time_ms = elapsed_ms
        self.phase = RoundPhase.LOCKED
        self._message = "AI MOVE LOCKED"

    def _prediction_update(self, prediction: HandPrediction, elapsed_ms: int) -> None:
        values = np.asarray(prediction.probabilities, dtype=np.float32).reshape(3)
        self._ema = (
            values
            if self._ema is None
            else self.config.ema_alpha * values + (1.0 - self.config.ema_alpha) * self._ema
        )
        leader = int(np.argmax(self._ema))
        self._leaders.append(leader)
        self._leaders = self._leaders[-self.config.stable_results :]
        self.probability_trace.append((elapsed_ms, values.tolist()))

    def _should_lock_early(self, elapsed_ms: int) -> bool:
        if self._ema is None or elapsed_ms < self.config.early_lock_start_ms:
            return False
        stable = len(self._leaders) == self.config.stable_results and len(set(self._leaders)) == 1
        sorted_values = np.sort(self._ema)
        margin = float(sorted_values[-1] - sorted_values[-2])
        return (
            stable
            and float(sorted_values[-1]) >= self.config.early_confidence
            and margin >= self.config.early_margin
        )

    def update(self, timestamp_ms: int, hand_result: HandPrediction | None) -> GameViewState:
        hand_present = hand_result is not None

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
                and prediction_elapsed_ms >= self.config.inference_start_ms
                and self.phase == RoundPhase.PREDICTING
            ):
                self._prediction_update(hand_result, prediction_elapsed_ms)
                if self._should_lock_early(prediction_elapsed_ms):
                    self._lock(prediction_elapsed_ms)
            if self.phase == RoundPhase.PREDICTING and elapsed_ms >= self.config.force_lock_ms:
                if self._ema is None:
                    self.reset()
                    self._message = "No hand detected - try again"
                    return self.view(timestamp_ms)
                self._lock(self.config.force_lock_ms)
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
                    self.reset()
                    self._message = "Final gesture missing - try again"
                    return self.view(timestamp_ms)
                final_mean = np.mean(np.stack(self._final_probabilities), axis=0)
                self.final_user = Gesture(int(np.argmax(final_mean)))
                assert self.ai_move is not None
                self.outcome = score_round(self.ai_move, self.final_user)
                self.phase = RoundPhase.RESULT
                self._result_timestamp = timestamp_ms
                self._message = self.outcome.value

        if self.phase == RoundPhase.RESULT:
            assert self._result_timestamp is not None
            if timestamp_ms - self._result_timestamp >= self.config.result_ms:
                if not hand_present:
                    self._absent_since = self._absent_since or timestamp_ms
                    if timestamp_ms - self._absent_since >= self.config.clear_ms:
                        self.reset()
                else:
                    self._absent_since = None
                    self._message = "Remove your hand for the next round"

        return self.view(timestamp_ms)

    def view(self, timestamp_ms: int) -> GameViewState:
        countdown = None
        if self.phase == RoundPhase.COUNTDOWN and self._go_timestamp is not None:
            countdown = max(1, math.ceil((self._go_timestamp - timestamp_ms) / 1000))
        probabilities = self._ema.copy() if self._ema is not None else np.zeros(3, dtype=np.float32)
        lead = None
        if self.lock_time_ms is not None:
            lead = self.config.reveal_ms - self.lock_time_ms
        return GameViewState(
            phase=self.phase,
            message=self._message,
            countdown=countdown,
            probabilities=probabilities,
            locked_user=self.locked_user,
            ai_move=self.ai_move,
            final_user=self.final_user,
            outcome=self.outcome,
            lock_time_ms=self.lock_time_ms,
            prediction_lead_ms=lead,
        )
