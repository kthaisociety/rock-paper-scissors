from __future__ import annotations

import numpy as np

from rps.game import (
    GameConfig,
    GameController,
    Gesture,
    HandPrediction,
    Outcome,
    RoundPhase,
    counter_move,
    lock_from_probability_trace,
    score_round,
)


def prediction(timestamp: int, values: tuple[float, float, float]) -> HandPrediction:
    return HandPrediction(timestamp, np.asarray(values, dtype=np.float32))


def test_counter_moves_and_scoring() -> None:
    assert counter_move(Gesture.ROCK) == Gesture.PAPER
    assert counter_move(Gesture.PAPER) == Gesture.SCISSORS
    assert counter_move(Gesture.SCISSORS) == Gesture.ROCK
    assert score_round(Gesture.PAPER, Gesture.ROCK) == Outcome.AI_WIN
    assert score_round(Gesture.ROCK, Gesture.ROCK) == Outcome.TIE
    assert score_round(Gesture.SCISSORS, Gesture.ROCK) == Outcome.USER_WIN


def test_trace_locks_early_after_three_stable_predictions() -> None:
    timestamps = np.asarray([150, 200, 250], dtype=np.int64)
    probabilities = np.asarray([[0.8, 0.1, 0.1]] * 3, dtype=np.float32)
    decision = lock_from_probability_trace(timestamps, probabilities)
    assert decision.gesture == Gesture.ROCK
    assert decision.lock_time_ms == 250


def test_trace_force_locks_uncertain_prediction() -> None:
    timestamps = np.asarray([200, 300, 440], dtype=np.int64)
    probabilities = np.asarray(
        [[0.34, 0.33, 0.33], [0.35, 0.34, 0.31], [0.36, 0.35, 0.29]],
        dtype=np.float32,
    )
    decision = lock_from_probability_trace(timestamps, probabilities)
    assert decision.gesture == Gesture.ROCK
    assert decision.lock_time_ms == 450


def test_controller_runs_complete_round() -> None:
    config = GameConfig(hand_stable_ms=0, countdown_ms=0)
    controller = GameController(config)
    state = controller.update(0, prediction(0, (0.8, 0.1, 0.1)))
    assert state.phase == RoundPhase.PREDICTING
    controller.update(200, prediction(200, (0.85, 0.1, 0.05)))
    controller.update(220, prediction(220, (0.86, 0.09, 0.05)))
    state = controller.update(240, prediction(240, (0.87, 0.08, 0.05)))
    assert state.phase == RoundPhase.LOCKED
    assert state.locked_user == Gesture.ROCK
    assert state.ai_move == Gesture.PAPER
    assert state.prediction_lead_ms == 710

    controller.update(650, prediction(650, (0.9, 0.05, 0.05)))
    controller.update(800, prediction(800, (0.9, 0.05, 0.05)))
    state = controller.update(950, prediction(950, (0.9, 0.05, 0.05)))
    assert state.phase == RoundPhase.RESULT
    assert state.outcome == Outcome.AI_WIN


def test_controller_invalidates_round_with_no_prediction() -> None:
    controller = GameController(GameConfig(hand_stable_ms=0, countdown_ms=0))
    controller.update(0, prediction(0, (0.8, 0.1, 0.1)))
    state = controller.update(450, None)
    assert state.phase == RoundPhase.READY
    assert "No hand" in state.message


def test_controller_requires_centered_rock_and_ignores_pre_go_prediction() -> None:
    controller = GameController(GameConfig(hand_stable_ms=0, countdown_ms=0))
    state = controller.update(0, prediction(0, (0.1, 0.8, 0.1)))
    assert state.phase == RoundPhase.READY

    off_center = HandPrediction(10, np.asarray((0.8, 0.1, 0.1)), centered=False)
    state = controller.update(10, off_center)
    assert state.phase == RoundPhase.READY

    state = controller.update(20, prediction(20, (0.8, 0.1, 0.1)))
    assert state.phase == RoundPhase.PREDICTING
    state = controller.update(220, prediction(19, (0.95, 0.03, 0.02)))
    assert state.phase == RoundPhase.PREDICTING
    assert state.probabilities.tolist() == [0.0, 0.0, 0.0]
