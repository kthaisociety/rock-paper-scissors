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


def test_invalid_retry_preserves_existing_score_and_round_totals() -> None:
    controller = GameController(GameConfig(hand_stable_ms=0, countdown_ms=0))
    scored = complete_round(
        controller,
        started_ms=0,
        final_values=(0.05, 0.05, 0.9),
    )
    assert scored.score.user_points == 1
    controller.reset_round()
    controller.update(2000, prediction(2000, (0.9, 0.05, 0.05)))
    state = controller.update(2450, None)
    assert state.phase == RoundPhase.READY
    assert state.score.user_points == 1
    assert state.score.rounds_played == 1


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


def complete_round(
    controller: GameController,
    *,
    started_ms: int,
    final_values: tuple[float, float, float],
):
    controller.update(started_ms, prediction(started_ms, (0.9, 0.05, 0.05)))
    controller.update(started_ms + 200, prediction(started_ms + 200, (0.9, 0.05, 0.05)))
    controller.update(started_ms + 220, prediction(started_ms + 220, (0.9, 0.05, 0.05)))
    controller.update(started_ms + 240, prediction(started_ms + 240, (0.9, 0.05, 0.05)))
    controller.update(started_ms + 650, prediction(started_ms + 650, final_values))
    controller.update(started_ms + 800, prediction(started_ms + 800, final_values))
    return controller.update(started_ms + 950, prediction(started_ms + 950, final_values))


def test_default_game_cadence_has_three_fast_named_beats() -> None:
    controller = GameController(GameConfig(hand_stable_ms=0))
    state = controller.update(0, prediction(0, (0.9, 0.05, 0.05)))
    assert state.phase == RoundPhase.COUNTDOWN
    assert (state.countdown, state.countdown_label) == (3, "ROCK")

    state = controller.update(600, prediction(600, (0.9, 0.05, 0.05)))
    assert (state.countdown, state.countdown_label) == (2, "PAPER")
    state = controller.update(1200, prediction(1200, (0.9, 0.05, 0.05)))
    assert (state.countdown, state.countdown_label) == (1, "SCISSORS")
    state = controller.update(1800, prediction(1800, (0.9, 0.05, 0.05)))
    assert state.phase == RoundPhase.PREDICTING
    assert state.countdown_label is None


def test_ready_progress_tracks_centered_fist_hold() -> None:
    controller = GameController(GameConfig(hand_stable_ms=250, countdown_ms=1800))
    state = controller.update(0, prediction(0, (0.9, 0.05, 0.05)))
    assert state.ready_progress == 0.0
    state = controller.update(125, prediction(125, (0.9, 0.05, 0.05)))
    assert state.ready_progress == 0.5


def test_scores_wins_ties_and_streaks_exactly_once() -> None:
    controller = GameController(GameConfig(hand_stable_ms=0, countdown_ms=0))

    state = complete_round(
        controller, started_ms=0, final_values=(0.05, 0.05, 0.9)
    )
    assert state.outcome == Outcome.USER_WIN
    assert state.score.user_points == 1
    assert state.score.user_streak == state.score.best_user_streak == 1
    assert state.score.rounds_played == 1
    repeated = controller.update(960, prediction(960, (0.05, 0.05, 0.9)))
    assert repeated.score.rounds_played == 1
    assert repeated.score.user_points == 1

    controller.reset_round()
    state = complete_round(
        controller, started_ms=2000, final_values=(0.05, 0.9, 0.05)
    )
    assert state.outcome == Outcome.TIE
    assert state.score.ties == 1
    assert state.score.user_points == 1
    assert state.score.user_streak == 1

    controller.reset_round()
    state = complete_round(
        controller, started_ms=4000, final_values=(0.9, 0.05, 0.05)
    )
    assert state.outcome == Outcome.AI_WIN
    assert state.score.ai_points == 1
    assert state.score.ai_streak == 1
    assert state.score.user_streak == 0
    assert state.score.rounds_played == 3


def test_first_to_three_enters_match_over_and_preserves_session_totals() -> None:
    controller = GameController(GameConfig(hand_stable_ms=0, countdown_ms=0))
    for index in range(3):
        if index:
            controller.reset_round()
        state = complete_round(
            controller,
            started_ms=index * 2000,
            final_values=(0.05, 0.05, 0.9),
        )

    assert state.phase == RoundPhase.MATCH_OVER
    assert state.match_winner.value == "YOU"
    assert state.score.user_points == 3
    assert state.score.user_matches == 1
    assert state.score.user_round_wins == 3
    assert state.effect_event == "user_match"

    controller.reset_match()
    state = controller.view(7000)
    assert state.phase == RoundPhase.READY
    assert state.score.user_points == 0
    assert state.score.user_matches == 1
    assert state.score.user_round_wins == 3
    assert state.score.best_user_streak == 3
    assert state.score.user_streak == 0


def test_hand_clear_advances_round_without_erasing_scores() -> None:
    config = GameConfig(
        hand_stable_ms=0,
        countdown_ms=0,
        result_ms=100,
        clear_ms=50,
    )
    controller = GameController(config)
    state = complete_round(
        controller, started_ms=0, final_values=(0.05, 0.05, 0.9)
    )
    assert state.phase == RoundPhase.RESULT
    controller.update(1050, None)
    state = controller.update(1100, None)
    assert state.phase == RoundPhase.READY
    assert state.score.user_points == 1


def test_match_over_waits_for_hand_clear_then_starts_fresh_match() -> None:
    config = GameConfig(
        hand_stable_ms=0,
        countdown_ms=0,
        match_result_ms=100,
        clear_ms=50,
    )
    controller = GameController(config)
    for index in range(3):
        if index:
            controller.reset_round()
        state = complete_round(
            controller,
            started_ms=index * 2000,
            final_values=(0.05, 0.05, 0.9),
        )

    assert state.phase == RoundPhase.MATCH_OVER
    controller.update(5050, None)
    state = controller.update(5100, None)
    assert state.phase == RoundPhase.READY
    assert state.score.user_points == 0
    assert state.score.user_matches == 1
    assert state.score.user_round_wins == 3


def test_session_reset_clears_every_score_counter() -> None:
    controller = GameController(GameConfig(hand_stable_ms=0, countdown_ms=0))
    complete_round(controller, started_ms=0, final_values=(0.05, 0.05, 0.9))
    controller.reset_session()
    state = controller.view(2000)
    assert state.score.user_points == 0
    assert state.score.user_round_wins == 0
    assert state.score.user_matches == 0
    assert state.score.rounds_played == 0
