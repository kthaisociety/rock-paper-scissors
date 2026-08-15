from __future__ import annotations

import numpy as np

from rps.game import GameViewState, Gesture, MatchWinner, Outcome, RoundPhase, ScoreSnapshot
from rps.model import GestureMLP, default_activation_scales
from rps.renderer import BoothRenderer, NetworkSnapshot, PerformanceStats, RenderMode


def test_renderer_handles_no_hand_and_extreme_activations() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    renderer = BoothRenderer(GestureMLP(), default_activation_scales())
    state = GameViewState(RoundPhase.READY, "Hold a closed fist")
    snapshot = NetworkSnapshot(
        features=np.full(63, 1e6, dtype=np.float32),
        act1=np.full(16, 1e6, dtype=np.float32),
        act2=np.full(8, 1e6, dtype=np.float32),
        probabilities=np.asarray([0.2, 0.3, 0.5], dtype=np.float32),
        hand_landmarks=None,
        trained=False,
    )
    rendered = renderer.render(frame, state, snapshot, PerformanceStats(fps=30.0))
    assert rendered.shape == frame.shape
    assert rendered.dtype == np.uint8
    assert np.any(rendered)


def test_renderer_handles_every_game_phase_in_both_modes() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    renderer = BoothRenderer(GestureMLP(), default_activation_scales())
    snapshot = NetworkSnapshot(
        probabilities=np.asarray([0.1, 0.2, 0.7], dtype=np.float32),
        trained=True,
    )
    score = ScoreSnapshot(
        user_points=2,
        ai_points=1,
        user_streak=2,
        best_user_streak=2,
        user_matches=1,
        ties=1,
    )
    states = [
        GameViewState(RoundPhase.READY, "Hold a fist", ready_progress=0.5, score=score),
        GameViewState(
            RoundPhase.COUNTDOWN,
            "Get ready",
            countdown=2,
            countdown_label="PAPER",
            score=score,
        ),
        GameViewState(RoundPhase.PREDICTING, "Shoot", score=score),
        GameViewState(
            RoundPhase.LOCKED,
            "AI move sealed",
            ai_move=Gesture.PAPER,
            score=score,
        ),
        GameViewState(
            RoundPhase.RESULT,
            "YOU WIN",
            ai_move=Gesture.PAPER,
            final_user=Gesture.SCISSORS,
            outcome=Outcome.USER_WIN,
            score=score,
            effect_event="user_win",
            event_id=1,
        ),
        GameViewState(
            RoundPhase.MATCH_OVER,
            "YOU WIN THE MATCH",
            ai_move=Gesture.PAPER,
            final_user=Gesture.SCISSORS,
            outcome=Outcome.USER_WIN,
            score=score,
            match_winner=MatchWinner.USER,
            effect_event="user_match",
            event_id=2,
        ),
    ]
    for mode in RenderMode:
        for state in states:
            rendered = renderer.render(frame, state, snapshot, mode=mode)
            assert rendered.shape == frame.shape
            assert rendered.dtype == np.uint8
            assert np.any(rendered)
