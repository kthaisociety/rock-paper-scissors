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
            "I PREDICT ROCK",
            locked_user=Gesture.ROCK,
            ai_move=Gesture.PAPER,
            lock_time_ms=246,
            score=score,
        ),
        GameViewState(
            RoundPhase.RESULT,
            "YOU FOOLED THE AI!",
            locked_user=Gesture.ROCK,
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
            locked_user=Gesture.ROCK,
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


def test_locked_prediction_and_latency_are_visible_without_revealing_response(
    monkeypatch,
) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    state = GameViewState(
        RoundPhase.LOCKED,
        "I PREDICT PAPER",
        locked_user=Gesture.PAPER,
        ai_move=Gesture.SCISSORS,
        lock_time_ms=246,
    )
    snapshot = NetworkSnapshot(trained=True)

    def recorder_for(rendered_text: list[str]):
        def record_text(_image, text, *_args, **_kwargs) -> None:
            rendered_text.append(text)

        return record_text

    for mode in RenderMode:
        renderer = BoothRenderer(GestureMLP(), default_activation_scales())
        rendered_text: list[str] = []
        record_text = recorder_for(rendered_text)
        monkeypatch.setattr(renderer, "_put_centered", record_text)
        monkeypatch.setattr(renderer, "_put_text", record_text)
        renderer.render(frame, state, snapshot, mode=mode)
        combined = " ".join(rendered_text)
        assert "I PREDICT PAPER" in combined
        assert "LOCKED IN 246 MS" in combined
        assert "RESPONSE LOCKED" in combined
        assert "AI SCISSORS" not in combined


def test_lock_flash_triggers_once_on_phase_entry(monkeypatch) -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    renderer = BoothRenderer(GestureMLP(), default_activation_scales())
    state = GameViewState(
        RoundPhase.LOCKED,
        "I PREDICT ROCK",
        locked_user=Gesture.ROCK,
        ai_move=Gesture.PAPER,
        lock_time_ms=246,
    )
    snapshot = NetworkSnapshot(trained=True)
    clock = [1.0]
    monkeypatch.setattr("rps.renderer.time.monotonic", lambda: clock[0])

    first_frame = renderer.render(frame, state, snapshot)
    clock[0] = 1.3
    later_frame = renderer.render(frame, state, snapshot)

    assert float(np.mean(first_frame)) > float(np.mean(later_frame))
