from __future__ import annotations

import wave

from rps.audio import SAMPLE_RATE, AudioFeedback
from rps.game import GameViewState, RoundPhase


def state(
    phase: RoundPhase,
    *,
    countdown_label: str | None = None,
    effect_event: str | None = None,
    event_id: int = 0,
) -> GameViewState:
    return GameViewState(
        phase,
        phase.value,
        countdown_label=countdown_label,
        effect_event=effect_event,
        event_id=event_id,
    )


def test_audio_routes_each_transition_once_and_writes_valid_wav(tmp_path) -> None:
    played = []
    audio = AudioFeedback(runtime_dir=tmp_path, player=lambda path: played.append(path))

    audio.update(state(RoundPhase.COUNTDOWN, countdown_label="ROCK"))
    audio.update(state(RoundPhase.COUNTDOWN, countdown_label="ROCK"))
    audio.update(state(RoundPhase.COUNTDOWN, countdown_label="PAPER"))
    audio.update(state(RoundPhase.PREDICTING))
    audio.update(state(RoundPhase.LOCKED))
    audio.update(state(RoundPhase.LOCKED))
    audio.update(state(RoundPhase.RESULT, effect_event="user_win", event_id=1))
    audio.update(state(RoundPhase.RESULT, effect_event="user_win", event_id=1))
    audio.update(state(RoundPhase.MATCH_OVER, effect_event="user_match", event_id=2))

    assert [path.name for path in played] == [
        "countdown.wav",
        "countdown.wav",
        "shoot.wav",
        "lock.wav",
        "user_win.wav",
        "match_win.wav",
    ]
    with wave.open(str(tmp_path / "match_win.wav"), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnframes() > 0


def test_muted_audio_tracks_state_without_playing(tmp_path) -> None:
    played = []
    audio = AudioFeedback(
        muted=True,
        runtime_dir=tmp_path,
        player=lambda path: played.append(path),
    )
    audio.update(state(RoundPhase.COUNTDOWN, countdown_label="ROCK"))
    audio.update(state(RoundPhase.PREDICTING))
    audio.update(state(RoundPhase.LOCKED))
    assert not played
    assert audio.toggle_muted() is False
    audio.update(state(RoundPhase.RESULT, effect_event="ai_win", event_id=1))
    assert [path.name for path in played] == ["ai_win.wav"]
