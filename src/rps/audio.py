from __future__ import annotations

import shutil
import subprocess
import wave
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from rps.constants import RUNTIME_DIR
from rps.game import GameViewState, RoundPhase

SAMPLE_RATE = 22_050


class AudioFeedback:
    """Generate tiny local cues and play them asynchronously with macOS afplay."""

    def __init__(
        self,
        *,
        muted: bool = False,
        runtime_dir: Path | None = None,
        player: Callable[[Path], None] | None = None,
    ) -> None:
        self.muted = muted
        self.runtime_dir = runtime_dir or RUNTIME_DIR / "audio"
        self._player = player
        self._afplay = shutil.which("afplay") if player is None else None
        self._last_phase: RoundPhase | None = None
        self._last_countdown_label: str | None = None
        self._last_event_id = -1

    @property
    def available(self) -> bool:
        return self._player is not None or self._afplay is not None

    def toggle_muted(self) -> bool:
        self.muted = not self.muted
        return self.muted

    @staticmethod
    def _tone(frequency: float, duration: float, volume: float = 0.24) -> np.ndarray:
        count = max(1, int(SAMPLE_RATE * duration))
        timeline = np.arange(count, dtype=np.float64) / SAMPLE_RATE
        envelope = np.minimum(1.0, timeline / 0.012)
        envelope *= np.minimum(1.0, (duration - timeline) / 0.025)
        return volume * envelope * np.sin(2.0 * np.pi * frequency * timeline)

    @classmethod
    def _render_sequence(cls, notes: Sequence[tuple[float, float]]) -> np.ndarray:
        silence = np.zeros(int(SAMPLE_RATE * 0.025), dtype=np.float64)
        parts: list[np.ndarray] = []
        for frequency, duration in notes:
            parts.extend((cls._tone(frequency, duration), silence))
        return np.concatenate(parts) if parts else silence

    @staticmethod
    def _notes(cue: str) -> tuple[tuple[float, float], ...]:
        return {
            "countdown": ((520.0, 0.07),),
            "shoot": ((780.0, 0.06), (980.0, 0.09)),
            "user_win": ((620.0, 0.08), (820.0, 0.12)),
            "ai_win": ((340.0, 0.10), (230.0, 0.14)),
            "tie": ((440.0, 0.08), (440.0, 0.08)),
            "match_win": ((560.0, 0.08), (740.0, 0.08), (980.0, 0.16)),
        }[cue]

    def _ensure_cue(self, cue: str) -> Path:
        path = self.runtime_dir / f"{cue}.wav"
        if path.exists():
            return path
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        samples = self._render_sequence(self._notes(cue))
        pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(SAMPLE_RATE)
            output.writeframes(pcm.tobytes())
        return path

    def play(self, cue: str) -> None:
        if self.muted or not self.available:
            return
        try:
            path = self._ensure_cue(cue)
            if self._player is not None:
                self._player(path)
                return
            assert self._afplay is not None
            subprocess.Popen(
                [self._afplay, str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            # Sound is optional; filesystem or player failures must not stop a match.
            return

    def update(self, state: GameViewState) -> None:
        if (
            state.phase == RoundPhase.COUNTDOWN
            and state.countdown_label is not None
            and state.countdown_label != self._last_countdown_label
        ):
            self.play("countdown")
        if state.phase == RoundPhase.PREDICTING and self._last_phase != RoundPhase.PREDICTING:
            self.play("shoot")
        if state.effect_event is not None and state.event_id != self._last_event_id:
            cue = {
                "user_win": "user_win",
                "ai_win": "ai_win",
                "tie": "tie",
                "user_match": "match_win",
                "ai_match": "ai_win",
            }.get(state.effect_event)
            if cue is not None:
                self.play(cue)
            self._last_event_id = state.event_id
        self._last_phase = state.phase
        self._last_countdown_label = (
            state.countdown_label if state.phase == RoundPhase.COUNTDOWN else None
        )
