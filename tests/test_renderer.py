from __future__ import annotations

import numpy as np

from rps.game import GameViewState, RoundPhase
from rps.model import GestureMLP, default_activation_scales
from rps.renderer import BoothRenderer, NetworkSnapshot, PerformanceStats


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
