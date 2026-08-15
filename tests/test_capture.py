from __future__ import annotations

from collections import Counter

import numpy as np

from rps.cli.capture import _draw_capture_ui, generate_prompts


def test_prompts_are_balanced_without_triples() -> None:
    prompts = generate_prompts(20, seed=42)
    assert Counter(prompts) == {0: 20, 1: 20, 2: 20}
    assert all(
        not (prompts[index] == prompts[index - 1] == prompts[index - 2])
        for index in range(2, len(prompts))
    )


def test_capture_ui_draws_mirrored_hand_joints() -> None:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    landmarks = np.full((21, 3), (0.8, 0.8, 0.0), dtype=np.float32)

    display = _draw_capture_ui(
        frame,
        label="ROCK",
        trial=1,
        total=3,
        phase_text="GO",
        frame_count=1,
        landmarks=landmarks,
    )

    mirrored_x = int((1.0 - 0.8) * (display.shape[1] - 1))
    y = int(0.8 * (display.shape[0] - 1))
    assert np.any(display[y - 8 : y + 9, mirrored_x - 8 : mirrored_x + 9] != 0)
