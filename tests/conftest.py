from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def hand_landmarks() -> np.ndarray:
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = (0.0, 0.0, 0.0)
    finger_x = (-0.7, -0.35, 0.0, 0.35, 0.7)
    starts = (1, 5, 9, 13, 17)
    for x, start in zip(finger_x, starts, strict=True):
        for offset in range(4):
            points[start + offset] = (x, -0.65 - 0.35 * offset, -0.03 * offset)
    points[9] = (0.0, -1.0, 0.0)
    return points
