from __future__ import annotations

import cv2
import numpy as np

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (17, 0),
)


def draw_mirrored_hand(
    image: np.ndarray,
    landmarks: np.ndarray | None,
    *,
    line_color: tuple[int, int, int] = (70, 230, 180),
    joint_color: tuple[int, int, int] = (245, 250, 250),
    outline_color: tuple[int, int, int] = (40, 150, 120),
    line_thickness: int = 3,
    joint_radius: int = 5,
) -> None:
    """Draw original-camera normalized landmarks on an already mirrored image."""

    points = np.asarray(landmarks) if landmarks is not None else np.empty((0, 3))
    if points.shape != (21, 3) or not np.isfinite(points[:, :2]).all():
        return
    height, width = image.shape[:2]
    positions = [
        (
            int(np.clip((1.0 - float(point[0])) * (width - 1), 0, width - 1)),
            int(np.clip(float(point[1]) * (height - 1), 0, height - 1)),
        )
        for point in points
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(
            image,
            positions[start],
            positions[end],
            line_color,
            line_thickness,
            cv2.LINE_AA,
        )
    for position in positions:
        cv2.circle(image, position, joint_radius, joint_color, -1, cv2.LINE_AA)
        cv2.circle(
            image,
            position,
            joint_radius + 2,
            outline_color,
            2,
            cv2.LINE_AA,
        )
