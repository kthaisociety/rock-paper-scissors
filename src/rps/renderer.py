from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from rps.features import LANDMARK_GROUPS, group_feature_indices, summarize_feature_groups
from rps.game import GameViewState, RoundPhase
from rps.model import CLASS_NAMES, GestureMLP

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
    (0, 17),
)


@dataclass(slots=True)
class NetworkSnapshot:
    features: np.ndarray = field(default_factory=lambda: np.zeros(63, dtype=np.float32))
    act1: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))
    act2: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    hand_landmarks: np.ndarray | None = None
    trained: bool = False
    device: str = "cpu"
    inference_ms: float = 0.0
    result_age_ms: float = 0.0


@dataclass(slots=True)
class PerformanceStats:
    fps: float = 0.0


class BoothRenderer:
    def __init__(
        self,
        model: GestureMLP,
        activation_scales: dict[str, list[float]],
    ) -> None:
        self.model = model
        self.activation_scales = activation_scales
        self._weights1 = model.fc1.weight.detach().cpu().numpy()
        self._weights2 = model.fc2.weight.detach().cpu().numpy()
        self._weights3 = model.fc3.weight.detach().cpu().numpy()

    @staticmethod
    def _positions(x: int, top: int, bottom: int, count: int) -> list[tuple[int, int]]:
        if count == 1:
            return [(x, (top + bottom) // 2)]
        return [(x, int(top + index * (bottom - top) / (count - 1))) for index in range(count)]

    @staticmethod
    def _color(intensity: float, *, output: bool = False) -> tuple[int, int, int]:
        value = float(np.clip(intensity, 0.0, 1.0))
        if output:
            return (int(30 + 40 * value), int(100 + 155 * value), int(40 + 100 * value))
        return (int(60 + 180 * value), int(60 + 195 * value), int(20 + 80 * value))

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float = 0.55,
        color: tuple[int, int, int] = (235, 245, 245),
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_connections(
        self,
        image: np.ndarray,
        source_positions: list[tuple[int, int]],
        destination_positions: list[tuple[int, int]],
        contributions: np.ndarray,
    ) -> None:
        maximum = max(float(np.percentile(contributions, 95)), 1e-6)
        for source_index, source in enumerate(source_positions):
            for destination_index, destination in enumerate(destination_positions):
                value = float(contributions[destination_index, source_index]) / maximum
                cv2.line(image, source, destination, self._color(value), 1, cv2.LINE_AA)

    def _draw_network(
        self,
        image: np.ndarray,
        snapshot: NetworkSnapshot,
        panel_right: int,
    ) -> None:
        top, bottom = 100, min(image.shape[0] - 70, 650)
        columns = [75, 210, 355, min(500, panel_right - 75)]
        input_positions = self._positions(columns[0], top + 25, bottom - 25, 6)
        layer1_positions = self._positions(columns[1], top, bottom, 16)
        layer2_positions = self._positions(columns[2], top + 15, bottom - 15, 8)
        output_positions = self._positions(columns[3], top + 100, bottom - 100, 3)

        group_indices = list(group_feature_indices().values())
        input_contributions = np.zeros((16, 6), dtype=np.float32)
        for destination in range(16):
            for group, indices in enumerate(group_indices):
                input_contributions[destination, group] = float(
                    np.mean(
                        np.abs(self._weights1[destination, indices] * snapshot.features[indices])
                    )
                )
        layer1_contributions = np.abs(self._weights2 * snapshot.act1[np.newaxis, :])
        layer2_contributions = np.abs(self._weights3 * snapshot.act2[np.newaxis, :])

        self._draw_connections(image, input_positions, layer1_positions, input_contributions)
        self._draw_connections(image, layer1_positions, layer2_positions, layer1_contributions)
        self._draw_connections(image, layer2_positions, output_positions, layer2_contributions)

        input_values = summarize_feature_groups(snapshot.features)
        input_values /= max(float(np.percentile(input_values, 95)), 1e-6)
        scales1 = np.maximum(np.asarray(self.activation_scales["act1"]), 1e-6)
        scales2 = np.maximum(np.asarray(self.activation_scales["act2"]), 1e-6)
        node_layers = (
            (input_positions, input_values, False),
            (layer1_positions, np.abs(snapshot.act1) / scales1, False),
            (layer2_positions, np.abs(snapshot.act2) / scales2, False),
            (output_positions, snapshot.probabilities, True),
        )
        for positions, values, output in node_layers:
            for position, value in zip(positions, values, strict=True):
                color = self._color(float(value), output=output)
                cv2.circle(image, position, 8 if output else 6, color, -1, cv2.LINE_AA)
                cv2.circle(image, position, 9 if output else 7, (200, 220, 220), 1, cv2.LINE_AA)

        for position, name in zip(input_positions, LANDMARK_GROUPS, strict=True):
            self._put_text(image, name, (position[0] - 50, position[1] - 11), 0.33, (150, 180, 190))
        for index, (position, name) in enumerate(zip(output_positions, CLASS_NAMES, strict=True)):
            confidence = int(round(float(snapshot.probabilities[index]) * 100))
            self._put_text(
                image,
                f"{name} {confidence}%",
                (position[0] + 15, position[1] + 5),
                0.48,
                (180, 255, 210),
            )

        for x, label in zip(columns, ("FEATURES", "LAYER 1", "LAYER 2", "OUTPUT"), strict=True):
            self._put_text(image, label, (x - 35, 88), 0.38, (130, 190, 200))

    def _draw_hand(self, image: np.ndarray, landmarks: np.ndarray | None) -> None:
        if landmarks is None or np.asarray(landmarks).shape != (21, 3):
            return
        height, width = image.shape[:2]
        positions = [
            (width - 1 - int(float(point[0]) * width), int(float(point[1]) * height))
            for point in landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, positions[start], positions[end], (70, 230, 180), 3, cv2.LINE_AA)
        for position in positions:
            cv2.circle(image, position, 5, (245, 250, 250), -1, cv2.LINE_AA)
            cv2.circle(image, position, 7, (40, 150, 120), 2, cv2.LINE_AA)

    def _draw_game(
        self, image: np.ndarray, state: GameViewState, snapshot: NetworkSnapshot
    ) -> None:
        height, width = image.shape[:2]
        center_x = max(650, width * 3 // 4)
        cv2.rectangle(image, (center_x - 250, 20), (width - 20, 160), (15, 22, 30), -1)
        self._put_text(image, state.phase.value, (center_x - 225, 50), 0.65, (80, 230, 220), 2)
        self._put_text(image, state.message, (center_x - 225, 85), 0.65, (245, 245, 245), 2)
        if state.countdown is not None:
            self._put_text(
                image,
                str(state.countdown),
                (center_x - 20, 145),
                1.8,
                (80, 255, 220),
                3,
            )
        if state.phase == RoundPhase.RESULT:
            user = state.final_user.name if state.final_user is not None else "?"
            ai = state.ai_move.name if state.ai_move is not None else "?"
            self._put_text(image, f"YOU: {user}   AI: {ai}", (center_x - 225, 125), 0.58)
        elif state.ai_move is not None:
            self._put_text(image, "AI move sealed until reveal", (center_x - 225, 125), 0.5)

        status_y = height - 35
        status = (
            f"{snapshot.device.upper()}  infer {snapshot.inference_ms:.2f} ms  "
            f"result age {snapshot.result_age_ms:.0f} ms"
        )
        self._put_text(image, status, (20, status_y), 0.45, (160, 190, 200))

    def render(
        self,
        frame_bgr: np.ndarray,
        state: GameViewState,
        snapshot: NetworkSnapshot,
        performance: PerformanceStats | None = None,
    ) -> np.ndarray:
        performance = performance or PerformanceStats()
        image = cv2.flip(frame_bgr, 1)
        height, width = image.shape[:2]
        panel_right = min(620, width - 20)
        overlay = image.copy()
        cv2.rectangle(overlay, (15, 15), (panel_right, height - 15), (8, 17, 24), -1)
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0.0, image)
        cv2.rectangle(image, (15, 15), (panel_right, 65), (10, 30, 40), -1)
        self._put_text(
            image,
            "LIVE NEURAL NET ACTIVATIONS",
            (35, 49),
            0.68,
            (100, 255, 225),
            2,
        )
        self._draw_network(image, snapshot, panel_right)
        self._draw_hand(image, snapshot.hand_landmarks)
        self._draw_game(image, state, snapshot)
        self._put_text(image, f"FPS {performance.fps:.1f}", (panel_right - 95, 48), 0.45)
        if not snapshot.trained:
            cv2.rectangle(
                image,
                (width // 2 - 250, height - 90),
                (width // 2 + 250, height - 45),
                (0, 0, 130),
                -1,
            )
            self._put_text(
                image,
                "UNTRAINED - VISUALIZATION MODE - SCORING DISABLED",
                (width // 2 - 235, height - 60),
                0.52,
                (255, 255, 255),
                2,
            )
        return image
