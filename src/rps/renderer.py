from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum

import cv2
import numpy as np

from rps.features import LANDMARK_GROUPS, group_feature_indices, summarize_feature_groups
from rps.game import GameViewState, Outcome, RoundPhase, prediction_result_message
from rps.landmark_drawing import draw_mirrored_hand
from rps.model import CLASS_NAMES, GestureMLP


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


class RenderMode(StrEnum):
    GAME = "game"
    NETWORK = "network"


@dataclass(frozen=True, slots=True)
class _Particle:
    x: float
    y: float
    vx: float
    vy: float
    size: int
    color: tuple[int, int, int]


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
        self._effect_event_id = -1
        self._effect_started = 0.0
        self._particles: list[_Particle] = []
        self._last_phase: RoundPhase | None = None
        self._lock_started = float("-inf")

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

    @staticmethod
    def _put_centered(
        image: np.ndarray,
        text: str,
        center: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        origin = (center[0] - size[0] // 2, center[1] + size[1] // 2)
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

    @staticmethod
    def _overlay_rect(
        image: np.ndarray,
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int],
        alpha: float,
    ) -> None:
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)

    @staticmethod
    def _outcome_color(state: GameViewState) -> tuple[int, int, int]:
        if state.phase == RoundPhase.MATCH_OVER:
            return (30, 190, 250)
        if state.outcome == Outcome.USER_WIN:
            return (80, 225, 110)
        if state.outcome == Outcome.AI_WIN:
            return (80, 90, 245)
        return (70, 210, 230)

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
        draw_mirrored_hand(image, landmarks)

    def _track_phase_effects(self, state: GameViewState) -> None:
        if state.phase == RoundPhase.LOCKED and self._last_phase != RoundPhase.LOCKED:
            self._lock_started = time.monotonic()
        self._last_phase = state.phase

    def _draw_lock_flash(self, image: np.ndarray) -> None:
        elapsed = time.monotonic() - self._lock_started
        if not 0.0 <= elapsed < 0.18:
            return
        height, width = image.shape[:2]
        self._overlay_rect(
            image,
            (0, 0),
            (width, height),
            (230, 220, 60),
            0.14 * (1.0 - elapsed / 0.18),
        )

    def _draw_scoreboard(
        self,
        image: np.ndarray,
        state: GameViewState,
        *,
        compact: bool = False,
    ) -> None:
        height, width = image.shape[:2]
        if compact:
            left, right, top, bottom = max(630, width - 620), width - 20, 20, 105
            score_scale = 1.15
            label_scale = 0.42
        else:
            left, right, top, bottom = 30, width - 30, 20, 152
            score_scale = 1.8
            label_scale = 0.52
        self._overlay_rect(image, (left, top), (right, bottom), (8, 17, 27), 0.88)
        center_x = (left + right) // 2
        punch = 1.0
        if state.effect_event is not None and time.monotonic() - self._effect_started < 0.35:
            punch = 1.12
        self._put_centered(
            image,
            f"{state.score.user_points}     -     {state.score.ai_points}",
            (center_x, top + 48),
            score_scale * punch,
            (245, 250, 250),
            3,
        )
        self._put_text(image, "YOU", (left + 24, top + 43), label_scale, (90, 235, 125), 2)
        ai_label_size, _ = cv2.getTextSize(
            "AI", cv2.FONT_HERSHEY_SIMPLEX, label_scale, 2
        )
        self._put_text(
            image,
            "AI",
            (right - ai_label_size[0] - 24, top + 43),
            label_scale,
            (100, 120, 255),
            2,
        )
        self._put_centered(
            image,
            f"FIRST TO {state.score.target_points}",
            (center_x, bottom - (19 if compact else 44)),
            0.42 if compact else 0.50,
            (110, 220, 225),
            1,
        )
        if not compact:
            session = (
                f"MATCHES  YOU {state.score.user_matches} - {state.score.ai_matches} AI"
                f"     TIES {state.score.ties}"
            )
            self._put_centered(
                image,
                session,
                (center_x, bottom - 16),
                0.42,
                (175, 200, 210),
                1,
            )

    def _draw_progress_bar(self, image: np.ndarray, progress: float) -> None:
        height, width = image.shape[:2]
        left, right = width // 2 - 210, width // 2 + 210
        top, bottom = height // 2 + 65, height // 2 + 86
        cv2.rectangle(image, (left, top), (right, bottom), (55, 70, 78), -1)
        filled = left + int((right - left) * float(np.clip(progress, 0.0, 1.0)))
        cv2.rectangle(image, (left, top), (filled, bottom), (80, 230, 210), -1)
        cv2.rectangle(image, (left, top), (right, bottom), (190, 220, 220), 1)

    def _draw_gesture_card(
        self,
        image: np.ndarray,
        *,
        center: tuple[int, int],
        title: str,
        gesture: str,
        color: tuple[int, int, int],
    ) -> None:
        width, height = 280, 150
        left, top = center[0] - width // 2, center[1] - height // 2
        right, bottom = center[0] + width // 2, center[1] + height // 2
        self._overlay_rect(image, (left, top), (right, bottom), (10, 20, 30), 0.88)
        cv2.rectangle(image, (left, top), (right, bottom), color, 3, cv2.LINE_AA)
        self._put_centered(image, title, (center[0], top + 28), 0.45, (190, 210, 220), 1)
        self._put_centered(image, gesture, (center[0], center[1] + 18), 1.05, color, 3)

    def _update_effect(self, image: np.ndarray, state: GameViewState) -> None:
        if state.effect_event is not None and state.event_id != self._effect_event_id:
            self._effect_event_id = state.event_id
            self._effect_started = time.monotonic()
            rng = np.random.default_rng(state.event_id)
            if state.effect_event == "tie":
                palette = [(40, 190, 245), (70, 220, 255), (120, 235, 250)]
            elif state.effect_event.startswith("user"):
                palette = [(80, 230, 110), (120, 255, 180), (30, 190, 250)]
            else:
                palette = [(80, 90, 245), (120, 150, 255), (70, 210, 230)]
            self._particles = [
                _Particle(
                    x=float(rng.uniform(0.15, 0.85)),
                    y=float(rng.uniform(0.15, 0.55)),
                    vx=float(rng.uniform(-0.16, 0.16)),
                    vy=float(rng.uniform(-0.25, -0.08)),
                    size=int(rng.integers(3, 8)),
                    color=palette[int(rng.integers(0, len(palette)))],
                )
                for _ in range(54)
            ]
        elapsed = time.monotonic() - self._effect_started
        if not self._particles or elapsed > 1.4:
            return
        height, width = image.shape[:2]
        if elapsed < 0.18:
            flash_color = self._outcome_color(state)
            self._overlay_rect(
                image,
                (0, 0),
                (width, height),
                flash_color,
                0.12 * (1.0 - elapsed / 0.18),
            )
        for particle in self._particles:
            x = int((particle.x + particle.vx * elapsed) * width)
            y = int((particle.y + particle.vy * elapsed + 0.42 * elapsed**2) * height)
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(image, (x, y), particle.size, particle.color, -1, cv2.LINE_AA)

    def _draw_game_first(
        self,
        image: np.ndarray,
        state: GameViewState,
        snapshot: NetworkSnapshot,
        performance: PerformanceStats,
    ) -> None:
        height, width = image.shape[:2]
        self._overlay_rect(image, (0, 0), (width, height), (5, 12, 20), 0.48)
        self._draw_hand(image, snapshot.hand_landmarks)
        self._draw_scoreboard(image, state)

        center = (width // 2, height // 2 - 20)
        accent = (80, 230, 215)
        if state.phase == RoundPhase.READY:
            self._put_centered(image, "HOLD A FIST TO START", center, 1.05, accent, 3)
            self._put_centered(
                image,
                "Can you fool the predicting AI?",
                (center[0], center[1] + 55),
                0.55,
                (220, 230, 235),
                1,
            )
            self._draw_progress_bar(image, state.ready_progress)
        elif state.phase == RoundPhase.COUNTDOWN:
            self._put_centered(
                image,
                state.countdown_label or "GET READY",
                center,
                2.0,
                accent,
                4,
            )
            self._put_centered(
                image,
                str(state.countdown or ""),
                (center[0], center[1] + 95),
                0.85,
                (210, 235, 235),
                2,
            )
        elif state.phase == RoundPhase.PREDICTING:
            self._put_centered(image, "SHOOT!", center, 2.2, (70, 240, 220), 5)
            self._put_centered(
                image,
                "AI is reading your move...",
                (center[0], center[1] + 75),
                0.60,
                (220, 235, 235),
                1,
            )
        elif state.phase == RoundPhase.LOCKED:
            predicted = state.locked_user.name if state.locked_user is not None else "?"
            self._put_centered(
                image,
                f"I PREDICT {predicted}",
                (center[0], center[1] - 25),
                1.45,
                (70, 235, 225),
                4,
            )
            self._put_centered(
                image,
                f"LOCKED IN {state.lock_time_ms or 0} MS",
                (center[0], center[1] + 30),
                0.62,
                (230, 245, 245),
                2,
            )
            self._put_centered(
                image,
                "Finish your gesture - or change it!",
                (center[0], center[1] + 72),
                0.52,
                (230, 240, 240),
                1,
            )
            self._draw_gesture_card(
                image,
                center=(width // 2 - 180, height - 195),
                title="PREDICTION",
                gesture=predicted,
                color=(70, 235, 225),
            )
            self._draw_gesture_card(
                image,
                center=(width // 2 + 180, height - 195),
                title="AI RESPONSE",
                gesture="LOCKED",
                color=(70, 190, 245),
            )
        else:
            result_color = self._outcome_color(state)
            title_y = center[1] - (40 if state.phase == RoundPhase.MATCH_OVER else 10)
            self._put_centered(
                image,
                state.message,
                (center[0], title_y),
                1.25,
                result_color,
                3,
            )
            user = state.final_user.name if state.final_user is not None else "?"
            ai = state.ai_move.name if state.ai_move is not None else "?"
            predicted = state.locked_user.name if state.locked_user is not None else "?"
            comparison_y = title_y + 62
            if (
                state.phase == RoundPhase.MATCH_OVER
                and state.locked_user is not None
                and state.final_user is not None
                and state.outcome is not None
            ):
                self._put_centered(
                    image,
                    prediction_result_message(
                        state.locked_user,
                        state.final_user,
                        state.outcome,
                    ),
                    (center[0], comparison_y),
                    0.58,
                    result_color,
                    2,
                )
                comparison_y += 42
            self._put_centered(
                image,
                f"PREDICTED {predicted}  |  FINAL {user}",
                (center[0], comparison_y),
                0.48,
                (220, 235, 235),
                1,
            )
            self._draw_gesture_card(
                image,
                center=(width // 2 - 180, height - 205),
                title="YOU",
                gesture=user,
                color=(80, 230, 110),
            )
            self._draw_gesture_card(
                image,
                center=(width // 2 + 180, height - 205),
                title="AI",
                gesture=ai,
                color=(90, 110, 250),
            )
            if state.score.user_streak >= 2:
                self._put_centered(
                    image,
                    f"{state.score.user_streak} WIN STREAK!",
                    (width // 2, 182),
                    0.50,
                    (50, 210, 250),
                    2,
                )

        self._update_effect(image, state)
        controls = "N network   M mute   R restart match   C clear session   Q quit"
        self._put_centered(image, controls, (width // 2, height - 20), 0.37, (165, 185, 195), 1)
        status = (
            f"{snapshot.device.upper()} {snapshot.inference_ms:.2f} ms   "
            f"{performance.fps:.0f} FPS"
        )
        self._put_text(image, status, (20, height - 18), 0.34, (135, 165, 175))

    def _draw_game(
        self, image: np.ndarray, state: GameViewState, snapshot: NetworkSnapshot
    ) -> None:
        height, width = image.shape[:2]
        center_x = max(650, width * 3 // 4)
        self._draw_scoreboard(image, state, compact=True)
        cv2.rectangle(image, (center_x - 250, 115), (width - 20, 235), (15, 22, 30), -1)
        self._put_text(image, state.phase.value, (center_x - 225, 145), 0.60, (80, 230, 220), 2)
        self._put_text(image, state.message, (center_x - 225, 180), 0.55, (245, 245, 245), 2)
        if state.countdown_label is not None:
            self._put_text(
                image,
                state.countdown_label,
                (center_x - 225, 220),
                0.85,
                (80, 255, 220),
                2,
            )
        if state.phase in {RoundPhase.RESULT, RoundPhase.MATCH_OVER}:
            user = state.final_user.name if state.final_user is not None else "?"
            ai = state.ai_move.name if state.ai_move is not None else "?"
            predicted = state.locked_user.name if state.locked_user is not None else "?"
            detail_y = 218
            detail_scale = 0.44
            if (
                state.phase == RoundPhase.MATCH_OVER
                and state.locked_user is not None
                and state.final_user is not None
                and state.outcome is not None
            ):
                self._put_text(
                    image,
                    prediction_result_message(
                        state.locked_user,
                        state.final_user,
                        state.outcome,
                    ),
                    (center_x - 225, 205),
                    0.40,
                    self._outcome_color(state),
                    1,
                )
                detail_y = 229
                detail_scale = 0.36
            self._put_text(
                image,
                f"PREDICTED {predicted}  |  FINAL {user}  |  AI {ai}",
                (center_x - 225, detail_y),
                detail_scale,
            )
        elif state.phase == RoundPhase.LOCKED:
            predicted = state.locked_user.name if state.locked_user is not None else "?"
            self._put_text(
                image,
                f"{predicted}  |  LOCKED IN {state.lock_time_ms or 0} MS  |  RESPONSE LOCKED",
                (center_x - 225, 218),
                0.40,
                (80, 240, 220),
                1,
            )
        elif state.ai_move is not None:
            self._put_text(image, "AI response locked", (center_x - 225, 218), 0.48)

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
        *,
        mode: RenderMode | str = RenderMode.GAME,
    ) -> np.ndarray:
        performance = performance or PerformanceStats()
        mode = RenderMode(mode)
        image = cv2.flip(frame_bgr, 1)
        height, width = image.shape[:2]
        self._track_phase_effects(state)
        if mode == RenderMode.GAME:
            self._draw_game_first(image, state, snapshot, performance)
        else:
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
        self._draw_lock_flash(image)
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
