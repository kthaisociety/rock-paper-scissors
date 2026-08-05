from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


@dataclass(frozen=True, slots=True)
class HandObservation:
    timestamp_ms: int
    landmarks: np.ndarray
    handedness: str
    handedness_score: float
    callback_monotonic_ms: float


@dataclass(frozen=True, slots=True)
class TrackerResult:
    timestamp_ms: int
    observation: HandObservation | None
    callback_monotonic_ms: float


class AsyncHandTracker:
    """Latest-only wrapper around MediaPipe's asynchronous Hand Landmarker."""

    def __init__(
        self,
        model_path: Path,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Hand Landmarker model not found: {model_path}")
        self._lock = threading.Lock()
        self._latest: TrackerResult | None = None
        self._last_submitted_timestamp = -1

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            result_callback=self._on_result,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _on_result(self, result: object, _image: mp.Image, timestamp_ms: int) -> None:
        callback_ms = time.monotonic_ns() / 1_000_000.0
        observation: HandObservation | None = None
        hand_landmarks = getattr(result, "hand_landmarks", ())
        handedness_results = getattr(result, "handedness", ())
        if hand_landmarks:
            landmarks = np.asarray(
                [[point.x, point.y, point.z] for point in hand_landmarks[0]], dtype=np.float32
            )
            handedness = "Unknown"
            handedness_score = 0.0
            if handedness_results and handedness_results[0]:
                category = handedness_results[0][0]
                handedness = str(category.category_name or category.display_name or "Unknown")
                handedness_score = float(category.score or 0.0)
            observation = HandObservation(
                timestamp_ms=timestamp_ms,
                landmarks=landmarks,
                handedness=handedness,
                handedness_score=handedness_score,
                callback_monotonic_ms=callback_ms,
            )
        tracker_result = TrackerResult(timestamp_ms, observation, callback_ms)
        with self._lock:
            if self._latest is None or timestamp_ms >= self._latest.timestamp_ms:
                self._latest = tracker_result

    def submit(self, frame_bgr: np.ndarray, timestamp_ms: int) -> None:
        timestamp_ms = max(int(timestamp_ms), self._last_submitted_timestamp + 1)
        self._last_submitted_timestamp = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._landmarker.detect_async(image, timestamp_ms)

    def latest(self, *, after_timestamp_ms: int = -1) -> TrackerResult | None:
        with self._lock:
            latest = self._latest
        if latest is None or latest.timestamp_ms <= after_timestamp_ms:
            return None
        return latest

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> AsyncHandTracker:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()
