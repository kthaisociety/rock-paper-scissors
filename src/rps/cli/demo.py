from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from rps.checkpoint import CheckpointError, load_checkpoint
from rps.constants import DEFAULT_CHECKPOINT_PATH
from rps.device import resolve_device
from rps.features import InvalidLandmarksError, preprocess_landmarks
from rps.game import GameController, GameViewState, HandPrediction, RoundPhase
from rps.model import calibrated_probabilities
from rps.renderer import BoothRenderer, NetworkSnapshot, PerformanceStats
from rps.setup_assets import AssetError, ensure_hand_landmarker_asset
from rps.tracking import AsyncHandTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the live student-fair booth demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--fullscreen", action="store_true")
    return parser


def _untrained_state() -> GameViewState:
    return GameViewState(
        phase=RoundPhase.READY,
        message="Train a checkpoint to enable gameplay",
    )


def main() -> None:
    args = build_parser().parse_args()
    try:
        asset_path = ensure_hand_landmarker_asset()
        cpu_loaded = load_checkpoint(args.checkpoint, "cpu", allow_untrained=True)
        device_name = resolve_device(args.device, cpu_loaded.model)
        loaded = load_checkpoint(args.checkpoint, device_name, allow_untrained=True)
    except (AssetError, CheckpointError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    camera = cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    window_name = "Mid-Gesture RPS AI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    controller = GameController()
    renderer = BoothRenderer(loaded.model, loaded.activation_scales)
    snapshot = NetworkSnapshot(trained=loaded.trained, device=device_name)
    performance = PerformanceStats()
    last_tracker_timestamp = -1
    latest_prediction: HandPrediction | None = None
    latest_observation_timestamp = -1
    frame_started = time.perf_counter()

    try:
        with AsyncHandTracker(asset_path) as tracker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames")
                now_ms = int(time.monotonic_ns() / 1_000_000)
                tracker.submit(frame, now_ms)
                tracker_result = tracker.latest(after_timestamp_ms=last_tracker_timestamp)
                if tracker_result is not None:
                    last_tracker_timestamp = tracker_result.timestamp_ms
                    observation = tracker_result.observation
                    if observation is None:
                        latest_prediction = None
                        snapshot = NetworkSnapshot(
                            trained=loaded.trained,
                            device=device_name,
                            result_age_ms=max(0.0, now_ms - tracker_result.timestamp_ms),
                        )
                    else:
                        try:
                            features = preprocess_landmarks(
                                observation.landmarks, observation.handedness
                            )
                        except InvalidLandmarksError:
                            latest_prediction = None
                        else:
                            inference_started = time.perf_counter_ns()
                            tensor = torch.as_tensor(
                                features.reshape(1, 63),
                                dtype=torch.float32,
                                device=loaded.device,
                            )
                            with torch.inference_mode():
                                output = loaded.model(tensor)
                                probabilities = calibrated_probabilities(
                                    output.logits, loaded.temperature
                                )
                                probabilities_np = probabilities[0].cpu().numpy()
                                act1 = output.act1[0].cpu().numpy()
                                act2 = output.act2[0].cpu().numpy()
                            if loaded.device.type == "mps":
                                torch.mps.synchronize()
                            inference_ms = (
                                time.perf_counter_ns() - inference_started
                            ) / 1_000_000.0
                            latest_prediction = HandPrediction(
                                timestamp_ms=observation.timestamp_ms,
                                probabilities=probabilities_np,
                                centered=(
                                    0.2 <= float(np.mean(observation.landmarks[:, 0])) <= 0.8
                                    and 0.15 <= float(np.mean(observation.landmarks[:, 1])) <= 0.85
                                ),
                            )
                            latest_observation_timestamp = observation.timestamp_ms
                            snapshot = NetworkSnapshot(
                                features=features,
                                act1=act1,
                                act2=act2,
                                probabilities=probabilities_np,
                                hand_landmarks=observation.landmarks,
                                trained=loaded.trained,
                                device=device_name,
                                inference_ms=inference_ms,
                                result_age_ms=max(0.0, now_ms - observation.timestamp_ms),
                            )

                if now_ms - latest_observation_timestamp > 200:
                    latest_prediction = None
                    snapshot.hand_landmarks = None
                    snapshot.features *= 0.85
                    snapshot.act1 *= 0.85
                    snapshot.act2 *= 0.85
                    snapshot.probabilities *= 0.85

                state = (
                    controller.update(now_ms, latest_prediction)
                    if loaded.trained
                    else _untrained_state()
                )
                now = time.perf_counter()
                frame_duration = max(now - frame_started, 1e-6)
                current_fps = 1.0 / frame_duration
                performance.fps = (
                    current_fps
                    if performance.fps == 0.0
                    else 0.1 * current_fps + 0.9 * performance.fps
                )
                frame_started = now
                display = renderer.render(frame, state, snapshot, performance)
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
