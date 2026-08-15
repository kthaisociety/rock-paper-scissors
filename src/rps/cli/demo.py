from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from rps.audio import AudioFeedback
from rps.checkpoint import CheckpointError, load_checkpoint
from rps.constants import DEFAULT_CHECKPOINT_PATH, DEFAULT_TEMPORAL_POLICY_PATH
from rps.device import resolve_device
from rps.features import InvalidLandmarksError, preprocess_landmarks
from rps.game import GameConfig, GameController, GameViewState, HandPrediction, RoundPhase
from rps.model import calibrated_probabilities
from rps.renderer import BoothRenderer, NetworkSnapshot, PerformanceStats, RenderMode
from rps.setup_assets import AssetError, ensure_hand_landmarker_asset
from rps.temporal import TemporalPolicyArtifactError, load_temporal_policy
from rps.tracking import AsyncHandTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the live student-fair booth demo")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--temporal-policy", type=Path, default=DEFAULT_TEMPORAL_POLICY_PATH)
    parser.add_argument(
        "--shadow-temporal-policy",
        type=Path,
        help="Run a matching candidate in parallel without letting it control gameplay",
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--mute", action="store_true", help="Start with sound cues muted")
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

    temporal_config = None
    if args.temporal_policy.exists():
        try:
            artifact = load_temporal_policy(
                args.temporal_policy,
                model_data_fingerprint=loaded.data_fingerprint,
                checkpoint_path=args.checkpoint,
            )
        except TemporalPolicyArtifactError as error:
            print(f"Temporal policy ignored; using baseline: {error}", file=sys.stderr)
        else:
            if artifact.status != "promoted":
                print(
                    "Temporal policy is not promoted; using baseline",
                    file=sys.stderr,
                )
            else:
                temporal_config = artifact.config
    else:
        print("Temporal policy not found; using baseline", file=sys.stderr)

    controller = GameController(GameConfig(temporal_policy=temporal_config))
    shadow_controller = None
    if args.shadow_temporal_policy is not None:
        try:
            shadow_artifact = load_temporal_policy(
                args.shadow_temporal_policy,
                model_data_fingerprint=loaded.data_fingerprint,
                checkpoint_path=args.checkpoint,
            )
        except TemporalPolicyArtifactError as error:
            raise SystemExit(f"Shadow temporal policy is invalid: {error}") from error
        shadow_controller = GameController(
            GameConfig(temporal_policy=shadow_artifact.config)
        )
    renderer = BoothRenderer(loaded.model, loaded.activation_scales)
    audio = AudioFeedback(muted=args.mute)
    render_mode = RenderMode.GAME
    snapshot = NetworkSnapshot(trained=loaded.trained, device=device_name)
    performance = PerformanceStats()
    last_tracker_timestamp = -1
    latest_prediction: HandPrediction | None = None
    latest_observation_timestamp = -1
    frame_started = time.perf_counter()
    shadow_round_logged = False

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
                                features=features,
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
                if loaded.trained and shadow_controller is not None:
                    shadow_state = shadow_controller.update(now_ms, latest_prediction)
                    if (
                        not shadow_round_logged
                        and state.lock_time_ms is not None
                        and shadow_state.lock_time_ms is not None
                    ):
                        print(
                            "Temporal shadow: "
                            f"active={state.locked_user.name}:{state.lock_time_ms}ms "
                            f"shadow={shadow_state.locked_user.name}:"
                            f"{shadow_state.lock_time_ms}ms"
                        )
                        shadow_round_logged = True
                    if state.phase == RoundPhase.READY:
                        shadow_round_logged = False
                audio.update(state)
                now = time.perf_counter()
                frame_duration = max(now - frame_started, 1e-6)
                current_fps = 1.0 / frame_duration
                performance.fps = (
                    current_fps
                    if performance.fps == 0.0
                    else 0.1 * current_fps + 0.9 * performance.fps
                )
                frame_started = now
                display = renderer.render(
                    frame,
                    state,
                    snapshot,
                    performance,
                    mode=render_mode,
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), ord("Q"), 27}:
                    break
                if key in {ord("n"), ord("N")}:
                    render_mode = (
                        RenderMode.NETWORK
                        if render_mode == RenderMode.GAME
                        else RenderMode.GAME
                    )
                elif key in {ord("m"), ord("M")}:
                    muted = audio.toggle_muted()
                    print(f"Audio {'muted' if muted else 'enabled'}")
                elif key in {ord("r"), ord("R")}:
                    controller.reset_match()
                    if shadow_controller is not None:
                        shadow_controller.reset_match()
                    shadow_round_logged = False
                elif key in {ord("c"), ord("C")}:
                    controller.reset_session()
                    if shadow_controller is not None:
                        shadow_controller.reset_session()
                    shadow_round_logged = False
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
