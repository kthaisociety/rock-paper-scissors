from __future__ import annotations

import argparse
import platform
import sys

import cv2

from rps.checkpoint import CheckpointError, load_checkpoint
from rps.constants import DEFAULT_CHECKPOINT_PATH
from rps.device import benchmark_devices, save_device_benchmark
from rps.setup_assets import AssetError, ensure_hand_landmarker_asset, file_sha256
from rps.tracking import AsyncHandTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify booth assets, model, camera, and device")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use a shortened device benchmark")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    failures: list[str] = []
    print(f"Python: {platform.python_version()} ({platform.machine()})")
    if sys.version_info[:2] != (3, 12):
        failures.append("The booth environment must use Python 3.12")

    try:
        asset = ensure_hand_landmarker_asset()
        print(f"Hand Landmarker: OK ({file_sha256(asset)[:12]}...)")
        with AsyncHandTracker(asset):
            print("MediaPipe initialization: OK")
    except (AssetError, FileNotFoundError, RuntimeError) as error:
        failures.append(str(error))

    try:
        loaded = load_checkpoint(
            DEFAULT_CHECKPOINT_PATH, "cpu", allow_untrained=args.allow_untrained
        )
        if loaded.trained:
            print(f"Checkpoint: OK ({loaded.data_fingerprint[:12]}...)")
        else:
            print("Checkpoint: UNTRAINED visualization mode allowed")
        benchmark = benchmark_devices(
            loaded.model,
            warmup=10 if args.quick else 100,
            iterations=50 if args.quick else 500,
        )
        save_device_benchmark(benchmark)
        for name, timing in benchmark.timings.items():
            print(f"Inference {name}: median={timing.median_ms:.3f} ms p95={timing.p95_ms:.3f} ms")
        print(f"Selected inference device: {benchmark.selected} ({benchmark.reason})")
    except (CheckpointError, RuntimeError) as error:
        failures.append(str(error))

    if not args.skip_camera:
        camera = cv2.VideoCapture(args.camera)
        try:
            ok, frame = camera.read()
            if not camera.isOpened() or not ok or frame is None:
                failures.append(f"Camera {args.camera} did not return a frame")
            else:
                print(f"Camera: OK ({frame.shape[1]}x{frame.shape[0]})")
        finally:
            camera.release()

    if failures:
        print("\nPreflight failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nPreflight passed")


if __name__ == "__main__":
    main()
