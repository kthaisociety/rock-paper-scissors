from __future__ import annotations

import argparse

from rps.setup_assets import AssetError, ensure_hand_landmarker_asset, file_sha256


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Download and verify the MediaPipe model asset")


def main() -> None:
    build_parser().parse_args()
    try:
        path = ensure_hand_landmarker_asset()
    except AssetError as error:
        raise SystemExit(str(error)) from error
    print(f"Hand Landmarker ready: {path}")
    print(f"SHA-256: {file_sha256(path)}")


if __name__ == "__main__":
    main()
