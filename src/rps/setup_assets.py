from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from rps.constants import (
    HAND_LANDMARKER_PATH,
    HAND_LANDMARKER_SHA256,
    HAND_LANDMARKER_URL,
)


class AssetError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_is_valid(path: Path = HAND_LANDMARKER_PATH) -> bool:
    return path.exists() and file_sha256(path) == HAND_LANDMARKER_SHA256


def ensure_hand_landmarker_asset(
    path: Path = HAND_LANDMARKER_PATH,
    *,
    download: bool = True,
) -> Path:
    if asset_is_valid(path):
        return path
    if path.exists():
        path.unlink()
    if not download:
        raise AssetError(f"Hand Landmarker asset is missing or invalid: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".task", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        request = urllib.request.Request(
            HAND_LANDMARKER_URL, headers={"User-Agent": "rps-booth/0.1"}
        )
        with (
            urllib.request.urlopen(request, timeout=60) as response,  # noqa: S310
            temporary_path.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        actual = file_sha256(temporary_path)
        if actual != HAND_LANDMARKER_SHA256:
            raise AssetError(
                "Hand Landmarker checksum mismatch: "
                f"expected {HAND_LANDMARKER_SHA256}, got {actual}"
            )
        os.replace(temporary_path, path)
    except (OSError, urllib.error.URLError) as error:
        raise AssetError(f"Could not download Hand Landmarker asset: {error}") from error
    finally:
        temporary_path.unlink(missing_ok=True)
    return path
