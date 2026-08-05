from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from rps.features import PREPROCESS_VERSION
from rps.model import CLASS_NAMES, GestureMLP, default_activation_scales

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    pass


@dataclass(slots=True)
class LoadedModel:
    model: GestureMLP
    device: torch.device
    trained: bool
    temperature: float = 1.0
    activation_scales: dict[str, list[float]] = field(default_factory=default_activation_scales)
    metrics: dict[str, Any] = field(default_factory=dict)
    data_fingerprint: str = "untrained"


def create_untrained_model(device: torch.device | str = "cpu", seed: int = 42) -> LoadedModel:
    torch.manual_seed(seed)
    resolved_device = torch.device(device)
    model = GestureMLP().to(resolved_device).eval()
    return LoadedModel(model=model, device=resolved_device, trained=False)


def save_checkpoint(
    path: Path,
    model: GestureMLP,
    *,
    temperature: float,
    activation_scales: dict[str, list[float]],
    data_fingerprint: str,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "class_names": list(CLASS_NAMES),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "temperature": float(temperature),
        "activation_scales": activation_scales,
        "data_fingerprint": data_fingerprint,
        "metrics": metrics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
    *,
    allow_untrained: bool = True,
) -> LoadedModel:
    resolved_device = torch.device(device)
    if not path.exists():
        if allow_untrained:
            return create_untrained_model(resolved_device)
        raise CheckpointError(f"Checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # pragma: no cover - PyTorch errors vary by version
        raise CheckpointError(f"Could not load checkpoint {path}: {error}") from error

    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError("Unsupported checkpoint format version")
    if payload.get("preprocess_version") != PREPROCESS_VERSION:
        raise CheckpointError("Checkpoint preprocessing version does not match the application")
    if tuple(payload.get("class_names", ())) != CLASS_NAMES:
        raise CheckpointError("Checkpoint class order does not match ROCK, PAPER, SCISSORS")

    model = GestureMLP()
    try:
        model.load_state_dict(payload["state_dict"], strict=True)
    except (KeyError, RuntimeError) as error:
        raise CheckpointError(f"Checkpoint model parameters are invalid: {error}") from error
    model.to(resolved_device).eval()
    scales = payload.get("activation_scales", default_activation_scales())
    if len(scales.get("act1", ())) != 16 or len(scales.get("act2", ())) != 8:
        raise CheckpointError("Checkpoint activation calibration has invalid dimensions")
    return LoadedModel(
        model=model,
        device=resolved_device,
        trained=True,
        temperature=float(payload.get("temperature", 1.0)),
        activation_scales=scales,
        metrics=payload.get("metrics", {}),
        data_fingerprint=str(payload.get("data_fingerprint", "unknown")),
    )
