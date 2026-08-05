from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rps.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from rps.model import GestureMLP, calibrated_probabilities, default_activation_scales


def test_model_shapes_and_probabilities() -> None:
    model = GestureMLP()
    output = model(torch.zeros((2, 63)))
    assert output.logits.shape == (2, 3)
    assert output.act1.shape == (2, 16)
    assert output.act2.shape == (2, 8)
    probabilities = calibrated_probabilities(output.logits, temperature=1.2)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    model = GestureMLP()
    save_checkpoint(
        path,
        model,
        temperature=1.25,
        activation_scales=default_activation_scales(),
        data_fingerprint="abc123",
        metrics={"accuracy": 0.9},
    )
    loaded = load_checkpoint(path, "cpu", allow_untrained=False)
    assert loaded.trained
    assert loaded.temperature == pytest.approx(1.25)
    assert loaded.data_fingerprint == "abc123"
    for expected, actual in zip(model.parameters(), loaded.model.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)


def test_missing_checkpoint_can_be_untrained(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    assert not load_checkpoint(missing, "cpu", allow_untrained=True).trained
    with pytest.raises(CheckpointError):
        load_checkpoint(missing, "cpu", allow_untrained=False)


def test_incompatible_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 999}, path)
    with pytest.raises(CheckpointError):
        load_checkpoint(path, "cpu", allow_untrained=False)
