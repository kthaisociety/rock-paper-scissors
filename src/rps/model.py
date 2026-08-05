from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from torch import nn

CLASS_NAMES = ("ROCK", "PAPER", "SCISSORS")


class ModelOutput(NamedTuple):
    logits: torch.Tensor
    act1: torch.Tensor
    act2: torch.Tensor


class GestureMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(63, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)

    def forward(self, features: torch.Tensor) -> ModelOutput:
        act1 = torch.relu(self.fc1(features))
        act2 = torch.relu(self.fc2(act1))
        logits = self.fc3(act2)
        return ModelOutput(logits=logits, act1=act1, act2=act2)


def calibrated_probabilities(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    safe_temperature = max(float(temperature), 1e-4)
    return torch.softmax(logits / safe_temperature, dim=-1)


@torch.inference_mode()
def calculate_activation_scales(
    model: GestureMLP, features: np.ndarray, device: torch.device | str
) -> dict[str, list[float]]:
    if len(features) == 0:
        return default_activation_scales()
    tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
    output = model(tensor)
    act1 = output.act1.detach().cpu().numpy()
    act2 = output.act2.detach().cpu().numpy()
    scales1 = np.maximum(np.percentile(np.abs(act1), 99, axis=0), 1e-3)
    scales2 = np.maximum(np.percentile(np.abs(act2), 99, axis=0), 1e-3)
    return {"act1": scales1.tolist(), "act2": scales2.tolist()}


def default_activation_scales() -> dict[str, list[float]]:
    return {"act1": [1.0] * 16, "act2": [1.0] * 8}
