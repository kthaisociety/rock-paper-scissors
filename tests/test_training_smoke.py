from __future__ import annotations

import torch
from torch import nn

from rps.model import GestureMLP


def test_model_can_complete_a_training_step() -> None:
    torch.manual_seed(42)
    model = GestureMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    features = torch.randn(18, 63)
    labels = torch.arange(18) % 3
    criterion = nn.CrossEntropyLoss()
    initial = float(criterion(model(features).logits, labels).detach())
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features).logits, labels)
        loss.backward()
        optimizer.step()
    final = float(criterion(model(features).logits, labels).detach())
    assert final < initial
