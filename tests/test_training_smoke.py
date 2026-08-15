from __future__ import annotations

import torch
from torch import nn

from rps.cli.train import build_parser
from rps.cli.tune_temporal import build_parser as build_temporal_parser
from rps.model import GestureMLP


def test_training_defaults_use_validated_balanced_refit_recipe() -> None:
    args = build_parser().parse_args([])

    assert args.epochs == 160
    assert args.patience == 40
    assert args.min_epochs == 60
    assert args.balanced_sampling
    assert args.scheduler == "cosine"
    assert args.refit_train_validation


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


def test_temporal_tuning_defaults_preserve_validated_refit_epoch_count() -> None:
    args = build_temporal_parser().parse_args(["tune"])
    assert args.fold_epochs == 32
    assert args.seed == 42
