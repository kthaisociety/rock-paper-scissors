from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rps.constants import DEVICE_CONFIG_PATH
from rps.model import GestureMLP, calibrated_probabilities


@dataclass(frozen=True, slots=True)
class DeviceTiming:
    device: str
    median_ms: float
    p95_ms: float


@dataclass(frozen=True, slots=True)
class DeviceBenchmark:
    selected: str
    timings: dict[str, DeviceTiming]
    reason: str


def mps_available() -> bool:
    return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())


def choose_device(timings: dict[str, DeviceTiming]) -> DeviceBenchmark:
    cpu = timings["cpu"]
    mps = timings.get("mps")
    if mps is not None and mps.median_ms <= cpu.median_ms * 0.9:
        return DeviceBenchmark("mps", timings, "MPS median latency is at least 10% lower")
    reason = "CPU selected because MPS is unavailable"
    if mps is not None:
        reason = "CPU selected because MPS was not at least 10% faster"
    return DeviceBenchmark("cpu", timings, reason)


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def benchmark_inference_device(
    state_dict: dict[str, torch.Tensor],
    device_name: str,
    *,
    warmup: int = 100,
    iterations: int = 500,
) -> DeviceTiming:
    device = torch.device(device_name)
    model = GestureMLP()
    model.load_state_dict(state_dict)
    model.to(device).eval()
    sample = np.linspace(-1.0, 1.0, 63, dtype=np.float32).reshape(1, 63)

    def run_once() -> None:
        tensor = torch.as_tensor(sample, dtype=torch.float32, device=device)
        with torch.inference_mode():
            output = model(tensor)
            probabilities = calibrated_probabilities(output.logits)
            _ = probabilities.cpu().numpy()
            _ = output.act1.cpu().numpy()
            _ = output.act2.cpu().numpy()
        _synchronize(device)

    for _ in range(warmup):
        run_once()

    durations: list[float] = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter_ns()
        run_once()
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return DeviceTiming(
        device=device_name,
        median_ms=float(statistics.median(durations)),
        p95_ms=_percentile(durations, 95),
    )


def benchmark_devices(
    model: GestureMLP, *, warmup: int = 100, iterations: int = 500
) -> DeviceBenchmark:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    timings = {
        "cpu": benchmark_inference_device(state_dict, "cpu", warmup=warmup, iterations=iterations)
    }
    if mps_available():
        timings["mps"] = benchmark_inference_device(
            state_dict, "mps", warmup=warmup, iterations=iterations
        )
    return choose_device(timings)


def benchmark_training_device(
    state_dict: dict[str, torch.Tensor],
    device_name: str,
    *,
    warmup: int = 5,
    iterations: int = 50,
    batch_size: int = 256,
) -> DeviceTiming:
    device = torch.device(device_name)
    model = GestureMLP()
    model.load_state_dict(state_dict)
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    cpu_features = torch.linspace(-1.0, 1.0, batch_size * 63).reshape(batch_size, 63)
    cpu_labels = torch.arange(batch_size, dtype=torch.long) % 3

    def run_once() -> None:
        features = cpu_features.to(device)
        labels = cpu_labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features).logits, labels)
        loss.backward()
        optimizer.step()
        _synchronize(device)

    for _ in range(warmup):
        run_once()

    durations: list[float] = []
    for _ in range(iterations):
        _synchronize(device)
        started = time.perf_counter_ns()
        run_once()
        durations.append((time.perf_counter_ns() - started) / 1_000_000.0)
    return DeviceTiming(
        device=device_name,
        median_ms=float(statistics.median(durations)),
        p95_ms=_percentile(durations, 95),
    )


def benchmark_training_devices(
    model: GestureMLP, *, warmup: int = 5, iterations: int = 50
) -> DeviceBenchmark:
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    timings = {
        "cpu": benchmark_training_device(state_dict, "cpu", warmup=warmup, iterations=iterations)
    }
    if mps_available():
        timings["mps"] = benchmark_training_device(
            state_dict, "mps", warmup=warmup, iterations=iterations
        )
    return choose_device(timings)


def save_device_benchmark(benchmark: DeviceBenchmark, path: Path = DEVICE_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected": benchmark.selected,
        "reason": benchmark.reason,
        "timings": {name: asdict(timing) for name, timing in benchmark.timings.items()},
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_cached_device(path: Path = DEVICE_CONFIG_PATH) -> str | None:
    if not path.exists():
        return None
    try:
        selected = json.loads(path.read_text(encoding="utf-8"))["selected"]
    except (KeyError, OSError, json.JSONDecodeError):
        return None
    if selected == "mps" and not mps_available():
        return None
    return selected if selected in {"cpu", "mps"} else None


def resolve_device(requested: str, model: GestureMLP | None = None) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "mps":
        if not mps_available():
            raise RuntimeError("MPS was requested but is not available")
        return "mps"
    cached = load_cached_device()
    if cached is not None:
        return cached
    if model is None:
        return "cpu"
    result = benchmark_devices(model)
    save_device_benchmark(result)
    return result.selected
