from __future__ import annotations

from rps.device import DeviceTiming, choose_device


def test_cpu_is_default_when_mps_is_unavailable() -> None:
    result = choose_device({"cpu": DeviceTiming("cpu", 0.2, 0.3)})
    assert result.selected == "cpu"


def test_mps_requires_ten_percent_improvement() -> None:
    fast = choose_device(
        {
            "cpu": DeviceTiming("cpu", 1.0, 1.2),
            "mps": DeviceTiming("mps", 0.8, 1.0),
        }
    )
    close = choose_device(
        {
            "cpu": DeviceTiming("cpu", 1.0, 1.2),
            "mps": DeviceTiming("mps", 0.95, 1.0),
        }
    )
    assert fast.selected == "mps"
    assert close.selected == "cpu"
