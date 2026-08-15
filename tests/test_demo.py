from __future__ import annotations

from rps.cli.demo import build_parser


def test_demo_audio_starts_enabled_and_supports_mute_flag() -> None:
    parser = build_parser()
    assert parser.parse_args([]).mute is False
    assert parser.parse_args(["--mute"]).mute is True
