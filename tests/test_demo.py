from __future__ import annotations

from rps.cli.demo import DEFAULT_RENDER_MODE, build_parser
from rps.renderer import RenderMode


def test_demo_audio_starts_enabled_and_supports_mute_flag() -> None:
    parser = build_parser()
    assert parser.parse_args([]).mute is False
    assert parser.parse_args(["--mute"]).mute is True


def test_demo_starts_in_combined_game_mode() -> None:
    assert DEFAULT_RENDER_MODE == RenderMode.GAME
