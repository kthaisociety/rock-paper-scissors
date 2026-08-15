from __future__ import annotations

from rps.cli.demo import DEFAULT_RENDER_MODE, build_parser
from rps.constants import DEFAULT_SCORE_DB_PATH
from rps.renderer import RenderMode


def test_demo_audio_starts_enabled_and_supports_mute_flag() -> None:
    parser = build_parser()
    assert parser.parse_args([]).mute is False
    assert parser.parse_args(["--mute"]).mute is True


def test_demo_starts_in_combined_game_mode() -> None:
    assert DEFAULT_RENDER_MODE == RenderMode.GAME


def test_demo_uses_durable_default_score_database() -> None:
    parser = build_parser()
    assert parser.parse_args([]).score_db == DEFAULT_SCORE_DB_PATH
    assert parser.parse_args(["--score-db", "/tmp/booth.sqlite3"]).score_db.name == (
        "booth.sqlite3"
    )
