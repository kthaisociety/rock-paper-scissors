from __future__ import annotations

import pytest

from rps.game import GameConfig, GameController, ScoreSnapshot
from rps.score_store import ScoreStoreError, SQLiteScoreStore


def populated_score() -> ScoreSnapshot:
    return ScoreSnapshot(
        user_points=2,
        ai_points=1,
        ties=4,
        user_round_wins=12,
        ai_round_wins=9,
        user_streak=2,
        ai_streak=0,
        best_user_streak=5,
        user_matches=3,
        ai_matches=2,
        rounds_played=25,
        target_points=3,
    )


def test_score_store_round_trips_and_overwrites_atomic_snapshot(tmp_path) -> None:
    path = tmp_path / ".rps" / "scores.sqlite3"
    with SQLiteScoreStore(path) as store:
        assert store.load() == ScoreSnapshot()
        store.save(populated_score())
        updated = ScoreSnapshot(user_points=1, ai_points=2, rounds_played=3)
        store.save(updated)

    with SQLiteScoreStore(path) as reopened:
        assert reopened.load() == updated


def test_controller_restores_current_match_and_session_totals() -> None:
    controller = GameController(score=populated_score())

    restored = controller.view(0)

    assert restored.score == populated_score()
    assert restored.phase.value == "READY"


def test_restart_after_persisted_match_win_starts_new_match() -> None:
    completed = ScoreSnapshot(
        user_points=3,
        ai_points=1,
        user_round_wins=8,
        ai_round_wins=4,
        user_streak=3,
        best_user_streak=4,
        user_matches=2,
        ai_matches=1,
        rounds_played=13,
    )

    restored = GameController(GameConfig(target_points=3), score=completed).view(0).score

    assert restored.user_points == restored.ai_points == 0
    assert restored.user_streak == restored.ai_streak == 0
    assert restored.user_matches == 2
    assert restored.user_round_wins == 8
    assert restored.rounds_played == 13


def test_invalid_persisted_score_is_rejected(tmp_path) -> None:
    path = tmp_path / "scores.sqlite3"
    with SQLiteScoreStore(path) as store:
        with pytest.raises(ScoreStoreError, match="negative"):
            store.save(ScoreSnapshot(ties=-1))

        store.connection.execute("PRAGMA ignore_check_constraints = ON")
        store.connection.execute(
            """
            INSERT INTO score_state (
                singleton, user_points, ai_points, ties, user_round_wins,
                ai_round_wins, user_streak, ai_streak, best_user_streak,
                user_matches, ai_matches, rounds_played, target_points
            ) VALUES (1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 3)
            """
        )
        store.connection.commit()
        with pytest.raises(ScoreStoreError, match="negative"):
            store.load()


def test_closed_store_reports_a_clear_error(tmp_path) -> None:
    store = SQLiteScoreStore(tmp_path / "scores.sqlite3")
    store.close()

    with pytest.raises(ScoreStoreError, match="closed"):
        store.load()


def test_database_uses_full_synchronous_durability(tmp_path) -> None:
    with SQLiteScoreStore(tmp_path / "scores.sqlite3") as store:
        synchronous = store.connection.execute("PRAGMA synchronous").fetchone()[0]
        journal_mode = store.connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert synchronous == 2
    assert journal_mode == "wal"
