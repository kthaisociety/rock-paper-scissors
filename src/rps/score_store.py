from __future__ import annotations

import sqlite3
from pathlib import Path

from rps.game import ScoreSnapshot


class ScoreStoreError(RuntimeError):
    """Raised when durable booth scores cannot be read or written."""


_SCORE_FIELDS = (
    "user_points",
    "ai_points",
    "ties",
    "user_round_wins",
    "ai_round_wins",
    "user_streak",
    "ai_streak",
    "best_user_streak",
    "user_matches",
    "ai_matches",
    "rounds_played",
    "target_points",
)


class SQLiteScoreStore:
    """Durable single-booth score state stored as one atomic SQLite row."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        connection: sqlite3.Connection | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=5.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS score_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    user_points INTEGER NOT NULL CHECK (user_points >= 0),
                    ai_points INTEGER NOT NULL CHECK (ai_points >= 0),
                    ties INTEGER NOT NULL CHECK (ties >= 0),
                    user_round_wins INTEGER NOT NULL CHECK (user_round_wins >= 0),
                    ai_round_wins INTEGER NOT NULL CHECK (ai_round_wins >= 0),
                    user_streak INTEGER NOT NULL CHECK (user_streak >= 0),
                    ai_streak INTEGER NOT NULL CHECK (ai_streak >= 0),
                    best_user_streak INTEGER NOT NULL CHECK (best_user_streak >= 0),
                    user_matches INTEGER NOT NULL CHECK (user_matches >= 0),
                    ai_matches INTEGER NOT NULL CHECK (ai_matches >= 0),
                    rounds_played INTEGER NOT NULL CHECK (rounds_played >= 0),
                    target_points INTEGER NOT NULL CHECK (target_points > 0),
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute("PRAGMA user_version = 1")
            connection.commit()
            self._connection = connection
        except (OSError, sqlite3.Error) as error:
            if connection is not None:
                connection.close()
            raise ScoreStoreError(
                f"Could not initialize score database {self.path}: {error}"
            ) from error

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise ScoreStoreError("Score database is closed")
        return self._connection

    def load(self, *, target_points: int = 3) -> ScoreSnapshot:
        try:
            row = self.connection.execute(
                f"SELECT {', '.join(_SCORE_FIELDS)} FROM score_state WHERE singleton = 1"
            ).fetchone()
        except sqlite3.Error as error:
            raise ScoreStoreError(f"Could not load scores from {self.path}: {error}") from error
        if row is None:
            return ScoreSnapshot(target_points=target_points)
        values = {field: int(row[field]) for field in _SCORE_FIELDS}
        snapshot = ScoreSnapshot(**values)
        self._validate(snapshot)
        return snapshot

    def save(self, score: ScoreSnapshot) -> None:
        self._validate(score)
        placeholders = ", ".join("?" for _ in _SCORE_FIELDS)
        updates = ", ".join(f"{field} = excluded.{field}" for field in _SCORE_FIELDS)
        values = tuple(getattr(score, field) for field in _SCORE_FIELDS)
        try:
            with self.connection:
                self.connection.execute(
                    f"""
                    INSERT INTO score_state (singleton, {', '.join(_SCORE_FIELDS)})
                    VALUES (1, {placeholders})
                    ON CONFLICT(singleton) DO UPDATE SET
                        {updates},
                        updated_at = CURRENT_TIMESTAMP
                    """,
                    values,
                )
        except sqlite3.Error as error:
            raise ScoreStoreError(f"Could not save scores to {self.path}: {error}") from error

    @staticmethod
    def _validate(score: ScoreSnapshot) -> None:
        for field in _SCORE_FIELDS[:-1]:
            if getattr(score, field) < 0:
                raise ScoreStoreError(f"Invalid negative score field: {field}")
        if score.target_points <= 0:
            raise ScoreStoreError("target_points must be positive")
        if score.user_points > score.target_points or score.ai_points > score.target_points:
            raise ScoreStoreError("Current match score exceeds target_points")
        if score.user_points == score.target_points and score.ai_points == score.target_points:
            raise ScoreStoreError("Both players cannot have won the current match")

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> SQLiteScoreStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
