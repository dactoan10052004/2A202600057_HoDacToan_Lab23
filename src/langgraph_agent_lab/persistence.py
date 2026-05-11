"""Checkpointer factory.

Supports memory (default, no infra), sqlite (file-backed, WAL mode), and postgres.
SQLite requires: pip install langgraph-checkpoint-sqlite
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpointer(
    kind: str = "memory", database_url: str | None = None
) -> BaseCheckpointSaver | None:
    """Return a LangGraph checkpointer for the given backend.

    - "none"    → no persistence (graph resets between runs)
    - "memory"  → in-process MemorySaver (default, no files needed)
    - "sqlite"  → file-backed SqliteSaver with WAL mode for concurrent reads
    - "postgres"→ PostgresSaver (requires DATABASE_URL)
    """
    if kind == "none":
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "SQLite checkpointer requires: pip install langgraph-checkpoint-sqlite"
            ) from exc
        db_path = database_url or "outputs/checkpoints.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Use WAL mode so concurrent readers don't block the writer.
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires: pip install langgraph-checkpoint-postgres"
            ) from exc
        return PostgresSaver.from_conn_string(database_url or "")

    raise ValueError(f"Unknown checkpointer kind: {kind!r}. Choose: none, memory, sqlite, postgres")
