"""SQLite – zdarzenia z timestampami (point-in-time backtest)."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _db_path() -> Path:
    cfg_path = ROOT / "config" / "pipeline.yaml"
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            rel = yaml.safe_load(f).get("storage", {}).get("sqlite_path", "data/quantbet.db")
    else:
        rel = "data/quantbet.db"
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


SCHEMA = """
CREATE TABLE IF NOT EXISTS data_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    entity_type TEXT,
    entity_id TEXT,
    payload_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    valid_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_fetched ON data_events(fetched_at);
CREATE INDEX IF NOT EXISTS idx_events_source ON data_events(source, entity_id);

CREATE TABLE IF NOT EXISTS paper_bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT,
    home_team TEXT,
    away_team TEXT,
    side TEXT,
    odds REAL,
    stake REAL,
    p_model REAL,
    ev REAL,
    result TEXT,
    pnl REAL,
    placed_at TEXT NOT NULL,
    settled_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_bets_placed ON paper_bets(placed_at);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_type TEXT,
    metrics_json TEXT,
    weights_json TEXT,
    created_at TEXT NOT NULL
);
"""


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_event(
    source: str,
    payload: Dict[str, Any],
    entity_type: str = "",
    entity_id: str = "",
    fetched_at: Optional[datetime] = None,
    valid_at: Optional[datetime] = None,
) -> None:
    now = fetched_at or datetime.now(timezone.utc)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO data_events (source, entity_type, entity_id, payload_json, fetched_at, valid_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                entity_type,
                entity_id,
                json.dumps(payload, default=str),
                now.isoformat(),
                valid_at.isoformat() if valid_at else None,
            ),
        )


def events_before(as_of: datetime, source: Optional[str] = None) -> pd.DataFrame:
    with connect() as conn:
        q = "SELECT * FROM data_events WHERE fetched_at <= ?"
        params: List[Any] = [as_of.isoformat()]
        if source:
            q += " AND source = ?"
            params.append(source)
        q += " ORDER BY fetched_at"
        rows = conn.execute(q, params).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def insert_paper_bet(row: Dict[str, Any]) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO paper_bets
            (event_id, home_team, away_team, side, odds, stake, p_model, ev, result, pnl, placed_at, settled_at)
            VALUES (:event_id, :home_team, :away_team, :side, :odds, :stake, :p_model, :ev, :result, :pnl, :placed_at, :settled_at)
            """,
            row,
        )


def list_paper_bets(limit: int = 100) -> pd.DataFrame:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM paper_bets ORDER BY placed_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def save_model_run(run_type: str, metrics: Dict, weights: Dict) -> None:
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO model_runs (run_type, metrics_json, weights_json, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (run_type, json.dumps(metrics), json.dumps(weights), datetime.now(timezone.utc).isoformat()),
        )
