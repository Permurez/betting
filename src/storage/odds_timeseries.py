"""Storage kursow czasowych: PostgreSQL + TimescaleDB (z fallbackiem)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd

from config_loader import env, load_pipeline_config
from storage.db import events_before

try:
    import psycopg
except Exception:  # pragma: no cover - opcjonalny driver
    psycopg = None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS odds_snapshots (
    id BIGSERIAL PRIMARY KEY,
    event_id TEXT NOT NULL,
    commence_time TIMESTAMPTZ,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    bookmaker_key TEXT,
    bookmaker TEXT,
    outcome_name TEXT NOT NULL,
    odds DOUBLE PRECISION NOT NULL,
    fetched_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_odds_event_time ON odds_snapshots(event_id, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_odds_match_time ON odds_snapshots(home_team, away_team, fetched_at DESC);
"""


def _cfg() -> dict:
    return load_pipeline_config().get("storage", {})


class OddsTimeseriesStore:
    def __init__(self, dsn: Optional[str] = None):
        cfg = _cfg()
        self.backend = str(cfg.get("backend", "sqlite")).lower()
        self.dsn = dsn or env("DATABASE_URL") or cfg.get("postgres_dsn", "")
        self._ready = False

    @property
    def use_timescaledb(self) -> bool:
        return self.backend in {"postgres", "timescaledb"} and bool(self.dsn) and psycopg is not None

    def _connect(self):
        if not self.use_timescaledb:
            return None
        return psycopg.connect(self.dsn, autocommit=True)

    def _ensure_schema(self) -> None:
        if self._ready or not self.use_timescaledb:
            return
        conn = self._connect()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
                cur.execute(
                    """
                    SELECT create_hypertable(
                        'odds_snapshots',
                        by_range('fetched_at'),
                        if_not_exists => TRUE
                    )
                    """
                )
        finally:
            conn.close()
        self._ready = True

    def insert_snapshots(self, odds_df: pd.DataFrame) -> None:
        if odds_df.empty:
            return
        if not self.use_timescaledb:
            return
        self._ensure_schema()
        rows = odds_df.copy()
        rows["fetched_at"] = pd.to_datetime(rows.get("fetched_at"), utc=True, errors="coerce")
        rows["commence_time"] = pd.to_datetime(rows.get("commence_time"), utc=True, errors="coerce")
        rows = rows.dropna(subset=["event_id", "home_team", "away_team", "outcome_name", "odds", "fetched_at"])
        if rows.empty:
            return
        payload = [
            (
                str(r["event_id"]),
                r["commence_time"].to_pydatetime() if not pd.isna(r["commence_time"]) else None,
                str(r["home_team"]),
                str(r["away_team"]),
                str(r.get("bookmaker_key", "")),
                str(r.get("bookmaker", "")),
                str(r["outcome_name"]),
                float(r["odds"]),
                r["fetched_at"].to_pydatetime(),
            )
            for _, r in rows.iterrows()
        ]
        conn = self._connect()
        if conn is None:
            return
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO odds_snapshots
                    (event_id, commence_time, home_team, away_team, bookmaker_key, bookmaker, outcome_name, odds, fetched_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    payload,
                )
        finally:
            conn.close()

    def snapshots_before(self, as_of: datetime, event_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
        if self.use_timescaledb:
            self._ensure_schema()
            conn = self._connect()
            if conn is None:
                return pd.DataFrame()
            try:
                with conn.cursor() as cur:
                    if event_ids:
                        ids = list({str(x) for x in event_ids if x})
                        cur.execute(
                            """
                            SELECT event_id, commence_time, home_team, away_team, bookmaker_key, bookmaker,
                                   outcome_name, odds, fetched_at
                            FROM odds_snapshots
                            WHERE fetched_at <= %s AND event_id = ANY(%s)
                            ORDER BY fetched_at
                            """,
                            (as_of, ids),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT event_id, commence_time, home_team, away_team, bookmaker_key, bookmaker,
                                   outcome_name, odds, fetched_at
                            FROM odds_snapshots
                            WHERE fetched_at <= %s
                            ORDER BY fetched_at
                            """,
                            (as_of,),
                        )
                    rows = cur.fetchall()
            finally:
                conn.close()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame(
                rows,
                columns=[
                    "event_id",
                    "commence_time",
                    "home_team",
                    "away_team",
                    "bookmaker_key",
                    "bookmaker",
                    "outcome_name",
                    "odds",
                    "fetched_at",
                ],
            )

        ev = events_before(as_of, source="odds_api")
        if ev.empty:
            return pd.DataFrame()
        parsed: List[dict] = []
        for _, row in ev.iterrows():
            payload = json.loads(row["payload_json"])
            payload["fetched_at"] = row.get("fetched_at")
            parsed.append(payload)
        out = pd.DataFrame(parsed)
        if event_ids:
            out = out[out["event_id"].astype(str).isin({str(x) for x in event_ids if x})]
        return out.reset_index(drop=True)
