"""Point-in-time: tylko dane dostepne przed momentem decyzji."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd


def apply_latency_cutoff(
    events_df: pd.DataFrame,
    decision_time: datetime,
    latency_minutes: int = 0,
) -> pd.DataFrame:
    """
    Filtruje zdarzenia: fetched_at <= decision_time - latency.
    Symuluje opoznienie publikacji / przetworzenia zrodla.
    """
    if events_df.empty or "fetched_at" not in events_df.columns:
        return events_df
    cutoff = decision_time - timedelta(minutes=latency_minutes)
    ts = pd.to_datetime(events_df["fetched_at"], utc=True)
    return events_df.loc[ts <= cutoff].copy()


def merge_odds_at_decision(
    matches: pd.DataFrame,
    odds_snapshots: pd.DataFrame,
    decision_col: str = "date",
) -> pd.DataFrame:
    """
    Dla kazdego meczu bierze ostatni snapshot kursow PRZED decision_col.
    odds_snapshots: event_id, fetched_at, odds_home, odds_away, ...
    """
    if odds_snapshots.empty:
        return matches

    out_rows = []
    odds_snapshots = odds_snapshots.copy()
    odds_snapshots["fetched_at"] = pd.to_datetime(odds_snapshots["fetched_at"], utc=True)

    for _, m in matches.iterrows():
        t = pd.to_datetime(m[decision_col], utc=True)
        ev_id = m.get("match_id", m.get("event_id", ""))
        snap = odds_snapshots[
            (odds_snapshots["event_id"] == ev_id) & (odds_snapshots["fetched_at"] <= t)
        ]
        row = m.to_dict()
        if not snap.empty:
            last = snap.sort_values("fetched_at").iloc[-1]
            for c in ("odds_home", "odds_away", "closing_odds_home", "closing_odds_away"):
                if c in last.index:
                    row[c] = last[c]
        out_rows.append(row)

    return pd.DataFrame(out_rows)
