"""Point-in-time: tylko dane dostepne przed momentem decyzji."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from storage.odds_timeseries import OddsTimeseriesStore

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
    odds_snapshots: Optional[pd.DataFrame] = None,
    decision_col: str = "date",
    latency_minutes: int = 0,
) -> pd.DataFrame:
    """
    Dla kazdego meczu bierze ostatni snapshot kursow PRZED decision_col.
    odds_snapshots: event_id, fetched_at, odds_home, odds_away, ...
    """
    if matches.empty:
        return matches

    if odds_snapshots is None or odds_snapshots.empty:
        as_of_max = pd.to_datetime(matches[decision_col], utc=True, errors="coerce").max()
        if pd.isna(as_of_max):
            return matches
        event_ids = [str(x) for x in matches.get("match_id", pd.Series(dtype=str)).dropna().tolist()]
        odds_snapshots = OddsTimeseriesStore().snapshots_before(as_of_max.to_pydatetime(), event_ids=event_ids)
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
        if latency_minutes > 0:
            snap = apply_latency_cutoff(snap, decision_time=t.to_pydatetime(), latency_minutes=latency_minutes)
        row = m.to_dict()
        if not snap.empty:
            last = snap.sort_values("fetched_at").iloc[-1]
            if "outcome_name" in snap.columns and "home_team" in snap.columns:
                home_team = row.get("team_a", row.get("home_team", ""))
                away_team = row.get("team_b", row.get("away_team", ""))
                home_line = snap[snap["outcome_name"] == home_team].sort_values("fetched_at").tail(1)
                away_line = snap[snap["outcome_name"] == away_team].sort_values("fetched_at").tail(1)
                if not home_line.empty and not away_line.empty:
                    row["odds_home"] = float(home_line["odds"].iloc[-1])
                    row["odds_away"] = float(away_line["odds"].iloc[-1])
                    row["closing_odds_home"] = row.get("closing_odds_home", row["odds_home"])
                    row["closing_odds_away"] = row.get("closing_odds_away", row["odds_away"])
            else:
                for c in ("odds_home", "odds_away", "closing_odds_home", "closing_odds_away"):
                    if c in last.index:
                        row[c] = last[c]
        out_rows.append(row)

    return pd.DataFrame(out_rows)
