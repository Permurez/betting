"""Budowa cech z uwzglednieniem point-in-time."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from features.lol_features import add_lol_features, resolve_feature_columns
from main import generate_features as _base_features
from pipeline.point_in_time import merge_odds_at_decision
from storage.db import events_before


def build_features(
    raw_df: pd.DataFrame,
    patch_df: Optional[pd.DataFrame] = None,
    use_point_in_time_odds: bool = False,
    odds_snapshots: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, list[str]]:
    df = raw_df.copy()
    if use_point_in_time_odds and odds_snapshots is not None and not odds_snapshots.empty:
        df = merge_odds_at_decision(df, odds_snapshots)

    df = _base_features(df)
    df = add_lol_features(df, patch_dates=patch_df)
    feature_cols = resolve_feature_columns(df)
    return df, feature_cols


def load_patch_history() -> pd.DataFrame:
    from datetime import datetime, timezone

    ev = events_before(datetime.now(timezone.utc), source="datadragon")
    if ev.empty:
        return pd.DataFrame()
    rows = []
    for _, e in ev.iterrows():
        import json

        payload = json.loads(e["payload_json"])
        if "latest" in payload:
            rows.append({"patch_version": payload["latest"], "released_at": e["fetched_at"]})
    return pd.DataFrame(rows).drop_duplicates(subset=["patch_version"]) if rows else pd.DataFrame()
