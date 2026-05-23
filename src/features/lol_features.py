"""Cechy LoL: patch, rank, forma – bez lookahead."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

BASE_FEATURE_COLUMNS = [
    "tA_winrate",
    "tB_winrate",
    "tA_kills",
    "tB_kills",
    "form_diff",
    "kills_diff",
]

LOL_EXTRA_COLUMNS = [
    "days_since_patch",
    "rank_diff",
    "games_a_7d",
    "games_b_7d",
]


def rank_to_numeric(tier: str, rank: str = "IV", lp: int = 0) -> float:
    tiers = {
        "IRON": 0,
        "BRONZE": 400,
        "SILVER": 800,
        "GOLD": 1200,
        "PLATINUM": 1600,
        "EMERALD": 2000,
        "DIAMOND": 2400,
        "MASTER": 2800,
        "GRANDMASTER": 3000,
        "CHALLENGER": 3200,
    }
    div = {"IV": 0, "III": 100, "II": 200, "I": 300}.get(str(rank).upper(), 0)
    return float(tiers.get(str(tier).upper(), 1000) + div + lp * 0.1)


def add_lol_features(df: pd.DataFrame, patch_dates: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Dodaje cechy LoL do ramki meczowej.
    patch_dates: columns [patch_version, released_at]
    Opcjonalne kolumny wejsciowe per druzyna/gracz: team_a_rank, team_b_rank, team_a_games_7d, ...
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], utc=True)

    if patch_dates is not None and not patch_dates.empty:
        patches = patch_dates.copy()
        patches["released_at"] = pd.to_datetime(patches["released_at"], utc=True)
        patches = patches.sort_values("released_at")

        def days_since(row_date):
            prior = patches[patches["released_at"] <= row_date]
            if prior.empty:
                return 30.0
            last = prior.iloc[-1]["released_at"]
            return max((row_date - last).total_seconds() / 86400.0, 0.0)

        out["days_since_patch"] = out["date"].apply(days_since)
    else:
        out["days_since_patch"] = 14.0

    if "team_a_rank" in out.columns and "team_b_rank" in out.columns:
        out["rank_diff"] = out["team_a_rank"].astype(float) - out["team_b_rank"].astype(float)
    else:
        out["rank_diff"] = 0.0

    out["games_a_7d"] = out.get("team_a_games_7d", pd.Series(0, index=out.index)).fillna(0)
    out["games_b_7d"] = out.get("team_b_games_7d", pd.Series(0, index=out.index)).fillna(0)

    return out


def resolve_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = list(BASE_FEATURE_COLUMNS)
    for c in LOL_EXTRA_COLUMNS:
        if c in df.columns:
            cols.append(c)
    return cols
