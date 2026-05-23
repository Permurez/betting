"""Wczytywanie i eksport danych meczowych."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd

REQUIRED_COLUMNS = [
    "match_id",
    "date",
    "team_a",
    "team_b",
    "team_a_kills",
    "team_b_kills",
    "odds_home",
    "odds_away",
    "closing_odds_home",
    "closing_odds_away",
    "target",
]

PathLike = Union[str, Path]


def load_matches_csv(source: Union[PathLike, BinaryIO]) -> pd.DataFrame:
    """Wczytuje CSV z meczami; sortuje chronologicznie i waliduje kolumny."""
    df = pd.read_csv(source)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Brakujace kolumny w CSV: {missing}. Wymagane: {REQUIRED_COLUMNS}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["target"] = df["target"].astype(int)

    for col in ("odds_home", "odds_away", "closing_odds_home", "closing_odds_away"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("team_a_kills", "team_b_kills"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("CSV zawiera puste wartosci w wymaganych kolumnach.")

    return df.sort_values("date").reset_index(drop=True)


def export_bets_csv(df: pd.DataFrame) -> bytes:
    """Eksportuje zawarte zaklady do CSV (bytes pod st.download_button)."""
    cols = [
        "date",
        "team_a",
        "team_b",
        "p_model",
        "odds_home",
        "odds_away",
        "bet_side",
        "bet_ev",
        "stake_fraction",
        "return_factor",
        "closing_odds",
        "clv_pct",
    ]
    placed = df.loc[df["bet_side"] != "none", [c for c in cols if c in df.columns]]
    return placed.to_csv(index=False).encode("utf-8")


def template_csv_bytes() -> bytes:
    """Minimalny szablon CSV do pobrania z UI."""
    path = Path(__file__).resolve().parents[1] / "data" / "sample_matches.csv"
    return path.read_bytes()
