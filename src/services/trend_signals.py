"""Proste sygnaly z naglowkow RSS (kontuzje, sklady, form)."""

from __future__ import annotations

import re
from typing import List

import pandas as pd

# Rozszerzaj liste w miare potrzeb (PL + EN)
SIGNAL_PATTERNS = {
    "injury": re.compile(r"injur|kontuzj|hurt|withdraw|ruled out|nie zagra", re.I),
    "roster": re.compile(r"roster|sklad|lineup|bench|zawieszon", re.I),
    "form": re.compile(r"win streak|seria|upset|nokaut|dominacj", re.I),
    "hype": re.compile(r"favorite|faworyt|must.?win|final|playoff", re.I),
}


def annotate_news(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje kolumny sygnalow do tabeli z RSS (title + summary)."""
    if df.empty:
        return df
    out = df.copy()
    text = (out.get("title", "") + " " + out.get("summary", "")).astype(str)
    for label, pattern in SIGNAL_PATTERNS.items():
        out[f"signal_{label}"] = text.str.contains(pattern, regex=True)
    out["signal_any"] = out[[c for c in out.columns if c.startswith("signal_")]].any(axis=1)
    return out


def trending_teams(df: pd.DataFrame, min_mentions: int = 2) -> pd.DataFrame:
    """Heurystyka: wspomniane druzyny w tytulach (wymaga listy znanego aliasu – opcjonalnie)."""
    # Placeholder pod mapowanie team aliases z configu
    return pd.DataFrame(columns=["team", "mentions", "injury_mentions"])
