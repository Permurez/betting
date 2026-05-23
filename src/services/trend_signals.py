"""Proste sygnaly z naglowkow RSS (kontuzje, sklady, form)."""

from __future__ import annotations

import re
from typing import List

import pandas as pd
from textblob import TextBlob

# Rozszerzaj liste w miare potrzeb (PL + EN)
SIGNAL_PATTERNS = {
    "injury": re.compile(r"injur|kontuzj|hurt|withdraw|ruled out|nie zagra", re.I),
    "roster": re.compile(r"roster|sklad|lineup|bench|zawieszon", re.I),
    "form": re.compile(r"win streak|seria|upset|nokaut|dominacj", re.I),
    "hype": re.compile(r"favorite|faworyt|must.?win|final|playoff|zmiażdży|zniszczy|hype", re.I),
}

def analyze_sentiment(text: str) -> float:
    """Zwraca wynik sentymentu od -1.0 (negatywny) do 1.0 (pozytywny)."""
    if not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity

def calculate_hype_score(df: pd.DataFrame, text_col: str = "caption") -> pd.DataFrame:
    """Tworzy Hype Score na podstawie sentymentu wpisow/wiadomosci."""
    if df.empty or text_col not in df.columns:
        return df
    
    out = df.copy()
    out["sentiment"] = out[text_col].apply(analyze_sentiment)
    # Hype Score np. polaczenie lajkow i sentimentu w przypaku Instagrama:
    if "likes" in out.columns:
        out["hype_score"] = out["sentiment"] * out["likes"].apply(lambda x: min(float(x)/1000.0, 10.0))
    else:
        out["hype_score"] = out["sentiment"] * 5.0 # Skalowanie
    return out

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
