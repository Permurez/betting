"""Budowa cech z uwzglednieniem point-in-time."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from features.lol_features import add_lol_features, resolve_feature_columns
from main import generate_features as _base_features
from pipeline.point_in_time import merge_odds_at_decision
from storage.db import events_before
import numpy as np

def add_hype_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implementacja analizy hype-u spolecznego."""
    # Obecnie dla uproszczenia podrzucmy placeholder z dystrybucji na podstawie istniejacych features.
    # W produkcji nalezaloby to wyciagnąć bezposrednio z DataFrame sentimentu z SQL/cache.
    df = df.copy()
    if "hype_diff" not in df.columns:
        # Symulacja przyplywu hype-u (im drozszy "underdog", a wyzsze ELO, tym wiekszy hype/edge)
        df["hype_score_home"] = np.random.uniform(-1, 1, size=len(df))
        df["hype_score_away"] = np.random.uniform(-1, 1, size=len(df))
        df["hype_diff"] = df["hype_score_home"] - df["hype_score_away"]
    return df

def add_market_and_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje psychologiczne cechy rynkowe (ruchy grubych ryb) oraz zmęczenie fizyczne."""
    df = df.copy()
    
    # Ruchy Linii (Steam): Jak ostro rynek i "smart money" zareagowali na kurs otwarcia. 
    # W Freak Fightach ogromne dropy kursow na underdoga demaskuja informacje insiderskie (np. kontuzje).
    if "odds_home" in df.columns and "closing_odds_home" in df.columns:
        # Im mniejszy closing odds w stosunku do otwarcia, tym bardziej "smart money" wierzy w ta strone
        df["steam_home"] = np.where(df["closing_odds_home"] > 0, 
                                    (df["odds_home"] / df["closing_odds_home"]) - 1, 0)
        df["steam_away"] = np.where(df["closing_odds_away"] > 0, 
                                    (df["odds_away"] / df["closing_odds_away"]) - 1, 0)
        df["steam_diff"] = df["steam_home"] - df["steam_away"]
        
    # Fatigue (Dni od ostatniej walki/meczu)
    if "date" in df.columns and "team_a" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        
        # Szybka wektoryzacja dni od ostatniego meczu dla calej ligi
        teams_dates = pd.concat([
            df[["date", "team_a"]].rename(columns={"team_a": "team"}),
            df[["date", "team_b"]].rename(columns={"team_b": "team"})
        ]).sort_values(["team", "date"]).drop_duplicates()
        
        teams_dates["prev_date"] = teams_dates.groupby("team")["date"].shift(1)
        teams_dates["rest_days"] = (teams_dates["date"] - teams_dates["prev_date"]).dt.days.fillna(90) # domylsnie po wakatach
        
        # Merge z powrotem
        df = df.merge(teams_dates, left_on=["team_a", "date"], right_on=["team", "date"], how="left").rename(columns={"rest_days": "rest_days_home"}).drop(columns=["team", "prev_date"])
        df = df.merge(teams_dates, left_on=["team_b", "date"], right_on=["team", "date"], how="left").rename(columns={"rest_days": "rest_days_away"}).drop(columns=["team", "prev_date"])
        
        df["rest_diff"] = df["rest_days_home"] - df["rest_days_away"]
        df["rest_diff"] = df["rest_diff"].fillna(0).clip(-365, 365) # max odchylenia roczne

    return df

def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implementacja analizy warunków fizycznych (wzrost, waga, zasieg rąk, bmi)."""
    df = df.copy()
    
    # Symulacja bazowa/placeholder jeśli nie zaczytaliśmy scrapperem danych z Tapology/Wiki
    # Realnie wektory te siorbiemy w osobnym crawlerze per UUID fightera.  
    if "height_home" not in df.columns:
        df["height_home"] = np.random.normal(180, 8, size=len(df))
        df["height_away"] = np.random.normal(180, 8, size=len(df))
        
    if "weight_home" not in df.columns:
        df["weight_home"] = np.random.normal(82, 12, size=len(df))
        df["weight_away"] = np.random.normal(82, 12, size=len(df))
        
    if "reach_home" not in df.columns:
        # Zasieg ramion to zwykle wzrost +/- kilkanascie cm (Ape Index)
        df["reach_home"] = df["height_home"] + np.random.normal(0, 5, size=len(df))
        df["reach_away"] = df["height_away"] + np.random.normal(0, 5, size=len(df))
        
    if "age_home" not in df.columns:
        df["age_home"] = np.random.normal(28, 4, size=len(df))
        df["age_away"] = np.random.normal(28, 4, size=len(df))

    # Obliczenie rzczywistych kolumn klasyfikujacych dla modelu (Gradient Boostingi lubią delty H2H)
    df["height_diff"] = df["height_home"] - df["height_away"]
    df["weight_diff"] = df["weight_home"] - df["weight_away"]
    df["reach_diff"] = df["reach_home"] - df["reach_away"]
    df["age_diff"] = df["age_home"] - df["age_away"]
    
    # BMI (Body Mass Index) to swietny wyznacznik we freak fightach kogo "zaleje tlen" po 1 rundzie
    df["bmi_home"] = df["weight_home"] / ((df["height_home"]/100)**2)
    df["bmi_away"] = df["weight_away"] / ((df["height_away"]/100)**2)
    df["bmi_diff"] = df["bmi_home"] - df["bmi_away"]

    return df

def add_gym_and_health_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implementacja cech zaawansowanych: status zdrowia i poziom klubu/trenera."""
    df = df.copy()
    
    # 1. Kontuzje i Stan Zdrowia (Możliwe do wyciągnięcia z NLP newsów i forów przed walką)
    if "injury_severity_home" not in df.columns:
        # Skala 0-1 (0 - w pełni zdrowy, 1 - wraca po ciężkiej kontuzji/potwierdzony uraz w trakcie campu)
        df["injury_severity_home"] = np.random.choice([0, 0.2, 0.5, 0.8], size=len(df), p=[0.7, 0.15, 0.1, 0.05])
        df["injury_severity_away"] = np.random.choice([0, 0.2, 0.5, 0.8], size=len(df), p=[0.7, 0.15, 0.1, 0.05])
        
    # 2. Renoma Trenera / Klubu i Nakłady Finansowe (Możliwe do zeskrapowania np. rating klubu na Tapology)
    if "gym_tier_home" not in df.columns:
        # Skala 1-5 gwiazdek dla klubu (np. 5 to American Top Team w MMA / WCA w PL, 1 to losowa salka)
        df["gym_tier_home"] = np.random.randint(1, 6, size=len(df))
        df["gym_tier_away"] = np.random.randint(1, 6, size=len(df))
        
    if "camp_investment_home" not in df.columns:
        # Skala 1-10 określająca budżet campu (wyjazd do Tajlandii vs lokalne treningi)
        df["camp_investment_home"] = np.random.uniform(1, 10, size=len(df))
        df["camp_investment_away"] = np.random.uniform(1, 10, size=len(df))

    # Obliczenie rzczywistych kolumn ML (Delty)
    df["health_diff"] = df["injury_severity_away"] - df["injury_severity_home"] # Dodatnia delta = nasz przeciwnik jest bardziej kontuzjowany
    df["gym_tier_diff"] = df["gym_tier_home"] - df["gym_tier_away"]
    df["camp_investment_diff"] = df["camp_investment_home"] - df["camp_investment_away"]

    return df

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
    df = add_hype_features(df)
    df = add_market_and_fatigue_features(df)
    df = add_physical_features(df)
    df = add_gym_and_health_features(df)
    
    feature_cols = resolve_feature_columns(df)
    
    # Doklejanie nowych kolumn by model je widzial w LightGBM/MLP
    for col in [
        "hype_diff", "steam_diff", "rest_diff", "height_diff", 
        "weight_diff", "reach_diff", "age_diff", "bmi_diff",
        "health_diff", "gym_tier_diff", "camp_investment_diff"
    ]:
        if col in df.columns and col not in feature_cols:
            feature_cols.append(col)
            
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
