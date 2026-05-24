"""Ingest real-feed: PandaScore + HLTV + Odds API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

from collectors.hltv_scraper import HLTVScraperCollector
from collectors.odds_api import TheOddsAPICollector
from collectors.stats_pandascore import PandaScoreCollector
from pipeline.team_mapping import TeamNameNormalizer, canonical_pair


def _prepare_odds_table(odds_raw: pd.DataFrame, normalizer: TeamNameNormalizer) -> pd.DataFrame:
    if odds_raw.empty:
        return pd.DataFrame()
    odds = normalizer.normalize_frame(odds_raw, ["home_team", "away_team", "outcome_name"])
    odds["pair_key"] = odds.apply(lambda r: canonical_pair(str(r["home_team"]), str(r["away_team"])), axis=1)
    odds["event_time"] = pd.to_datetime(odds.get("commence_time"), utc=True, errors="coerce")
    rows = []
    for event_id, grp in odds.groupby("event_id"):
        home = str(grp["home_team"].iloc[0])
        away = str(grp["away_team"].iloc[0])
        home_lines = grp[grp["outcome_name"] == home]
        away_lines = grp[grp["outcome_name"] == away]
        if home_lines.empty or away_lines.empty:
            continue
        rows.append(
            {
                "event_id": str(event_id),
                "pair_key": canonical_pair(home, away),
                "date": grp["event_time"].iloc[0],
                "odds_home": float(home_lines["odds"].max()),
                "odds_away": float(away_lines["odds"].max()),
                "closing_odds_home": float(home_lines["odds"].median()),
                "closing_odds_away": float(away_lines["odds"].median()),
            }
        )
    return pd.DataFrame(rows)


def _prepare_pandascore_matches(df: pd.DataFrame, normalizer: TeamNameNormalizer) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    base = df.copy()
    base = base[pd.notna(base.get("target"))].copy()
    if base.empty:
        return pd.DataFrame()
    base["date"] = pd.to_datetime(base["begin_at"], utc=True, errors="coerce")
    base = normalizer.normalize_frame(base, ["team_a", "team_b"])
    base["match_id"] = base["match_id"].astype(str)
    base["target"] = pd.to_numeric(base["target"], errors="coerce").fillna(0).astype(int)
    base["team_a_kills"] = pd.to_numeric(base.get("team_a_kills", 0), errors="coerce").fillna(0).astype(int)
    base["team_b_kills"] = pd.to_numeric(base.get("team_b_kills", 0), errors="coerce").fillna(0).astype(int)
    base["pair_key"] = base.apply(lambda r: canonical_pair(str(r["team_a"]), str(r["team_b"])), axis=1)
    return base


def _inject_hltv_aliases(normalizer: TeamNameNormalizer, hltv_df: pd.DataFrame) -> None:
    if hltv_df.empty:
        return
    teams = pd.concat([hltv_df["team_a"], hltv_df["team_b"]]).dropna().astype(str).unique().tolist()
    for t in teams:
        key = " ".join(t.lower().split())
        normalizer.aliases.setdefault(key, t)


def load_real_pipeline_data(
    n_matches: int = 5000,
    fallback_to_synthetic: bool = True,
) -> pd.DataFrame:
    normalizer = TeamNameNormalizer()
    odds_res = TheOddsAPICollector().safe_fetch()
    panda_res = PandaScoreCollector().safe_fetch()
    hltv_res = HLTVScraperCollector().safe_fetch()
    _inject_hltv_aliases(normalizer, hltv_res.data if hltv_res.success else pd.DataFrame())

    odds = _prepare_odds_table(odds_res.data if odds_res.success else pd.DataFrame(), normalizer)
    matches = _prepare_pandascore_matches(panda_res.data if panda_res.success else pd.DataFrame(), normalizer)

    if not matches.empty:
        matches = matches.merge(
            odds.drop(columns=["date"], errors="ignore"),
            on="pair_key",
            how="left",
        )
    out = matches[
        [
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
            "team_mapping_unresolved",
        ]
    ].copy() if not matches.empty else pd.DataFrame()

    if not out.empty:
        for col in ("odds_home", "odds_away", "closing_odds_home", "closing_odds_away"):
            out[col] = pd.to_numeric(out[col], errors="coerce")
        # fallback dla brakow kursowych: implied fair + marza
        missing = out["odds_home"].isna() | out["odds_away"].isna()
        if missing.any():
            rng = np.random.default_rng(42)
            p = np.clip(rng.normal(0.5, 0.12, size=missing.sum()), 0.08, 0.92)
            margin = 1.06
            out.loc[missing, "odds_home"] = margin / p
            out.loc[missing, "odds_away"] = margin / (1 - p)
        out["closing_odds_home"] = out["closing_odds_home"].fillna(out["odds_home"])
        out["closing_odds_away"] = out["closing_odds_away"].fillna(out["odds_away"])
        out = out.dropna().sort_values("date").drop_duplicates("match_id").tail(n_matches).reset_index(drop=True)

    if len(out) >= min(n_matches, 200):
        return out

    if fallback_to_synthetic:
        from main import generate_full_pipeline_data

        needed = max(n_matches - len(out), 200 - len(out), 0)
        synth = generate_full_pipeline_data(max(needed, 0)) if needed > 0 else pd.DataFrame()
        if out.empty:
            return synth.head(n_matches).reset_index(drop=True)
        return pd.concat([out, synth], ignore_index=True).sort_values("date").tail(n_matches).reset_index(drop=True)
    return out


def load_pipeline_input_data(
    n_matches: int = 5000,
    source: str = "api",
    fallback_to_synthetic: bool = True,
) -> pd.DataFrame:
    src = (source or "api").lower()
    if src == "synthetic":
        from main import generate_full_pipeline_data

        return generate_full_pipeline_data(n_matches)
    if src == "api":
        return load_real_pipeline_data(n_matches=n_matches, fallback_to_synthetic=fallback_to_synthetic)
    raise ValueError(f"Nieznane zrodlo danych: {source}")


def ingestion_healthcheck() -> Dict[str, Optional[str]]:
    return {"checked_at": datetime.now(timezone.utc).isoformat(), "source": "api"}
