"""Porownanie kursow miedzy bukmacherami – gdzie najlepiej postawic."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class OddsComparator:
    """Analiza linii H2H z The Odds API (lub innego znormalizowanego CSV)."""

    REQUIRED = {"event_id", "home_team", "away_team", "bookmaker", "outcome_name", "odds"}

    def __init__(self, odds_df: pd.DataFrame):
        self.df = odds_df.copy()
        missing = self.REQUIRED - set(self.df.columns)
        if missing:
            raise ValueError(f"Brak kolumn: {missing}")

    def best_odds_per_outcome(self) -> pd.DataFrame:
        """Dla kazdego meczu i outcome: najwyzszy kurs + bukmacher."""
        idx = self.df.groupby(["event_id", "outcome_name"])["odds"].idxmax()
        best = self.df.loc[idx].copy()
        best = best.rename(columns={"bookmaker": "best_bookmaker", "odds": "best_odds"})
        return best.sort_values(["commence_time", "home_team"], na_position="last")

    def event_summary(self) -> pd.DataFrame:
        """Dwa wiersze na mecz (home/away) z najlepszymi kursami i srednia rynku."""
        best = self.best_odds_per_outcome()
        rows: List[dict] = []

        for event_id, grp in self.df.groupby("event_id"):
            home = grp["home_team"].iloc[0]
            away = grp["away_team"].iloc[0]
            commence = grp["commence_time"].iloc[0] if "commence_time" in grp.columns else None

            def _side(team_name: str) -> Dict:
                sub = grp[grp["outcome_name"] == team_name]
                if sub.empty:
                    return {}
                best_row = sub.loc[sub["odds"].idxmax()]
                avg_odds = sub["odds"].mean()
                return {
                    "best_odds": float(best_row["odds"]),
                    "best_bookmaker": best_row["bookmaker"],
                    "avg_odds": float(avg_odds),
                    "edge_vs_avg_pct": float((best_row["odds"] / avg_odds - 1) * 100)
                    if avg_odds > 0
                    else 0.0,
                }

            h = _side(home)
            a = _side(away)
            rows.append(
                {
                    "event_id": event_id,
                    "commence_time": commence,
                    "home_team": home,
                    "away_team": away,
                    "best_home_odds": h.get("best_odds"),
                    "best_home_book": h.get("best_bookmaker"),
                    "home_edge_vs_avg_pct": h.get("edge_vs_avg_pct"),
                    "best_away_odds": a.get("best_odds"),
                    "best_away_book": a.get("best_bookmaker"),
                    "away_edge_vs_avg_pct": a.get("edge_vs_avg_pct"),
                }
            )

        return pd.DataFrame(rows)

    def find_arbitrage(self, min_profit_pct: float = 0.0) -> pd.DataFrame:
        """Prosty arbitraz 2-way: najlepszy kurs na kazda strone z roznych bukow."""
        arbs: List[dict] = []
        for event_id, grp in self.df.groupby("event_id"):
            home = grp["home_team"].iloc[0]
            away = grp["away_team"].iloc[0]
            home_lines = grp[grp["outcome_name"] == home]
            away_lines = grp[grp["outcome_name"] == away]
            if home_lines.empty or away_lines.empty:
                continue

            best_home = home_lines.loc[home_lines["odds"].idxmax()]
            best_away = away_lines.loc[away_lines["odds"].idxmax()]
            inv_sum = 1 / best_home["odds"] + 1 / best_away["odds"]
            if inv_sum < 1:
                profit_pct = (1 / inv_sum - 1) * 100
                if profit_pct >= min_profit_pct:
                    arbs.append(
                        {
                            "event_id": event_id,
                            "home_team": home,
                            "away_team": away,
                            "home_odds": best_home["odds"],
                            "home_book": best_home["bookmaker"],
                            "away_odds": best_away["odds"],
                            "away_book": best_away["bookmaker"],
                            "arb_profit_pct": profit_pct,
                            "implied_total": inv_sum,
                        }
                    )
        return pd.DataFrame(arbs).sort_values("arb_profit_pct", ascending=False)

    def recommendation_table(self, model_probs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Laczy najlepsze kursy z opcjonalnymi p_model (kolumny: event_id, p_home).
        EV = p * odds - 1 dla strony home.
        """
        summary = self.event_summary()
        if model_probs is not None and "p_home" in model_probs.columns:
            summary = summary.merge(
                model_probs[["event_id", "p_home"]], on="event_id", how="left"
            )
            summary["ev_home_best"] = summary["p_home"] * summary["best_home_odds"] - 1
            summary["ev_away_best"] = (1 - summary["p_home"]) * summary["best_away_odds"] - 1
            summary["best_side"] = np.where(
                summary["ev_home_best"] >= summary["ev_away_best"], "home", "away"
            )
            summary["best_ev"] = summary[["ev_home_best", "ev_away_best"]].max(axis=1)
            summary["recommended_book"] = np.where(
                summary["best_side"] == "home",
                summary["best_home_book"],
                summary["best_away_book"],
            )
        return summary
