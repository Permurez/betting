"""Paper trading – symulowane stawki w SQLite."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config_loader import execution_mode, load_pipeline_config
from execution.broker import BetOrder, Broker, LiveBroker
from storage.db import insert_paper_bet, list_paper_bets


class PaperTrader:
    def __init__(
        self,
        bankroll: float = 10000.0,
        kelly_fraction: float = 0.1,
        min_ev: float = 0.03,
        max_stake_pct: float = 0.15,
    ):
        cfg = load_pipeline_config().get("execution", {})
        self.bankroll = bankroll or cfg.get("initial_bankroll", 10000)
        self.kelly = kelly_fraction or cfg.get("kelly_fraction", 0.1)
        self.min_ev = min_ev or cfg.get("min_ev", 0.03)
        self.max_stake_pct = max_stake_pct or cfg.get("max_stake_pct", 0.15)
        self.mode = execution_mode()

    def _broker(self) -> Broker:
        if self.mode.upper() == "LIVE":
            return LiveBroker()
        return None

    def evaluate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dodaje kolumny bet_side, stake, bet_ev jak backtester."""
        out = df.copy()
        if "p_model" not in out.columns:
            raise ValueError("Brak kolumny p_model.")

        out["ev_home"] = out["p_model"] * out["odds_home"] - 1
        out["ev_away"] = (1 - out["p_model"]) * out["odds_away"] - 1
        cond_h = (out["ev_home"] > out["ev_away"]) & (out["ev_home"] >= self.min_ev)
        cond_a = (out["ev_away"] > out["ev_home"]) & (out["ev_away"] >= self.min_ev)
        out["bet_side"] = np.select([cond_h, cond_a], ["home", "away"], default="none")
        out["bet_ev"] = np.select([cond_h, cond_a], [out["ev_home"], out["ev_away"]], default=0.0)
        out["bet_odds"] = np.select([cond_h, cond_a], [out["odds_home"], out["odds_away"]], default=1.0)
        raw_k = np.where(out["bet_odds"] > 1, out["bet_ev"] / (out["bet_odds"] - 1), 0.0)
        out["stake_fraction"] = np.clip(
            np.where(out["bet_side"] != "none", raw_k * self.kelly, 0.0),
            0.0,
            self.max_stake_pct,
        )
        out["stake_amount"] = out["stake_fraction"] * self.bankroll
        return out

    def place_from_signals(self, df: pd.DataFrame) -> List[int]:
        """Zapisuje paper bety dla wierszy z bet_side != none."""
        signals = self.evaluate_signals(df)
        placed = signals[signals["bet_side"] != "none"]
        ids: List[int] = []
        now = datetime.now(timezone.utc).isoformat()

        for _, row in placed.iterrows():
            insert_paper_bet(
                {
                    "event_id": str(row.get("match_id", row.get("event_id", ""))),
                    "home_team": str(row.get("team_a", row.get("home_team", ""))),
                    "away_team": str(row.get("team_b", row.get("away_team", ""))),
                    "side": row["bet_side"],
                    "odds": float(row["bet_odds"]),
                    "stake": float(row["stake_amount"]),
                    "p_model": float(row["p_model"]),
                    "ev": float(row["bet_ev"]),
                    "result": "open",
                    "pnl": 0.0,
                    "placed_at": now,
                    "settled_at": None,
                }
            )
            ids.append(len(ids) + 1)

        return ids

    def settle_open_bets(self, results_df: pd.DataFrame) -> int:
        """Rozlicza otwarte bety po znanym wyniku (target: 1=home win)."""
        open_bets = list_paper_bets(500)
        if open_bets.empty:
            return 0
        open_bets = open_bets[open_bets["result"] == "open"]
        settled = 0
        # Uproszczenie: dopasowanie po event_id / team names w przyszlosci przez DB update
        return settled

    def summary(self) -> Dict[str, float]:
        bets = list_paper_bets(1000)
        if bets.empty:
            return {"open": 0, "total_pnl": 0.0}
        closed = bets[bets["result"] != "open"]
        return {
            "open": int((bets["result"] == "open").sum()),
            "closed": len(closed),
            "total_pnl": float(closed["pnl"].sum()) if len(closed) else 0.0,
        }
