"""Metryki modelu i backtestu."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def evaluate_classifier(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """ROC-AUC, log-loss i Brier na zbiorze testowym."""
    y_true = np.asarray(y_true)
    y_proba = np.clip(np.asarray(y_proba), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
    }


def max_drawdown_pct(bankroll: pd.Series) -> float:
    """Maksymalny drawdown w % od szczytu kapitalu."""
    if bankroll.empty:
        return 0.0
    peak = bankroll.cummax()
    dd = (bankroll - peak) / peak.replace(0, np.nan)
    return float(dd.min() * 100)


def sharpe_ratio(returns: pd.Series) -> float:
    """Uproszczony Sharpe na serii strategy_return (per zaklad)."""
    active = returns[returns != 0]
    if len(active) < 2 or active.std() == 0:
        return 0.0
    return float((active.mean() / active.std()) * np.sqrt(len(active)))


def enrich_backtest_metrics(
    df: pd.DataFrame,
    placed_mask: pd.Series,
    initial_bankroll: float,
) -> Dict[str, float]:
    """Uzupelnia metryki backtestu o drawdown, Sharpe i srednie EV."""
    placed = df.loc[placed_mask]
    extra: Dict[str, float] = {
        "max_drawdown_pct": max_drawdown_pct(df["bankroll"]) if "bankroll" in df.columns else 0.0,
        "sharpe_ratio": sharpe_ratio(df["strategy_return"]) if "strategy_return" in df.columns else 0.0,
        "avg_bet_ev_pct": float(placed["bet_ev"].mean() * 100) if len(placed) > 0 else 0.0,
    }
    if len(placed) > 0 and "closing_odds" in placed.columns:
        extra["avg_clv"] = float(((placed["bet_odds"] / placed["closing_odds"]) - 1).mean() * 100)
    return extra
