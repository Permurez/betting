"""Walk-forward backtest – wiele okien czasowych, bez wycieku."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from main import TARGET_COLUMN, VectorizedBacktester, train_model
from models.ensemble import EnsembleModel
from pipeline.feature_builder import build_features, load_patch_history
from storage.db import save_model_run


def walk_forward_splits(n: int, n_folds: int, min_train: int) -> List[Tuple[slice, slice]]:
    """Rosnace okno treningu + kolejny blok testowy."""
    fold_size = max((n - min_train) // n_folds, 1)
    splits = []
    for i in range(n_folds):
        test_start = min_train + i * fold_size
        test_end = min(test_start + fold_size, n)
        if test_start >= n or test_end <= test_start:
            break
        splits.append((slice(0, test_start), slice(test_start, test_end)))
    return splits


def run_walk_forward(
    raw_df: pd.DataFrame,
    n_folds: int = 5,
    min_train_rows: int = 500,
    use_ensemble: bool = True,
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.1,
    min_ev: float = 0.03,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Dla kazdego foldu: trening na przeszlosci, predykcja na nastepnym odcinku.
    Zwraca sklejony DataFrame z kolumna fold + metryki zbiorcze.
    """
    patch_df = load_patch_history()
    processed, feature_cols = build_features(
        raw_df.sort_values("date").reset_index(drop=True),
        patch_df=patch_df if not patch_df.empty else None,
    )
    n = len(processed)
    splits = walk_forward_splits(n, n_folds, min_train_rows)
    if not splits:
        raise ValueError(f"Za malo danych ({n}) dla walk-forward (min_train={min_train_rows}).")

    all_parts: List[pd.DataFrame] = []
    fold_metrics: List[Dict] = []
    last_weights = {}

    for fold_i, (train_sl, test_sl) in enumerate(splits):
        train_full = processed.iloc[train_sl]
        test_part = processed.iloc[test_sl].copy()
        if len(train_full) < 100 or len(test_part) < 10:
            continue

        val_size = max(int(len(train_full) * 0.15), 50)
        train_df = train_full.iloc[:-val_size]
        val_df = train_full.iloc[-val_size:]

        if use_ensemble:
            model = train_model(feature_cols, train_df, val_df, force_ensemble=True)
            last_weights = model.weights_dict() if isinstance(model, EnsembleModel) else {}
            preds = model.predict_proba(test_part[feature_cols])
        else:
            model = train_model(feature_cols, train_df, val_df, force_ensemble=False)
            preds = model.predict_proba(test_part[feature_cols])

        test_part["p_model"] = preds
        test_part["fold"] = fold_i

        try:
            auc = roc_auc_score(test_part[TARGET_COLUMN], preds)
            ll = log_loss(test_part[TARGET_COLUMN], np.clip(preds, 1e-6, 1 - 1e-6))
        except ValueError:
            auc, ll = 0.5, 999.0

        fold_metrics.append({"fold": fold_i, "roc_auc": auc, "log_loss": ll, "test_rows": len(test_part)})
        all_parts.append(test_part)

    combined = pd.concat(all_parts, ignore_index=True)
    backtester = VectorizedBacktester(initial_bankroll, kelly_fraction, min_ev)
    results_df, bt_metrics = backtester.run(combined)

    summary = {
        "folds": len(fold_metrics),
        "mean_roc_auc": float(np.mean([f["roc_auc"] for f in fold_metrics])),
        "mean_log_loss": float(np.mean([f["log_loss"] for f in fold_metrics])),
        **bt_metrics,
        "ensemble_weights": last_weights,
    }
    save_model_run("walk_forward", summary, last_weights)
    return results_df, summary
