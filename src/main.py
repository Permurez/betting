import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from config_loader import load_pipeline_config, use_ensemble
from evaluation import enrich_backtest_metrics, evaluate_classifier
from features.lol_features import BASE_FEATURE_COLUMNS, resolve_feature_columns
from models.ensemble import EnsembleModel
from models.lgbm_model import QuantModel, MODELS_DIR

warnings.filterwarnings("ignore")

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
TARGET_COLUMN = "target"


def generate_full_pipeline_data(n_matches: int = 5000) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=n_matches, freq="4h")
    teams = [f"Team_{i}" for i in range(1, 21)]

    team_strength = {team: np.random.uniform(0.3, 0.7) for team in teams}
    data = []

    for i, date in enumerate(dates):
        team_a, team_b = np.random.choice(teams, 2, replace=False)
        p_true = team_strength[team_a] / (team_strength[team_a] + team_strength[team_b])
        result_a = 1 if np.random.uniform() < p_true else 0

        kills_a = int(np.random.normal(50 + 40 * team_strength[team_a], 10))
        kills_b = int(np.random.normal(50 + 40 * team_strength[team_b], 10))

        public_bias = 0.05 if p_true > 0.55 else (-0.05 if p_true < 0.45 else 0.0)
        p_market = np.clip(p_true + public_bias + np.random.normal(0, 0.02), 0.01, 0.99)

        margin = 1.06
        odds_home = margin / p_market
        odds_away = margin / (1 - p_market)

        p_closing = np.clip(p_market * 0.5 + p_true * 0.5, 0.01, 0.99)

        data.append(
            {
                "match_id": f"M_{i}",
                "date": date,
                "team_a": team_a,
                "team_b": team_b,
                "team_a_kills": kills_a,
                "team_b_kills": kills_b,
                "odds_home": odds_home,
                "odds_away": odds_away,
                "closing_odds_home": margin / p_closing,
                "closing_odds_away": margin / (1 - p_closing),
                "target": result_a,
            }
        )

        team_strength[team_a] += np.random.normal(0, 0.005)
        team_strength[team_b] += np.random.normal(0, 0.005)

    return pd.DataFrame(data)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    df["team_a_win"] = df["target"]
    df["team_b_win"] = 1 - df["target"]

    hist_a = df[
        ["match_id", "date", "team_a", "team_a_win", "team_a_kills"]
    ].rename(
        columns={"team_a": "team", "team_a_win": "is_win", "team_a_kills": "kills"}
    )
    hist_b = df[
        ["match_id", "date", "team_b", "team_b_win", "team_b_kills"]
    ].rename(
        columns={"team_b": "team", "team_b_win": "is_win", "team_b_kills": "kills"}
    )

    history = pd.concat([hist_a, hist_b]).sort_values(["team", "date"]).reset_index(
        drop=True
    )
    history["past_win"] = history.groupby("team")["is_win"].shift(1)
    history["past_kills"] = history.groupby("team")["kills"].shift(1)

    history["ewm_winrate"] = history.groupby("team")["past_win"].transform(
        lambda x: x.ewm(span=10, min_periods=1).mean()
    )
    history["rolling_kills"] = history.groupby("team")["past_kills"].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean()
    )

    features = history[["match_id", "team", "ewm_winrate", "rolling_kills"]]

    df = (
        df.merge(
            features,
            left_on=["match_id", "team_a"],
            right_on=["match_id", "team"],
            how="left",
        )
        .rename(columns={"ewm_winrate": "tA_winrate", "rolling_kills": "tA_kills"})
        .drop(columns=["team"])
    )
    df = (
        df.merge(
            features,
            left_on=["match_id", "team_b"],
            right_on=["match_id", "team"],
            how="left",
        )
        .rename(columns={"ewm_winrate": "tB_winrate", "rolling_kills": "tB_kills"})
        .drop(columns=["team"])
    )

    df[["tA_winrate", "tB_winrate"]] = df[["tA_winrate", "tB_winrate"]].fillna(0.5)
    df[["tA_kills", "tB_kills"]] = df[["tA_kills", "tB_kills"]].fillna(50)

    df["form_diff"] = df["tA_winrate"] - df["tB_winrate"]
    df["kills_diff"] = df["tA_kills"] - df["tB_kills"]

    return df.drop(columns=["team_a_win", "team_b_win"])


def train_model(
    feature_cols: List[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    force_ensemble: Optional[bool] = None,
) -> Union[QuantModel, EnsembleModel]:
    use_ens = use_ensemble() if force_ensemble is None else force_ensemble
    if use_ens:
        model = EnsembleModel(feature_cols)
    else:
        model = QuantModel(feature_cols)
    model.train(
        train_df[feature_cols],
        train_df[TARGET_COLUMN],
        val_df[feature_cols],
        val_df[TARGET_COLUMN],
    )
    return model


def save_trained_model(model: Union[QuantModel, EnsembleModel], path: Optional[Path] = None) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = path or MODELS_DIR / "quant_model.joblib"
    payload: Dict[str, Any] = {"features": model.features, "type": "ensemble" if isinstance(model, EnsembleModel) else "lgbm"}
    if isinstance(model, EnsembleModel):
        payload["weights"] = model.weights_dict()
        model.lgbm.save(MODELS_DIR / "_ens_lgbm.joblib")
        joblib.dump(payload, out)
    else:
        model.save(out)
    return Path(out)


class VectorizedBacktester:
    def __init__(self, initial_bankroll: float = 1000.0, kelly_fraction: float = 0.1, min_ev: float = 0.02):
        self.bankroll = initial_bankroll
        self.kelly = kelly_fraction
        self.min_ev = min_ev

    def run(self, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
        df = test_df.copy()
        df["ev_home"] = df["p_model"] * df["odds_home"] - 1
        df["ev_away"] = (1 - df["p_model"]) * df["odds_away"] - 1

        cond_home = (df["ev_home"] > df["ev_away"]) & (df["ev_home"] >= self.min_ev)
        cond_away = (df["ev_away"] > df["ev_home"]) & (df["ev_away"] >= self.min_ev)

        df["bet_side"] = np.select([cond_home, cond_away], ["home", "away"], default="none")
        df["bet_ev"] = np.select([cond_home, cond_away], [df["ev_home"], df["ev_away"]], default=0.0)
        df["bet_odds"] = np.select([cond_home, cond_away], [df["odds_home"], df["odds_away"]], default=1.0)
        df["closing_odds"] = np.select(
            [cond_home, cond_away],
            [df["closing_odds_home"], df["closing_odds_away"]],
            default=1.0,
        )

        df["raw_kelly"] = np.where(df["bet_odds"] > 1, df["bet_ev"] / (df["bet_odds"] - 1), 0.0)
        df["stake_fraction"] = np.clip(
            np.where(df["bet_side"] != "none", df["raw_kelly"] * self.kelly, 0.0),
            0.0,
            0.15,
        )

        ret_home = np.where(df["target"] == 1, df["odds_home"] - 1, -1.0)
        ret_away = np.where(df["target"] == 0, df["odds_away"] - 1, -1.0)
        df["return_factor"] = np.select(
            [df["bet_side"] == "home", df["bet_side"] == "away"],
            [ret_home, ret_away],
            default=0.0,
        )

        df["strategy_return"] = df["stake_fraction"] * df["return_factor"]
        df["bankroll"] = self.bankroll * (1 + df["strategy_return"]).cumprod()

        placed_mask = df["bet_side"] != "none"
        placed_bets = df[placed_mask]
        if len(placed_bets) > 0:
            df.loc[placed_mask, "clv_pct"] = (
                (df.loc[placed_mask, "bet_odds"] / df.loc[placed_mask, "closing_odds"]) - 1
            ) * 100

        margin = 1.0
        df["p_market_home"] = margin / df["odds_home"]
        df["edge_home_pct"] = (df["p_model"] - df["p_market_home"]) * 100

        metrics = {
            "total_bets": float(len(placed_bets)),
            "final_bankroll": float(df["bankroll"].iloc[-1]) if not df.empty else float(self.bankroll),
            "roi": float(
                ((df["bankroll"].iloc[-1] - self.bankroll) / self.bankroll * 100)
                if not df.empty
                else 0.0
            ),
            "win_rate": float((placed_bets["return_factor"] > 0).mean() * 100)
            if len(placed_bets) > 0
            else 0.0,
            "avg_clv": float(
                ((placed_bets["bet_odds"] / placed_bets["closing_odds"]) - 1).mean() * 100
            )
            if len(placed_bets) > 0
            else 0.0,
        }
        metrics.update(enrich_backtest_metrics(df, placed_mask, self.bankroll))

        return df, metrics


FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
TARGET_COLUMN = "target"


def _cfg_ratios() -> Tuple[float, float]:
    bt = load_pipeline_config().get("backtest", {})
    return bt.get("train_ratio", 0.7), bt.get("val_ratio", 0.15)


def temporal_split(
    df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, vr = _cfg_ratios()
    train_ratio = train_ratio if train_ratio != 0.7 else tr
    val_ratio = val_ratio if val_ratio != 0.15 else vr
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:].copy()


def run_pipeline_from_dataframe(
    raw_data: pd.DataFrame,
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.1,
    min_ev: float = 0.03,
    save_model: bool = False,
    use_ensemble_model: Optional[bool] = None,
) -> Tuple[pd.DataFrame, Union[QuantModel, EnsembleModel], Dict[str, float], Dict[str, float]]:
    """Pelny pipeline: cechy -> trening -> backtest na zbiorze testowym."""
    if len(raw_data) < 200:
        raise ValueError("Za malo wierszy (min. 200 meczow) do sensownego podzialu train/val/test.")

    from pipeline.feature_builder import build_features, load_patch_history

    patch_df = load_patch_history()
    processed_data, feature_cols = build_features(raw_data, patch_df=patch_df if not patch_df.empty else None)
    train_df, val_df, test_df = temporal_split(processed_data)

    model = train_model(feature_cols, train_df, val_df, force_ensemble=use_ensemble_model)
    preds = model.predict_proba(test_df[feature_cols])
    test_df = test_df.copy()
    test_df["p_model"] = preds

    model_metrics = evaluate_classifier(test_df[TARGET_COLUMN].values, preds)
    if isinstance(model, EnsembleModel):
        model_metrics["ensemble_weights"] = model.weights_dict()

    if save_model:
        save_trained_model(model)

    backtester = VectorizedBacktester(
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        min_ev=min_ev,
    )
    results_df, backtest_metrics = backtester.run(test_df)
    backtest_metrics.update(model_metrics)

    return results_df, model, backtest_metrics, model_metrics


def run_pipeline(
    n_matches: int = 5000,
    initial_bankroll: float = 10000.0,
    kelly_fraction: float = 0.1,
    min_ev: float = 0.03,
    save_model: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], float]:
    raw_data = generate_full_pipeline_data(n_matches)
    results_df, _model, metrics, model_metrics = run_pipeline_from_dataframe(
        raw_data,
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        min_ev=min_ev,
        save_model=save_model,
    )
    return results_df, raw_data, metrics, model_metrics["roc_auc"]


if __name__ == "__main__":
    print("QuantBet - uruchamianie pipeline (dane syntetyczne)...")
    results, _test, metrics, auc = run_pipeline(n_matches=10000)
    print(f"ROC-AUC (test): {auc:.4f} | Log-loss: {metrics.get('log_loss', 0.0):.4f}")
    
    print("\n--- Wyniki Backtestu ---")
    print(f"Zaklady postawione: {int(metrics.get('total_bets', 0))}")
    print(f"ROI: {metrics.get('roi', 0.0):.2f}%")
    print(f"Poczatkowy bankroll: 10000.00 -> Koncowy: {metrics.get('final_bankroll', 0.0):.2f}")
    print(f"CLV Srednie: {metrics.get('avg_clv', 0.0):.2f}%")
    print(f"Srednia Wartosc Oczekiwana (EV) postawionych zakladow: {metrics.get('avg_bet_ev_pct', 0.0):.2f}%")
    print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0.0):.1f}%")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
    
    print("\n--- Dokladnosc Flagowania Betów ---")
    print(f"Win Rate dla postawionych zakladow: {metrics.get('bet_win_rate_pct', 0.0):.2f}%")
    print(f"Implied Win Rate z kursu: {metrics.get('implied_win_rate_pct', 0.0):.2f}%")
    print(f"Przecietne P(Win) z modelu: {metrics.get('model_p_mean_pct', 0.0):.2f}%")
    
    # Save the log
    results.to_csv("backtest_results.csv", index=False)
    print("\nZapisano backtest_results.csv")
