import warnings
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")


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


class QuantModel:
    def __init__(self, features: List[str]):
        self.features = features
        self.base_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            n_jobs=-1,
        )
        self.calibrated_model = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        self.base_model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model, method="sigmoid", cv="prefit"
        )
        self.calibrated_model.fit(X_val, y_val)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated_model is None:
            raise ValueError("Model is not trained and calibrated yet.")
        return self.calibrated_model.predict_proba(X)[:, 1]


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

        placed_bets = df[df["bet_side"] != "none"]
        metrics = {
            "total_bets": float(len(placed_bets)),
            "final_bankroll": float(df["bankroll"].iloc[-1]) if not df.empty else float(self.bankroll),
            "roi": float(((df["bankroll"].iloc[-1] - self.bankroll) / self.bankroll * 100) if not df.empty else 0.0),
            "win_rate": float((placed_bets["return_factor"] > 0).mean() * 100) if len(placed_bets) > 0 else 0.0,
            "avg_clv": float(((placed_bets["bet_odds"] / placed_bets["closing_odds"]) - 1).mean() * 100) if len(placed_bets) > 0 else 0.0,
        }

        return df, metrics


def run_pipeline(n_matches: int = 5000) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    raw_data = generate_full_pipeline_data(n_matches)
    processed_data = generate_features(raw_data)

    n = len(processed_data)
    train_df = processed_data.iloc[: int(n * 0.7)]
    val_df = processed_data.iloc[int(n * 0.7) : int(n * 0.85)]
    test_df = processed_data.iloc[int(n * 0.85) :].copy()

    features = ["tA_winrate", "tB_winrate", "tA_kills", "tB_kills", "form_diff", "kills_diff"]
    target = "target"

    model = QuantModel(features)
    model.train(train_df[features], train_df[target], val_df[features], val_df[target])

    preds = model.predict_proba(test_df[features])
    test_df["p_model"] = preds

    backtester = VectorizedBacktester(initial_bankroll=10000.0, kelly_fraction=0.1, min_ev=0.03)
    results_df, metrics = backtester.run(test_df)

    return results_df, test_df, metrics
