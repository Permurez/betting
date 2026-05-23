"""Ensemble LightGBM + MLP z wagami dopasowanymi na walidacji (log-loss)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss

from models.lgbm_model import QuantModel
from models.mlp_model import MLPProbabilityModel


@dataclass
class EnsembleWeights:
    lightgbm: float = 0.6
    mlp: float = 0.4


class EnsembleModel:
    def __init__(self, features: List[str]):
        self.features = features
        self.lgbm = QuantModel(features)
        self.mlp = MLPProbabilityModel(features)
        self.weights = EnsembleWeights()

    def train(self, X_train, y_train, X_val, y_val) -> None:
        self.lgbm.train(X_train, y_train, X_val, y_val)
        self.mlp.train(X_train, y_train, X_val, y_val)
        self.weights = self._fit_weights(X_val, y_val)

    def _fit_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> EnsembleWeights:
        p_lgb = self.lgbm.predict_proba(X_val)
        p_mlp = self.mlp.predict_proba(X_val)
        y = y_val.values.astype(float)

        def objective(w: np.ndarray) -> float:
            w1, w2 = w[0], 1.0 - w[0]
            p = np.clip(w1 * p_lgb + w2 * p_mlp, 1e-6, 1 - 1e-6)
            return log_loss(y, p)

        res = minimize(objective, x0=[0.5], bounds=[(0.05, 0.95)], method="L-BFGS-B")
        w_lgb = float(res.x[0])
        return EnsembleWeights(lightgbm=w_lgb, mlp=1.0 - w_lgb)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p_lgb = self.lgbm.predict_proba(X)
        p_mlp = self.mlp.predict_proba(X)
        w = self.weights
        return np.clip(w.lightgbm * p_lgb + w.mlp * p_mlp, 1e-6, 1 - 1e-6)

    def weights_dict(self) -> Dict[str, float]:
        return {"lightgbm": self.weights.lightgbm, "mlp": self.weights.mlp}
