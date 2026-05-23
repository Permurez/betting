"""LightGBM + Platt scaling."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


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
            estimator=FrozenEstimator(self.base_model),
            method="sigmoid",
        )
        self.calibrated_model.fit(X_val, y_val)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated_model is None:
            raise ValueError("Model is not trained and calibrated yet.")
        return self.calibrated_model.predict_proba(X)[:, 1]

    def save(self, path: Optional[Path] = None) -> Path:
        if self.calibrated_model is None:
            raise ValueError("Brak wytrenowanego modelu do zapisu.")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        out = path or MODELS_DIR / "quant_model.joblib"
        joblib.dump(
            {"features": self.features, "calibrated_model": self.calibrated_model},
            out,
        )
        return Path(out)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "QuantModel":
        src = path or MODELS_DIR / "quant_model.joblib"
        payload = joblib.load(src)
        model = cls(payload["features"])
        model.calibrated_model = payload["calibrated_model"]
        return model
