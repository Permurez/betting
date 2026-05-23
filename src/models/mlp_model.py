"""Prosta siec MLP (sklearn) z kalibracja – alternatywa dla LightGBM."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.neural_network import MLPClassifier


class MLPProbabilityModel:
    def __init__(self, features: List[str]):
        self.features = features
        self.base = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
        )
        self.calibrated = None

    def train(self, X_train, y_train, X_val, y_val) -> None:
        self.base.fit(X_train, y_train)
        self.calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(self.base),
            method="sigmoid",
        )
        self.calibrated.fit(X_val, y_val)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.calibrated is None:
            raise ValueError("MLP not trained.")
        return self.calibrated.predict_proba(X)[:, 1]
