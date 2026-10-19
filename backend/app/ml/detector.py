from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest

from .data import METRIC_COLUMNS
from .preprocessing import FEATURE_COLUMNS, FeatureTransformer


@dataclass
class IsolationForestDetector:
    seed: int = 42
    contamination: float = 0.03
    threshold: float | None = None
    model: IsolationForest | None = None
    transformer: FeatureTransformer | None = None
    version: str = "isolation-forest-v1"

    def fit(self, training_data: pd.DataFrame) -> "IsolationForestDetector":
        if not 0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        self.transformer = FeatureTransformer().fit(training_data)
        normal = training_data.loc[~training_data["is_anomaly"]]
        features = self.transformer.transform(normal)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=1,
        ).fit(features)
        if self.threshold is None:
            self.threshold = 0.0
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.transformer is None or self.threshold is None:
            raise RuntimeError("detector must be fitted before prediction")
        scores = -self.model.decision_function(self.transformer.transform(data))
        result = data.copy()
        result["anomaly_score"] = scores
        result["is_predicted_anomaly"] = scores > self.threshold
        return result

    def metric_contributions(self, row: pd.Series, baseline: pd.Series) -> list[dict[str, float | str]]:
        contributions = []
        for metric in METRIC_COLUMNS:
            deviation = abs(float(row[metric]) - float(baseline[metric]))
            scale = float(self.transformer.scaler.scale_[FEATURE_COLUMNS.index(metric)]) if self.transformer and self.transformer.scaler else 1.0
            contributions.append({"metric": metric, "deviation": deviation, "standardized_deviation": deviation / max(scale, 1e-9)})
        return sorted(contributions, key=lambda item: float(item["standardized_deviation"]), reverse=True)
