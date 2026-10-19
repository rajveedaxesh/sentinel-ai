from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data import METRIC_COLUMNS, REQUIRED_COLUMNS

FEATURE_COLUMNS = [*METRIC_COLUMNS, "hour_sin", "hour_cos", "day_sin", "day_cos"]
RANGES = {
    "cpu_utilization": (0, 100),
    "memory_utilization": (0, 100),
    "request_rate": (0, None),
    "api_latency_ms": (0, None),
    "error_rate": (0, 100),
    "network_traffic_mbps": (0, None),
}


def validate_metrics(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {', '.join(missing_columns)}")
    clean = data.loc[:, REQUIRED_COLUMNS].copy()
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    if clean["timestamp"].isna().any():
        raise ValueError("timestamps must be valid")
    if clean["service"].isna().any() or (clean["service"].astype(str).str.strip().str.len() == 0).any():
        raise ValueError("service must be present")
    if clean.duplicated(["timestamp", "service"]).any():
        raise ValueError("duplicate timestamp/service records are not allowed")
    for column, (minimum, maximum) in RANGES.items():
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        if clean[column].isna().any() or not np.isfinite(clean[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
        if (clean[column] < minimum).any() or (maximum is not None and (clean[column] > maximum).any()):
            raise ValueError(f"{column} is outside the valid range")
    def parse_label(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        return None

    labels = clean["is_anomaly"].map(parse_label)
    if labels.isna().any():
        raise ValueError("is_anomaly must contain boolean values")
    if clean["anomaly_type"].isna().any() or (clean["anomaly_type"].astype(str).str.strip().str.len() == 0).any():
        raise ValueError("anomaly_type must be present")
    clean["is_anomaly"] = labels.astype(bool)
    clean["anomaly_type"] = clean["anomaly_type"].astype(str)
    return clean.sort_values(["service", "timestamp"]).reset_index(drop=True)


def add_time_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    hour = result["timestamp"].dt.hour + result["timestamp"].dt.minute / 60
    day = result["timestamp"].dt.dayofweek
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    result["day_sin"] = np.sin(2 * np.pi * day / 7)
    result["day_cos"] = np.cos(2 * np.pi * day / 7)
    return result


def chronological_split(data: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    ordered = validate_metrics(data)
    timestamps = np.sort(ordered["timestamp"].unique())
    if len(timestamps) < 2:
        raise ValueError("at least two unique timestamps are required")
    split_at = max(1, min(len(timestamps) - 1, int(len(timestamps) * (1 - test_fraction))))
    boundary = timestamps[split_at]
    train = ordered[ordered["timestamp"] < boundary]
    test = ordered[ordered["timestamp"] >= boundary]
    return train.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class FeatureTransformer:
    scaler: StandardScaler | None = None

    def fit(self, training_data: pd.DataFrame) -> "FeatureTransformer":
        featured = add_time_features(training_data)
        normal = featured.loc[~featured["is_anomaly"], FEATURE_COLUMNS]
        if len(normal) < 2:
            raise ValueError("at least two normal training rows are required")
        self.scaler = StandardScaler().fit(normal)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("transformer must be fitted on training data first")
        featured = add_time_features(data)
        return self.scaler.transform(featured[FEATURE_COLUMNS])
