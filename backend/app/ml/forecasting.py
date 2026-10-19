from __future__ import annotations

import numpy as np
import pandas as pd


def forecast_series(
    history: pd.DataFrame, *, metric: str, horizon: int = 24, interval_z: float = 1.96
) -> pd.DataFrame:
    """Forecast from past observations using a daily seasonal-naive baseline and recent trend."""
    if metric not in history.columns:
        raise ValueError(f"unknown metric: {metric}")
    if horizon < 1 or horizon > 168:
        raise ValueError("horizon must be between 1 and 168")
    ordered = history.sort_values("timestamp")[["timestamp", metric]].dropna()
    if len(ordered) < 2:
        raise ValueError("at least two observations are required")
    values = ordered[metric].astype(float).to_numpy()
    interval = 24 if len(values) >= 48 else max(1, len(values) // 2)
    seasonal = values[-interval:]
    recent = values[-min(len(values), 24):]
    slope = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) >= 3 else 0.0
    slope = float(np.clip(slope, -np.std(recent) * 0.1, np.std(recent) * 0.1))
    residuals = values[interval:] - values[:-interval] if len(values) > interval else values - np.mean(values)
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float(np.std(values))
    last_time = pd.Timestamp(ordered["timestamp"].iloc[-1])
    freq = pd.infer_freq(ordered["timestamp"].tail(min(len(ordered), 100))) or "h"
    timestamps = pd.date_range(last_time, periods=horizon + 1, freq=freq)[1:]
    predictions = np.array([seasonal[index % len(seasonal)] + slope * (index + 1) for index in range(horizon)])
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "metric": metric,
            "predicted_value": predictions,
            "lower_bound": predictions - interval_z * sigma,
            "upper_bound": predictions + interval_z * sigma,
        }
    )


def evaluate_forecast(data: pd.DataFrame, *, metric: str, test_fraction: float = 0.2) -> dict[str, float | int]:
    ordered = data.sort_values("timestamp").reset_index(drop=True)
    split_at = max(2, min(len(ordered) - 1, int(len(ordered) * (1 - test_fraction))))
    training, testing = ordered.iloc[:split_at], ordered.iloc[split_at:]
    predictions = forecast_series(training, metric=metric, horizon=len(testing))
    actual = testing[metric].astype(float).to_numpy()
    predicted = predictions["predicted_value"].to_numpy()
    errors = actual - predicted
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors**2))), "n": int(len(errors))}
