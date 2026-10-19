from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from .data import METRIC_COLUMNS
from .detector import IsolationForestDetector
from .preprocessing import chronological_split, validate_metrics


def dataset_fingerprint(data: pd.DataFrame) -> str:
    canonical = validate_metrics(data).sort_values(["timestamp", "service"])
    payload = canonical.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S.%f%z",
        float_format="%.12g",
        lineterminator="\n",
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def three_sigma_scores(training: pd.DataFrame, testing: pd.DataFrame) -> np.ndarray:
    normal = training.loc[~training["is_anomaly"]]
    means = normal.groupby("service")[METRIC_COLUMNS].mean()
    stds = normal.groupby("service")[METRIC_COLUMNS].std().replace(0, 1).fillna(1)
    scores = np.zeros(len(testing), dtype=float)
    for service, indices in testing.groupby("service").groups.items():
        deviations = (testing.loc[indices, METRIC_COLUMNS] - means.loc[service]).abs().div(stds.loc[service])
        scores[testing.index.get_indexer(indices)] = deviations.max(axis=1).to_numpy()
    return scores


def _classification_metrics(truth: np.ndarray, scores: np.ndarray, predicted: np.ndarray) -> dict:
    matrix = confusion_matrix(truth, predicted, labels=[False, True])
    return {
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(truth, scores)) if truth.any() else 0.0,
        "confusion_matrix": {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])},
        "false_alarms": int(matrix[0, 1]),
    }


def evaluate_fitted_detector(
    data: pd.DataFrame,
    detector: IsolationForestDetector,
    *,
    training_data: pd.DataFrame | None = None,
    testing_data: pd.DataFrame | None = None,
) -> dict:
    if (training_data is None) != (testing_data is None):
        raise ValueError("training_data and testing_data must be supplied together")
    if training_data is None:
        training, testing = chronological_split(data)
    else:
        training = validate_metrics(training_data)
        testing = validate_metrics(testing_data)
        if training.empty or testing.empty or training.timestamp.max() >= testing.timestamp.min():
            raise ValueError("training and test periods must be non-empty and strictly chronological")
        if dataset_fingerprint(pd.concat([training, testing], ignore_index=True)) != dataset_fingerprint(data):
            raise ValueError("training and test partitions must exactly cover the evaluation dataset")
    predictions = detector.predict(testing)
    truth = testing["is_anomaly"].astype(bool).to_numpy()
    candidate = _classification_metrics(
        truth,
        predictions["anomaly_score"].to_numpy(),
        predictions["is_predicted_anomaly"].to_numpy(),
    )
    baseline_scores = three_sigma_scores(training, testing)
    baseline = _classification_metrics(truth, baseline_scores, baseline_scores > 3.0)
    return {
        "model_version": detector.version,
        "seed": detector.seed,
        "contamination": detector.contamination,
        "threshold": detector.threshold,
        "dataset_fingerprint": dataset_fingerprint(data),
        "train_rows": int(len(training)),
        "test_rows": int(len(testing)),
        "holdout_start": testing["timestamp"].min().isoformat(),
        "holdout_end": testing["timestamp"].max().isoformat(),
        **candidate,
        "baseline": baseline,
    }


def evaluate_detector(data: pd.DataFrame, *, seed: int = 42, contamination: float = 0.03) -> dict:
    training, testing = chronological_split(data)
    detector = IsolationForestDetector(seed=seed, contamination=contamination).fit(training)
    return evaluate_fitted_detector(data, detector, training_data=training, testing_data=testing)
