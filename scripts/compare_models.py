from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import settings
from app.ml.data import METRIC_COLUMNS, load_metrics
from app.ml.detector import IsolationForestDetector
from app.ml.evaluation import dataset_fingerprint, evaluate_fitted_detector, three_sigma_scores
from app.ml.preprocessing import FeatureTransformer, chronological_split, validate_metrics


def chronological_train_validation_test(
    data: pd.DataFrame, *, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must be positive and leave a test period")
    ordered = validate_metrics(data)
    timestamps = np.sort(ordered["timestamp"].unique())
    if len(timestamps) < 5:
        raise ValueError("at least five unique timestamps are required")
    train_end = max(1, min(len(timestamps) - 2, int(len(timestamps) * train_fraction)))
    validation_end = max(train_end + 1, min(len(timestamps) - 1, int(len(timestamps) * (train_fraction + validation_fraction))))
    train_boundary = timestamps[train_end]
    validation_boundary = timestamps[validation_end]
    return (
        ordered[ordered.timestamp < train_boundary].reset_index(drop=True),
        ordered[(ordered.timestamp >= train_boundary) & (ordered.timestamp < validation_boundary)].reset_index(drop=True),
        ordered[ordered.timestamp >= validation_boundary].reset_index(drop=True),
    )


def _metrics(labels: np.ndarray, scores: np.ndarray, flags: np.ndarray) -> dict:
    matrix = confusion_matrix(labels, flags, labels=[False, True])
    return {
        "precision": float(precision_score(labels, flags, zero_division=0)),
        "recall": float(recall_score(labels, flags, zero_division=0)),
        "f1": float(f1_score(labels, flags, zero_division=0)),
        "average_precision": float(average_precision_score(labels, scores)) if labels.any() else 0.0,
        "false_alarms": int(matrix[0, 1]),
        "confusion_matrix": {"tn": int(matrix[0, 0]), "fp": int(matrix[0, 1]), "fn": int(matrix[1, 0]), "tp": int(matrix[1, 1])},
    }


def select_threshold(validation_labels: np.ndarray, validation_scores: np.ndarray) -> tuple[float, dict]:
    if not np.isfinite(validation_scores).all():
        raise ValueError("validation scores must be finite")
    candidates = np.append(np.unique(validation_scores), np.nextafter(validation_scores.max(), np.inf))
    positives = int(validation_labels.sum())
    average_precision = float(average_precision_score(validation_labels, validation_scores)) if positives else 0.0
    best_threshold = float(candidates[0])
    best_metrics = None
    best_key = None
    for threshold in candidates:
        predicted = validation_scores > threshold
        true_positives = int(np.count_nonzero(validation_labels & predicted))
        false_positives = int(np.count_nonzero(~validation_labels & predicted))
        false_negatives = positives - true_positives
        true_negatives = int(len(validation_labels) - positives - false_positives)
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
        recall = true_positives / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": average_precision,
            "false_alarms": false_positives,
            "confusion_matrix": {"tn": true_negatives, "fp": false_positives, "fn": false_negatives, "tp": true_positives},
        }
        key = (result["f1"], result["precision"], -result["false_alarms"], result["average_precision"])
        if best_key is None or key > best_key:
            best_threshold, best_metrics, best_key = float(threshold), result, key
    return best_threshold, best_metrics


def _fit_isolation_forest(training: pd.DataFrame, *, per_service: bool, trees: int, contamination: float, seed: int) -> dict:
    fitted = {}
    groups = training.groupby("service", sort=True) if per_service else [("__all__", training)]
    for service, service_training in groups:
        transformer = FeatureTransformer().fit(service_training)
        normal = service_training.loc[~service_training.is_anomaly]
        model = IsolationForest(n_estimators=trees, contamination=contamination, random_state=seed, n_jobs=1)
        model.fit(transformer.transform(normal))
        fitted[service] = (transformer, model)
    return {"per_service": per_service, "models": fitted}


def _isolation_scores(fitted: dict, data: pd.DataFrame) -> np.ndarray:
    scores = np.zeros(len(data), dtype=float)
    for service, indices in data.groupby("service", sort=True).groups.items():
        model_key = service if fitted["per_service"] else "__all__"
        transformer, model = fitted["models"][model_key]
        scores[data.index.get_indexer(indices)] = -model.decision_function(transformer.transform(data.loc[indices]))
    return scores


def _fit_robust_seasonal(training: pd.DataFrame) -> dict:
    normal = training.loc[~training.is_anomaly].copy()
    normal["hour"] = normal.timestamp.dt.hour
    grouped = normal.groupby(["service", "hour"])[METRIC_COLUMNS]
    medians = grouped.median()
    mads = grouped.agg(lambda values: float(np.median(np.abs(values - np.median(values)))))
    service_medians = normal.groupby("service")[METRIC_COLUMNS].median()
    service_mads = normal.groupby("service")[METRIC_COLUMNS].agg(
        lambda values: float(np.median(np.abs(values - np.median(values))))
    )
    service_scales = normal.groupby("service")[METRIC_COLUMNS].std().fillna(1)
    return {"medians": medians, "mads": mads, "service_medians": service_medians, "service_mads": service_mads, "service_scales": service_scales}


def _robust_hourly_residuals(fitted: dict, data: pd.DataFrame) -> np.ndarray:
    residuals = np.zeros((len(data), len(METRIC_COLUMNS)), dtype=float)
    for position, (_, row) in enumerate(data.iterrows()):
        key = (row.service, row.timestamp.hour)
        if key in fitted["medians"].index:
            center = fitted["medians"].loc[key]
            mad = fitted["mads"].loc[key]
        else:
            center = fitted["service_medians"].loc[row.service]
            mad = fitted["service_mads"].loc[row.service]
        fallback = fitted["service_scales"].loc[row.service]
        robust_scale = (1.4826 * mad).where(mad > 1e-8, fallback).clip(lower=1e-8)
        residuals[position] = ((row[METRIC_COLUMNS] - center) / robust_scale).to_numpy(dtype=float)
    return residuals


def _fit_seasonal_residual_iforest(training: pd.DataFrame, *, trees: int, contamination: float, seed: int) -> dict:
    seasonal = _fit_robust_seasonal(training)
    normal = training.loc[~training.is_anomaly]
    residuals = _robust_hourly_residuals(seasonal, normal)
    scaler = StandardScaler().fit(residuals)
    model = IsolationForest(n_estimators=trees, contamination=contamination, random_state=seed, n_jobs=1)
    model.fit(scaler.transform(residuals))
    return {"seasonal": seasonal, "scaler": scaler, "model": model}


def _seasonal_residual_iforest_scores(fitted: dict, data: pd.DataFrame) -> np.ndarray:
    residuals = _robust_hourly_residuals(fitted["seasonal"], data)
    return -fitted["model"].decision_function(fitted["scaler"].transform(residuals))


def _robust_seasonal_scores(fitted: dict, data: pd.DataFrame) -> np.ndarray:
    return np.max(np.abs(_robust_hourly_residuals(fitted, data)), axis=1)


def _label_summary(data: pd.DataFrame) -> dict:
    counts = data.loc[data.is_anomaly].groupby(["service", "anomaly_type"]).size()
    return {f"{service}:{kind}": int(count) for (service, kind), count in counts.items()}


def _false_positive_diagnostics(training: pd.DataFrame, testing: pd.DataFrame, flags: np.ndarray) -> dict:
    normal = training.loc[~training.is_anomaly]
    centers = normal.groupby("service")[METRIC_COLUMNS].median()
    scales = normal.groupby("service")[METRIC_COLUMNS].std().replace(0, 1).fillna(1)
    false_positive_positions = np.flatnonzero(flags & ~testing.is_anomaly.to_numpy())
    false_negative_rows = testing.loc[~flags & testing.is_anomaly]
    by_service: dict[str, int] = {}
    by_utc_hour: dict[str, int] = {}
    dominant_deviations: dict[str, int] = {}
    for position in false_positive_positions:
        row = testing.iloc[position]
        service = str(row.service)
        by_service[service] = by_service.get(service, 0) + 1
        hour = f"{row.timestamp.hour:02d}:00"
        by_utc_hour[hour] = by_utc_hour.get(hour, 0) + 1
        deviations = (row[METRIC_COLUMNS] - centers.loc[service]).abs() / scales.loc[service]
        metric = str(deviations.idxmax())
        dominant_deviations[metric] = dominant_deviations.get(metric, 0) + 1
    missed = false_negative_rows.groupby(["service", "anomaly_type"]).size()
    return {
        "false_positives_by_service": by_service,
        "false_positives_by_utc_hour": by_utc_hour,
        "largest_univariate_standardized_deviation": dominant_deviations,
        "false_negatives_by_service_and_type": {f"{service}:{kind}": int(count) for (service, kind), count in missed.items()},
    }


def compare_models(data: pd.DataFrame, *, seed: int = 42) -> dict:
    training, validation, testing = chronological_train_validation_test(data)
    validation_labels = validation.is_anomaly.to_numpy(dtype=bool)
    test_labels = testing.is_anomaly.to_numpy(dtype=bool)

    baseline_validation_score = three_sigma_scores(training, validation)
    baseline_test_score = three_sigma_scores(training, testing)
    baseline_test_flags = baseline_test_score > 3
    baseline_validation = _metrics(validation_labels, baseline_validation_score, baseline_validation_score > 3)
    baseline_test = _metrics(test_labels, baseline_test_score, baseline_test_flags)
    baseline_test["diagnostics"] = _false_positive_diagnostics(training, testing, baseline_test_flags)

    candidate_factories = {
        "iforest_pooled_200_c0.03": lambda: (_fit_isolation_forest(training, per_service=False, trees=200, contamination=0.03, seed=seed), "iforest"),
        "iforest_pooled_200_c0.01": lambda: (_fit_isolation_forest(training, per_service=False, trees=200, contamination=0.01, seed=seed), "iforest"),
        "iforest_pooled_200_c0.10": lambda: (_fit_isolation_forest(training, per_service=False, trees=200, contamination=0.10, seed=seed), "iforest"),
        "iforest_pooled_500_c0.03": lambda: (_fit_isolation_forest(training, per_service=False, trees=500, contamination=0.03, seed=seed), "iforest"),
        "iforest_per_service_200_c0.03": lambda: (_fit_isolation_forest(training, per_service=True, trees=200, contamination=0.03, seed=seed), "iforest"),
        "iforest_seasonal_residual_200_c0.03": lambda: (_fit_seasonal_residual_iforest(training, trees=200, contamination=0.03, seed=seed), "seasonal_iforest"),
        "robust_hourly_mad": lambda: (_fit_robust_seasonal(training), "robust"),
    }
    results = {}
    for name, factory in candidate_factories.items():
        fitted, kind = factory()
        scorer = {
            "iforest": _isolation_scores,
            "seasonal_iforest": _seasonal_residual_iforest_scores,
            "robust": _robust_seasonal_scores,
        }[kind]
        validation_scores = scorer(fitted, validation)
        threshold, validation_metrics = select_threshold(validation_labels, validation_scores)
        test_scores = scorer(fitted, testing)
        test_flags = test_scores > threshold
        results[name] = {
            "threshold_selected_on_validation": threshold,
            "validation": validation_metrics,
            "test": _metrics(test_labels, test_scores, test_flags),
            "test_false_positive_diagnostics": _false_positive_diagnostics(training, testing, test_flags),
        }

    selected_name = max(
        results,
        key=lambda name: (
            results[name]["validation"]["f1"],
            results[name]["validation"]["precision"],
            -results[name]["validation"]["false_alarms"],
            results[name]["validation"]["average_precision"],
        ),
    )
    production_training, production_testing = chronological_split(data)
    production_detector = IsolationForestDetector(seed=seed, contamination=0.03).fit(production_training)
    production_predictions = production_detector.predict(production_testing)
    production_flags = production_predictions.is_predicted_anomaly.to_numpy()
    production_scores = production_predictions.anomaly_score
    production_score_ranges = {
        "normal": production_scores.loc[~production_testing.is_anomaly].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(),
        "anomaly": production_scores.loc[production_testing.is_anomaly].quantile([0, 0.25, 0.5, 0.75, 1]).to_dict(),
    }
    production_evaluation = evaluate_fitted_detector(
        data,
        production_detector,
        training_data=production_training,
        testing_data=production_testing,
    )
    production_evaluation["false_positive_diagnostics"] = _false_positive_diagnostics(
        production_training, production_testing, production_flags
    )
    production_evaluation["score_quantiles_by_truth"] = production_score_ranges
    return {
        "dataset_fingerprint": dataset_fingerprint(data),
        "seed": seed,
        "split": {
            "train_rows": len(training),
            "validation_rows": len(validation),
            "test_rows": len(testing),
            "train_end": training.timestamp.max().isoformat(),
            "validation_start": validation.timestamp.min().isoformat(),
            "validation_end": validation.timestamp.max().isoformat(),
            "test_start": testing.timestamp.min().isoformat(),
            "test_end": testing.timestamp.max().isoformat(),
        },
        "anomaly_labels_by_period": {
            "train": _label_summary(training),
            "validation": _label_summary(validation),
            "test": _label_summary(testing),
        },
        "baseline_three_sigma": {"validation": baseline_validation, "test": baseline_test},
        "production_80_20_evaluation": production_evaluation,
        "candidates": results,
        "selected_on_validation": selected_name,
        "selected_candidate_test": results[selected_name]["test"],
        "selected_candidate_test_diagnostics": results[selected_name]["test_false_positive_diagnostics"],
        "selection_rule": "Highest validation F1, then validation precision, fewer validation false alarms, and validation average precision; test metrics are not used for selection or threshold tuning.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare anomaly models using chronological validation and test periods.")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "model-comparison.json")
    arguments = parser.parse_args()
    data = validate_metrics(load_metrics(settings.data_path))
    destination = arguments.output.resolve()
    docs_directory = (ROOT / "docs").resolve()
    if not destination.is_relative_to(docs_directory):
        parser.error("report output must remain within the project docs directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(compare_models(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Comparison report written to {destination}")


if __name__ == "__main__":
    main()