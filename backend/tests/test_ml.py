import numpy as np
import pandas as pd
import pytest

from app.ml.data import generate_metrics, save_metrics
from app.ml.detector import IsolationForestDetector
from app.ml.evaluation import evaluate_detector, evaluate_fitted_detector, three_sigma_scores
from app.ml.forecasting import evaluate_forecast, forecast_series
from app.ml.preprocessing import FeatureTransformer, chronological_split, validate_metrics


def test_generator_is_reproducible_and_labels_anomaly_types():
    first = generate_metrics(seed=19, periods=240)
    second = generate_metrics(seed=19, periods=240)
    pd.testing.assert_frame_equal(first, second)
    assert first["service"].nunique() == 3
    assert first["is_anomaly"].any()
    assert set(first.loc[first.is_anomaly, "anomaly_type"]) != {"normal"}


def test_generator_caps_size_and_csv_writes_stay_in_project_data():
    with pytest.raises(ValueError, match="90 days"):
        generate_metrics(periods=24 * 90 + 1)
    with pytest.raises(ValueError, match="project data directory"):
        save_metrics(generate_metrics(periods=48), "C:/outside/metrics.csv")


def test_validation_rejects_duplicate_and_invalid_ranges():
    data = generate_metrics(periods=72)
    with pytest.raises(ValueError, match="duplicate"):
        validate_metrics(pd.concat([data, data.iloc[[0]]], ignore_index=True))
    data.loc[0, "cpu_utilization"] = 101
    with pytest.raises(ValueError, match="valid range"):
        validate_metrics(data)


def test_validation_rejects_unknown_boolean_labels():
    data = generate_metrics(periods=72)
    data["is_anomaly"] = data["is_anomaly"].astype(object)
    data.loc[0, "is_anomaly"] = "sometimes"
    with pytest.raises(ValueError, match="boolean"):
        validate_metrics(data)


def test_split_is_chronological_and_transformer_fits_training_only():
    data = generate_metrics(periods=240)
    train, test = chronological_split(data)
    assert train.timestamp.max() < test.timestamp.min()
    transformer = FeatureTransformer().fit(train)
    fitted_mean = transformer.scaler.mean_.copy()
    altered_test = test.copy()
    altered_test.loc[:, "cpu_utilization"] = 99
    transformer.transform(altered_test)
    np.testing.assert_array_equal(fitted_mean, transformer.scaler.mean_)


def test_detector_excludes_label_columns_and_predicts_scores():
    data = generate_metrics(periods=240)
    train, test = chronological_split(data)
    detector = IsolationForestDetector().fit(train)
    output = detector.predict(test)
    assert "is_anomaly" not in detector.transformer.scaler.feature_names_in_
    assert detector.threshold == 0.0
    assert output.anomaly_score.notna().all()
    assert output.is_predicted_anomaly.dtype == bool


def test_forecast_uses_history_and_evaluation_reports_errors():
    data = generate_metrics(periods=240)
    service_data = data[data.service == "checkout"].reset_index(drop=True)
    forecast = forecast_series(service_data.iloc[:180], metric="cpu_utilization", horizon=12)
    assert len(forecast) == 12
    assert forecast.timestamp.min() > service_data.timestamp.iloc[179]
    result = evaluate_forecast(service_data, metric="cpu_utilization")
    assert result["n"] > 0 and result["mae"] >= 0 and result["rmse"] >= 0


def test_evaluation_calculates_metrics_from_holdout():
    data = generate_metrics(periods=240)
    result = evaluate_detector(data)
    assert result["test_rows"] > 0
    assert 0 <= result["precision"] <= 1
    assert result["false_alarms"] == result["confusion_matrix"]["fp"]
    assert result["dataset_fingerprint"]
    assert result["holdout_start"] <= result["holdout_end"]
    assert sum(result["baseline"]["confusion_matrix"].values()) == result["test_rows"]
    assert result["baseline"]["false_alarms"] == result["baseline"]["confusion_matrix"]["fp"]
    assert 0 <= result["baseline"]["average_precision"] <= 1


def test_three_sigma_statistics_ignore_labeled_training_anomalies():
    data = generate_metrics(periods=240)
    training, testing = chronological_split(data)
    original_scores = three_sigma_scores(training, testing)
    changed_training = training.copy()
    changed_training.loc[changed_training["is_anomaly"], "cpu_utilization"] = 100
    changed_scores = three_sigma_scores(changed_training, testing)
    np.testing.assert_array_equal(original_scores, changed_scores)


def test_fitted_evaluation_requires_exact_chronological_partition_coverage():
    data = generate_metrics(periods=240)
    training, testing = chronological_split(data)
    detector = IsolationForestDetector().fit(training)
    report = evaluate_fitted_detector(data, detector, training_data=training, testing_data=testing)
    assert report["test_rows"] == len(testing)
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate_fitted_detector(data, detector, training_data=training, testing_data=testing.iloc[:-3])
