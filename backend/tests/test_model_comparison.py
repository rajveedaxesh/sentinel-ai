import numpy as np

from app.ml.data import generate_metrics
from scripts.compare_models import chronological_train_validation_test, compare_models, select_threshold


def test_three_way_split_uses_disjoint_global_timestamp_ranges():
    data = generate_metrics(seed=31, periods=120)
    training, validation, testing = chronological_train_validation_test(data)
    assert training.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < testing.timestamp.min()
    assert len(training) + len(validation) + len(testing) == len(data)


def test_threshold_is_selected_from_validation_scores():
    labels = np.array([False, False, True, True])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, metrics = select_threshold(labels, scores)
    assert threshold == 0.2
    assert metrics["f1"] == 1


def test_comparison_reports_common_untouched_test_and_validation_selection():
    report = compare_models(generate_metrics(seed=32, periods=120))
    assert report["selected_on_validation"] in report["candidates"]
    assert report["selected_candidate_test"] == report["candidates"][report["selected_on_validation"]]["test"]
    assert sum(report["baseline_three_sigma"]["test"]["confusion_matrix"].values()) == report["split"]["test_rows"]
    assert report["split"]["train_rows"] + report["split"]["validation_rows"] + report["split"]["test_rows"] == 360
    assert report["production_80_20_evaluation"]["baseline"]["confusion_matrix"]


def test_final_test_labels_do_not_change_model_or_threshold_selection():
    data = generate_metrics(seed=33, periods=120)
    _, _, testing = chronological_train_validation_test(data)
    first = compare_models(data)
    changed = data.copy()
    test_times = set(testing.timestamp)
    test_rows = changed.timestamp.isin(test_times)
    changed.loc[test_rows, "is_anomaly"] = ~changed.loc[test_rows, "is_anomaly"]
    second = compare_models(changed)
    assert first["selected_on_validation"] == second["selected_on_validation"]
    for model_name in first["candidates"]:
        assert first["candidates"][model_name]["threshold_selected_on_validation"] == second["candidates"][model_name]["threshold_selected_on_validation"]