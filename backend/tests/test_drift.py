from app.ml.data import generate_metrics
from app.ml.drift import drift_report


def test_distribution_drift_is_not_claimed_as_model_degradation():
    reference = generate_metrics(seed=4, periods=120)
    shifted = reference.copy()
    shifted["cpu_utilization"] = (shifted["cpu_utilization"] + 15).clip(0, 100)
    report = drift_report(reference, shifted)
    assert report["distribution_drift_detected"] is True
    assert report["model_degradation"]["confirmed"] is False
    assert "not assessed" in report["model_degradation"]["assessment"]


def test_degradation_requires_labeled_metric_comparison():
    data = generate_metrics(seed=4, periods=120)
    report = drift_report(data, data, labeled_metrics={"f1": 0.5}, reference_f1=0.8)
    assert report["model_degradation"]["confirmed"] is True