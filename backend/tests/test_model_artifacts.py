import joblib
import pandas as pd
import pytest

from app.ml.data import generate_metrics
from app.ml.detector import IsolationForestDetector
from app.ml.evaluation import dataset_fingerprint, evaluate_detector
from app.ml.preprocessing import chronological_split
from app.services.monitoring import MonitoringService
from scripts.promote_candidate import promote_candidate


def _write_dataset(path, data):
    data.to_csv(path, index=False)
    return path


def test_dataset_fingerprint_is_stable_across_csv_roundtrip(tmp_path):
    data = generate_metrics(seed=11, periods=96)
    dataset_path = _write_dataset(tmp_path / "metrics.csv", data)
    loaded = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    assert dataset_fingerprint(data) == dataset_fingerprint(loaded)


def test_matching_artifact_is_reused_and_mismatch_does_not_replace_it(tmp_path):
    data = generate_metrics(seed=13, periods=120)
    dataset_path = _write_dataset(tmp_path / "metrics.csv", data)
    artifact_path = tmp_path / "active.joblib"
    first = MonitoringService.build(dataset_path, artifact_path)
    original = artifact_path.read_bytes()

    reused = MonitoringService.build(dataset_path, artifact_path)
    assert reused.detector.seed == first.detector.seed
    assert artifact_path.read_bytes() == original

    changed = data.copy()
    changed["cpu_utilization"] = (changed["cpu_utilization"] + 0.1).clip(0, 100)
    changed_path = _write_dataset(tmp_path / "changed.csv", changed)
    with pytest.raises(ValueError, match="fingerprint does not match"):
        MonitoringService.build(changed_path, artifact_path)
    assert artifact_path.read_bytes() == original


def test_legacy_artifact_without_fingerprint_is_not_silently_retrained(tmp_path):
    data = generate_metrics(seed=7, periods=120)
    dataset_path = _write_dataset(tmp_path / "metrics.csv", data)
    training, _ = chronological_split(data)
    legacy_detector = IsolationForestDetector().fit(training)
    artifact_path = tmp_path / "legacy.joblib"
    joblib.dump(legacy_detector, artifact_path)
    original = artifact_path.read_bytes()

    with pytest.raises(ValueError, match="no dataset fingerprint"):
        MonitoringService.build(dataset_path, artifact_path)
    assert artifact_path.read_bytes() == original


@pytest.mark.parametrize("reason", ["minimum", "baseline", "fingerprint"])
def test_rejected_candidate_never_changes_active_artifact(tmp_path, reason):
    data = generate_metrics(seed=5, periods=240)
    dataset_path = _write_dataset(tmp_path / "metrics.csv", data)
    fingerprint = dataset_fingerprint(data)
    training, _ = chronological_split(data)
    detector = IsolationForestDetector().fit(training)
    detector.threshold = float("inf")
    evaluation = evaluate_detector(data)
    evaluation["f1"] = 0.99
    bundle_fingerprint = "not-the-current-data" if reason == "fingerprint" else fingerprint
    candidates_root = tmp_path / "candidates"
    candidates_root.mkdir()
    candidate_path = candidates_root / "candidate.joblib"
    joblib.dump({"detector": detector, "dataset_fingerprint": bundle_fingerprint, "evaluation": evaluation}, candidate_path)
    active_path = tmp_path / "active.joblib"
    active_path.write_bytes(b"existing-active-artifact")
    original = active_path.read_bytes()

    with pytest.raises(ValueError):
        promote_candidate(candidate_path, minimum_f1=0.5 if reason != "baseline" else 0,
                          candidates_root=candidates_root,
                          dataset_path=dataset_path, active_path=active_path)
    assert active_path.read_bytes() == original


def test_qualified_candidate_is_promoted_atomically(tmp_path):
    data = generate_metrics(seed=5, periods=120)
    dataset_path = _write_dataset(tmp_path / "metrics.csv", data)
    fingerprint = dataset_fingerprint(data)
    training, _ = chronological_split(data)
    detector = IsolationForestDetector().fit(training)
    detector.threshold = float("-inf")
    evaluation = evaluate_detector(data)
    evaluation["dataset_fingerprint"] = fingerprint
    candidates_root = tmp_path / "candidates"
    candidates_root.mkdir()
    candidate_path = candidates_root / "candidate.joblib"
    joblib.dump({"detector": detector, "dataset_fingerprint": fingerprint, "evaluation": evaluation}, candidate_path)
    active_path = tmp_path / "active.joblib"

    promoted = promote_candidate(candidate_path, minimum_f1=0.4, candidates_root=candidates_root,
                                 dataset_path=dataset_path, active_path=active_path,
                                 allow_baseline_regression=True)
    stored = joblib.load(promoted)
    assert stored["dataset_fingerprint"] == fingerprint
    assert isinstance(stored["detector"], IsolationForestDetector)