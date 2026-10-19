import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import joblib

from app.core.config import settings
from app.ml.data import load_metrics
from app.ml.detector import IsolationForestDetector
from app.ml.evaluation import dataset_fingerprint, evaluate_fitted_detector
from app.ml.preprocessing import chronological_split, validate_metrics


def promote_candidate(
    candidate_path: Path,
    *,
    minimum_f1: float,
    allow_baseline_regression: bool = False,
    candidates_root: Path | None = None,
    dataset_path: Path | None = None,
    active_path: Path | None = None,
) -> Path:
    if not 0 <= minimum_f1 <= 1:
        raise ValueError("minimum_f1 must be between 0 and 1")
    candidate_path = candidate_path.resolve()
    candidates_root = (candidates_root or ROOT / "models" / "candidates").resolve()
    if not candidate_path.is_relative_to(candidates_root):
        raise ValueError("candidate must be located within models/candidates")
    bundle = joblib.load(candidate_path)
    if not isinstance(bundle, dict) or not isinstance(bundle.get("detector"), IsolationForestDetector) or "evaluation" not in bundle:
        raise ValueError("candidate artifact is missing a detector or evaluation metadata")
    dataset = validate_metrics(load_metrics(dataset_path or settings.data_path))
    if bundle["dataset_fingerprint"] != dataset_fingerprint(dataset):
        raise ValueError("candidate evaluation fingerprint does not match the current dataset")
    training, testing = chronological_split(dataset)
    evaluation = evaluate_fitted_detector(dataset, bundle["detector"], training_data=training, testing_data=testing)
    if float(evaluation["f1"]) < minimum_f1:
        raise ValueError(f"recomputed candidate F1 {evaluation['f1']:.3f} is below the supplied minimum {minimum_f1:.3f}")
    baseline_f1 = evaluation["baseline"]["f1"]
    if evaluation["f1"] < baseline_f1 and not allow_baseline_regression:
        raise ValueError(f"recomputed candidate F1 {evaluation['f1']:.3f} is below baseline F1 {baseline_f1:.3f}; validate and explicitly allow this regression to continue")
    destination = active_path or settings.artifact_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".joblib", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump({"detector": bundle["detector"], "dataset_fingerprint": bundle["dataset_fingerprint"]}, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path and temporary_path.exists():
            temporary_path.unlink()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and explicitly promote a candidate model artifact.")
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--minimum-f1", type=float, required=True)
    parser.add_argument("--allow-baseline-regression", action="store_true")
    arguments = parser.parse_args()
    try:
        destination = promote_candidate(
            arguments.candidate,
            minimum_f1=arguments.minimum_f1,
            allow_baseline_regression=arguments.allow_baseline_regression,
        )
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(f"Promoted validated candidate to {destination}")


if __name__ == "__main__":
    main()