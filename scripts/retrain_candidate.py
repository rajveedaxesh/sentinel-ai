import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import joblib

from app.ml.data import load_metrics
from app.ml.detector import IsolationForestDetector
from app.ml.evaluation import evaluate_detector
from app.ml.preprocessing import chronological_split, validate_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a candidate without promoting it.")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "sample" / "metrics.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.03)
    arguments = parser.parse_args()
    dataset = validate_metrics(load_metrics(arguments.data))
    evaluation = evaluate_detector(dataset, seed=arguments.seed, contamination=arguments.contamination)
    training, _ = chronological_split(dataset)
    candidate = IsolationForestDetector(seed=arguments.seed, contamination=arguments.contamination).fit(training)
    candidates = ROOT / "models" / "candidates"
    candidates.mkdir(parents=True, exist_ok=True)
    artifact = candidates / f"{evaluation['dataset_fingerprint'][:12]}-{arguments.seed}-c{arguments.contamination:g}.joblib"
    joblib.dump({
        "detector": candidate,
        "dataset_fingerprint": evaluation["dataset_fingerprint"],
        "evaluation": evaluation,
    }, artifact)
    print(f"Candidate artifact: {artifact}")
    print(f"Evaluation: precision={evaluation['precision']:.3f} recall={evaluation['recall']:.3f} f1={evaluation['f1']:.3f} false_alarms={evaluation['false_alarms']}")
    print(f"Three-sigma baseline F1: {evaluation['baseline']['f1']:.3f}")
    print("Candidate was not promoted. Review holdout results and explicitly approve any deployment separately.")


if __name__ == "__main__":
    main()