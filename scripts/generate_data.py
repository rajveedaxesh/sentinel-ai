import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.ml.data import generate_metrics, save_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled synthetic infrastructure metrics.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--periods", type=int, default=24 * 14, help="Hourly points per service, maximum 90 days.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "sample" / "metrics.csv")
    arguments = parser.parse_args()
    destination = save_metrics(generate_metrics(seed=arguments.seed, periods=arguments.periods), arguments.output)
    print(f"Generated {arguments.periods * 3} synthetic rows at {destination}")


if __name__ == "__main__":
    main()