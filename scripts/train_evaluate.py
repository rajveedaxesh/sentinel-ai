import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.db.session import Base, SessionLocal, engine
from app.ml.data import load_metrics
from app.services.monitoring import MonitoringService


def main() -> None:
    service = MonitoringService.build()
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        service.seed_database(session)
    print(f"Model: {service.evaluation['model_version']}")
    print(f"Dataset fingerprint: {service.evaluation['dataset_fingerprint']}")
    print(f"Holdout precision={service.evaluation['precision']:.3f} recall={service.evaluation['recall']:.3f} f1={service.evaluation['f1']:.3f}")
    print(f"False alarms: {service.evaluation['false_alarms']}")
    print(f"Forecast holdout: {service.evaluation['forecasting']}")


if __name__ == "__main__":
    main()