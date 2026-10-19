from __future__ import annotations

import joblib
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import EvaluationRecord
from app.ml.data import METRIC_COLUMNS, generate_metrics, load_metrics, save_metrics
from app.ml.detector import IsolationForestDetector
from app.ml.drift import drift_report
from app.ml.evaluation import dataset_fingerprint, evaluate_fitted_detector
from app.ml.forecasting import evaluate_forecast, forecast_series
from app.ml.preprocessing import chronological_split, validate_metrics
from app.services.alerts import create_alert_records


class MonitoringService:
    def __init__(self, data: pd.DataFrame, training: pd.DataFrame, detector: IsolationForestDetector, scored: pd.DataFrame, evaluation: dict):
        self.data = data
        self.training = training
        self.detector = detector
        self.scored = scored
        self.evaluation = evaluation

    @classmethod
    def build(cls, data_path=None, artifact_path=None) -> "MonitoringService":
        source = data_path or settings.data_path
        artifact = artifact_path or settings.artifact_path
        if not source.exists():
            save_metrics(generate_metrics(seed=42), source)
        data = validate_metrics(load_metrics(source))
        training, holdout = chronological_split(data)
        fingerprint = dataset_fingerprint(data)
        detector = None
        if artifact.exists():
            stored = joblib.load(artifact)
            if isinstance(stored, dict) and stored.get("dataset_fingerprint") == fingerprint:
                detector = stored.get("detector")
            elif isinstance(stored, IsolationForestDetector):
                raise ValueError("persisted model has no dataset fingerprint; validate and explicitly promote a candidate model")
            else:
                raise ValueError("persisted model fingerprint does not match the dataset; validate and promote a candidate model")
        if not isinstance(detector, IsolationForestDetector):
            detector = IsolationForestDetector(seed=42, contamination=0.03).fit(training)
            artifact.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"detector": detector, "dataset_fingerprint": fingerprint}, artifact)
        evaluation = evaluate_fitted_detector(data, detector, training_data=training, testing_data=holdout)
        scored = detector.predict(holdout)
        first_service = data.service.iloc[0]
        service_data = data[data.service == first_service]
        evaluation["forecasting"] = {
            metric: evaluate_forecast(service_data, metric=metric) for metric in ("cpu_utilization", "api_latency_ms")
        }
        evaluation["drift"] = drift_report(training, holdout)
        return cls(data, training, detector, scored, evaluation)

    def seed_database(self, session: Session) -> None:
        fingerprint = self.evaluation["dataset_fingerprint"]
        existing = session.scalar(select(EvaluationRecord).where(EvaluationRecord.dataset_fingerprint == fingerprint))
        if existing is None:
            session.add(EvaluationRecord(model_version=self.evaluation["model_version"], dataset_fingerprint=fingerprint, payload=self.evaluation))
        else:
            existing.model_version = self.evaluation["model_version"]
            existing.payload = self.evaluation
        create_alert_records(self.scored, self.training.loc[~self.training.is_anomaly], session, self.detector)
        session.commit()

    def services(self) -> list[dict]:
        latest = self.data.sort_values("timestamp").groupby("service").tail(1)
        anomaly_counts = self.scored.groupby("service")["is_predicted_anomaly"].sum().to_dict()
        items = []
        for _, row in latest.iterrows():
            metrics = {metric: float(row[metric]) for metric in METRIC_COLUMNS}
            status = "critical" if metrics["cpu_utilization"] >= 92 or metrics["error_rate"] >= 3 else "watch" if anomaly_counts.get(row["service"], 0) else "healthy"
            items.append({"name": row["service"], "last_seen": row["timestamp"], "metrics": metrics,
                          "anomaly_count": int(anomaly_counts.get(row["service"], 0)), "status": status})
        return items

    def current_metrics(self, service: str | None = None) -> list[dict]:
        latest = self.data.sort_values("timestamp").groupby("service").tail(1)
        if service:
            latest = latest[latest.service == service]
        return [{"service": row["service"], "timestamp": row["timestamp"], **{metric: float(row[metric]) for metric in METRIC_COLUMNS}}
                for _, row in latest.iterrows()]

    def history(self, *, service: str, metric: str, limit: int) -> list[dict]:
        if metric not in METRIC_COLUMNS:
            raise ValueError("unknown metric")
        selected = self.data[self.data.service == service].sort_values("timestamp").tail(limit)
        scored = self.scored[self.scored.service == service].set_index("timestamp")
        return [{"timestamp": row.timestamp, "service": service, "metric": metric, "value": float(row[metric]),
                 "anomaly": bool(scored.loc[row.timestamp, "is_predicted_anomaly"]) if row.timestamp in scored.index else False,
                 "anomaly_score": float(scored.loc[row.timestamp, "anomaly_score"]) if row.timestamp in scored.index else None}
                for _, row in selected.iterrows()]

    def anomalies(self, *, service: str | None, limit: int) -> list[dict]:
        selected = self.scored[self.scored.is_predicted_anomaly]
        if service:
            selected = selected[selected.service == service]
        selected = selected.sort_values("timestamp", ascending=False).head(limit)
        return [{"timestamp": row.timestamp, "service": row.service, "anomaly_score": float(row.anomaly_score),
                 "metrics": {metric: float(row[metric]) for metric in METRIC_COLUMNS}} for _, row in selected.iterrows()]

    def forecasts(self, *, service: str, metric: str, horizon: int) -> list[dict]:
        history = self.data[self.data.service == service]
        if history.empty:
            raise ValueError("unknown service")
        return forecast_series(history, metric=metric, horizon=horizon).to_dict(orient="records")