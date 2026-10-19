import json
import logging

import joblib
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base
from app.db.session import get_db
from app.ml.data import generate_metrics
import app.main as main

app = main.app


def test_structured_log_formatter_emits_valid_json():
    record = logging.LogRecord("sentinelai.test", logging.INFO, "test.py", 1, 'metric "cpu"=%s', (42,), None)
    payload = json.loads(main.StructuredFormatter().format(record))
    assert payload["level"] == "info"
    assert payload["logger"] == "sentinelai.test"
    assert payload["message"] == 'metric "cpu"=42'


def _isolate_database(tmp_path, monkeypatch):
    database_path = tmp_path / "api-test.db"
    test_engine = create_engine(f"sqlite:///{database_path.as_posix()}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    test_sessions = sessionmaker(bind=test_engine, autoflush=False, expire_on_commit=False)
    monkeypatch.setattr(main, "engine", test_engine)
    monkeypatch.setattr(main, "SessionLocal", test_sessions)

    def override_get_db():
        session = test_sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    return test_engine


def test_demo_api_health_data_forecast_and_alert_lifecycle(tmp_path, monkeypatch):
    test_engine = _isolate_database(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "artifact_path", tmp_path / "isolation_forest.joblib")
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["demo_data"] is True
        assert health.json()["model_ready"] is True

        services = client.get("/api/v1/services").json()
        assert len(services) == 3
        service = services[0]["name"]
        history = client.get("/api/v1/metrics/history", params={"service": service, "metric": "cpu_utilization", "limit": 24})
        assert history.status_code == 200 and len(history.json()) == 24

        forecast = client.get("/api/v1/forecasts", params={"service": service, "metric": "cpu_utilization", "horizon": 8})
        assert forecast.status_code == 200 and len(forecast.json()) == 8
        assert forecast.json()[0]["lower_bound"] <= forecast.json()[0]["predicted_value"]

        alerts = client.get("/api/v1/alerts", params={"page": 1, "page_size": 10})
        assert alerts.status_code == 200
        if alerts.json()["items"]:
            alert_id = alerts.json()["items"][0]["id"]
            updated = client.patch(f"/api/v1/alerts/{alert_id}", json={"status": "acknowledged"})
            assert updated.status_code == 200
            assert updated.json()["status"] == "acknowledged"
            persisted = client.get("/api/v1/alerts", params={"status": "acknowledged"})
            assert any(item["id"] == alert_id for item in persisted.json()["items"])

        evaluation = client.get("/api/v1/evaluation")
        assert evaluation.status_code == 200
        assert "dataset_fingerprint" in evaluation.json()
        drift = client.get("/api/v1/drift")
        assert drift.status_code == 200
        assert drift.json()["model_degradation"]["confirmed"] is False
        assert client.get("/api/v1/forecasts", params={"service": service, "horizon": 999}).status_code == 422
    app.dependency_overrides.clear()
    test_engine.dispose()


def test_dataset_mismatch_degrades_health_without_replacing_artifact(tmp_path, monkeypatch):
    test_engine = _isolate_database(tmp_path, monkeypatch)
    data_path = tmp_path / "metrics.csv"
    generate_metrics(seed=24, periods=96).to_csv(data_path, index=False)
    artifact_path = tmp_path / "active.joblib"
    joblib.dump({"detector": None, "dataset_fingerprint": "stale-fingerprint"}, artifact_path)
    original_artifact = artifact_path.read_bytes()
    monkeypatch.setattr(settings, "data_path", data_path)
    monkeypatch.setattr(settings, "artifact_path", artifact_path)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "degraded"
        assert health.json()["model_ready"] is False
        assert client.get("/api/v1/services").status_code == 503

    assert artifact_path.read_bytes() == original_artifact
    app.dependency_overrides.clear()
    test_engine.dispose()
