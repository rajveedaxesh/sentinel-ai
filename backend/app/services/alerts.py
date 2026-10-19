from __future__ import annotations

import uuid

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AlertRecord
from app.ml.data import METRIC_COLUMNS

THRESHOLDS = {
    "cpu_utilization": (70.0, 80.0, 92.0),
    "memory_utilization": (75.0, 82.0, 94.0),
    "api_latency_ms": (200.0, 250.0, 500.0),
    "error_rate": (0.8, 1.0, 3.0),
}
INVESTIGATION = {
    "cpu_utilization": ["Inspect recent deployments and CPU-heavy requests.", "Check worker saturation and autoscaling events."],
    "memory_utilization": ["Inspect heap, cache, and process memory trends.", "Check for retained objects or a memory leak."],
    "api_latency_ms": ["Inspect latency by route and dependency.", "Check database and upstream service response times."],
    "error_rate": ["Inspect recent error logs and affected routes.", "Check dependency health and retry rates."],
    "request_rate": ["Check traffic routing, upstream health, and demand changes."],
    "network_traffic_mbps": ["Inspect network saturation, retransmits, and recent traffic changes."],
}


def severity_for(metric: str, value: float, model_anomaly: bool) -> str | None:
    limits = THRESHOLDS.get(metric)
    if limits and value >= limits[2]:
        return "critical"
    if limits and value >= limits[1]:
        return "high"
    if limits and value >= limits[0]:
        return "low"
    if model_anomaly:
        return "medium"
    return None


def _timestamp_key(value) -> str:
    timestamp = pd.Timestamp(value)
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def create_alert_records(scored: pd.DataFrame, baseline_data: pd.DataFrame, session: Session, detector, *, persist: bool = True) -> list[AlertRecord]:
    existing_keys = {
        (service, _timestamp_key(timestamp), metric)
        for service, timestamp, metric in session.execute(select(AlertRecord.service, AlertRecord.timestamp, AlertRecord.metric)).all()
    }
    baselines = baseline_data.groupby("service")[METRIC_COLUMNS].median()
    generated: list[AlertRecord] = []
    for service, records in scored.groupby("service", sort=False):
        baseline = baselines.loc[service]
        for _, row in records.iterrows():
            contributions = detector.metric_contributions(row, baseline)
            changed = []
            for metric in METRIC_COLUMNS:
                value = float(row[metric])
                rule_triggered = metric in THRESHOLDS and value >= THRESHOLDS[metric][0]
                if rule_triggered:
                    changed.append((metric, severity_for(metric, value, bool(row["is_predicted_anomaly"]))))
            if bool(row["is_predicted_anomaly"]):
                dominant_metric = str(contributions[0]["metric"])
                changed.append((dominant_metric, severity_for(dominant_metric, float(row[dominant_metric]), True)))
            if not changed:
                continue
            metric, severity = max(
                changed,
                key=lambda candidate: (
                    ["low", "medium", "high", "critical"].index(candidate[1] or "low"),
                    next(float(item["standardized_deviation"]) for item in contributions if item["metric"] == candidate[0]),
                ),
            )
            timestamp = pd.Timestamp(row["timestamp"]).to_pydatetime()
            key = (service, _timestamp_key(timestamp), metric)
            if key in existing_keys:
                continue
            generated.append(AlertRecord(
                id=str(uuid.uuid4()), service=service, timestamp=timestamp, severity=severity or "low",
                metric=metric, observed_value=float(row[metric]), expected_value=float(baseline[metric]),
                anomaly_score=float(row["anomaly_score"]),
                evidence=[{"metric": item["metric"], "standardized_deviation": round(float(item["standardized_deviation"]), 3)} for item in contributions[:3]],
                investigation_steps=INVESTIGATION.get(metric, ["Compare this observation with recent service changes."]), status="open",
            ))
            existing_keys.add(key)
    if persist:
        session.add_all(generated)
        session.commit()
    return generated