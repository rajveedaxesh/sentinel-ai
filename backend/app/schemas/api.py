from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ServiceSummary(BaseModel):
    name: str
    last_seen: datetime
    metrics: dict[str, float]
    anomaly_count: int
    status: Literal["healthy", "watch", "critical"]


class MetricPoint(BaseModel):
    timestamp: datetime
    service: str
    metric: str
    value: float
    anomaly: bool = False
    anomaly_score: float | None = None


class ForecastPoint(BaseModel):
    timestamp: datetime
    metric: str
    predicted_value: float
    lower_bound: float
    upper_bound: float


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    service: str
    timestamp: datetime
    severity: str
    metric: str
    observed_value: float
    expected_value: float | None
    anomaly_score: float | None
    evidence: list[dict]
    investigation_steps: list[str]
    status: str
    created_at: datetime
    updated_at: datetime


class AlertUpdate(BaseModel):
    status: Literal["acknowledged", "resolved"]


class AlertPage(BaseModel):
    items: list[AlertOut]
    total: int
    page: int
    page_size: int


class EvaluationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    created_at: datetime
    model_version: str
    dataset_fingerprint: str
    payload: dict


class DriftFeature(BaseModel):
    psi: float
    distribution_drift: bool


class DriftDegradation(BaseModel):
    confirmed: bool
    reference_f1: float | None
    current_f1: float | None
    assessment: str


class DriftOut(BaseModel):
    feature_distribution: dict[str, DriftFeature]
    distribution_drift_detected: bool
    model_degradation: DriftDegradation
    method: str


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    demo_data: bool = True
    model_ready: bool
    dataset_rows: int
    message: str
