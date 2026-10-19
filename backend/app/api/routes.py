from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import AlertRecord, EvaluationRecord
from app.db.session import get_db
from app.schemas.api import AlertOut, AlertPage, AlertUpdate, DriftOut, EvaluationOut, ForecastPoint, HealthOut, MetricPoint, ServiceSummary

router = APIRouter()
MetricName = Literal["cpu_utilization", "memory_utilization", "request_rate", "api_latency_ms", "error_rate", "network_traffic_mbps"]


def monitoring(request: Request):
    service = getattr(request.app.state, "monitoring", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Monitoring model or dataset is unavailable")
    return service


@router.get("/health", response_model=HealthOut)
def health(request: Request):
    service = getattr(request.app.state, "monitoring", None)
    error = getattr(request.app.state, "startup_error", None)
    return HealthOut(status="ok" if service else "degraded", model_ready=service is not None,
                     dataset_rows=len(service.data) if service else 0,
                     message="Synthetic demo dataset; not live production telemetry." if service else f"Monitoring unavailable: {error or 'initialization failed'}")


@router.get("/api/v1/services", response_model=list[ServiceSummary])
def services(service=Depends(monitoring)):
    return service.services()


@router.get("/api/v1/metrics")
def current_metrics(service_name: str | None = Query(default=None, alias="service"), service=Depends(monitoring)):
    results = service.current_metrics(service_name)
    if service_name and not results:
        raise HTTPException(status_code=404, detail="Service not found")
    return results


@router.get("/api/v1/metrics/history", response_model=list[MetricPoint])
def metric_history(
    service_name: str = Query(alias="service", min_length=1, max_length=100),
    metric: MetricName = "cpu_utilization",
    limit: int = Query(default=168, ge=1, le=5000),
    service=Depends(monitoring),
):
    try:
        records = service.history(service=service_name, metric=metric, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if not records:
        raise HTTPException(status_code=404, detail="Service not found")
    return records


@router.get("/api/v1/anomalies")
def anomalies(service_name: str | None = Query(default=None, alias="service", max_length=100), limit: int = Query(default=100, ge=1, le=1000), service=Depends(monitoring)):
    if service_name and service_name not in {item["name"] for item in service.services()}:
        raise HTTPException(status_code=404, detail="Service not found")
    return service.anomalies(service=service_name, limit=limit)


@router.get("/api/v1/forecasts", response_model=list[ForecastPoint])
def forecasts(
    service_name: str = Query(alias="service", min_length=1, max_length=100),
    metric: MetricName = "cpu_utilization",
    horizon: int = Query(default=24, ge=1, le=168),
    service=Depends(monitoring),
):
    try:
        return service.forecasts(service=service_name, metric=metric, horizon=horizon)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/api/v1/alerts", response_model=AlertPage)
def alerts(
    severity: Literal["critical", "high", "medium", "low"] | None = None,
    status: Literal["open", "acknowledged", "resolved"] | None = None,
    service_name: str | None = Query(default=None, alias="service", max_length=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = select(AlertRecord)
    count_query = select(func.count()).select_from(AlertRecord)
    if severity:
        query = query.where(AlertRecord.severity == severity)
        count_query = count_query.where(AlertRecord.severity == severity)
    if status:
        query = query.where(AlertRecord.status == status)
        count_query = count_query.where(AlertRecord.status == status)
    if service_name:
        query = query.where(AlertRecord.service == service_name)
        count_query = count_query.where(AlertRecord.service == service_name)
    total = db.scalar(count_query) or 0
    items = db.scalars(query.order_by(AlertRecord.timestamp.desc()).offset((page - 1) * page_size).limit(page_size)).all()
    return AlertPage(items=items, total=total, page=page, page_size=page_size)


@router.patch("/api/v1/alerts/{alert_id}", response_model=AlertOut)
def update_alert(alert_id: str, update: AlertUpdate, db: Session = Depends(get_db)):
    alert = db.get(AlertRecord, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.status = update.status
    try:
        db.commit()
        db.refresh(alert)
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(status_code=503, detail="Could not persist alert update") from exc
    return alert


@router.get("/api/v1/evaluation", response_model=EvaluationOut)
def evaluation(db: Session = Depends(get_db)):
    record = db.scalar(select(EvaluationRecord).order_by(EvaluationRecord.created_at.desc()))
    if record is None:
        raise HTTPException(status_code=503, detail="No evaluation results are available")
    return record


@router.get("/api/v1/drift", response_model=DriftOut)
def drift(service=Depends(monitoring)):
    return service.evaluation["drift"]