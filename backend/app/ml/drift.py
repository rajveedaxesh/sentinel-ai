from __future__ import annotations

import numpy as np
import pandas as pd

from .data import METRIC_COLUMNS


def _population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    boundaries = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(boundaries) < 3:
        return 0.0
    boundaries[0] = -np.inf
    boundaries[-1] = np.inf
    reference_counts = np.histogram(reference, bins=boundaries)[0].astype(float)
    current_counts = np.histogram(current, bins=boundaries)[0].astype(float)
    reference_share = np.clip(reference_counts / max(reference_counts.sum(), 1), 1e-6, 1)
    current_share = np.clip(current_counts / max(current_counts.sum(), 1), 1e-6, 1)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    labeled_metrics: dict | None = None,
    reference_f1: float | None = None,
    degradation_tolerance: float = 0.1,
) -> dict:
    """Report feature distribution shifts separately from measured labeled degradation."""
    feature_reports = {}
    for metric in METRIC_COLUMNS:
        psi = _population_stability_index(reference[metric].dropna().to_numpy(), current[metric].dropna().to_numpy())
        feature_reports[metric] = {"psi": psi, "distribution_drift": psi >= 0.2}
    measured_f1 = labeled_metrics.get("f1") if labeled_metrics else None
    degradation_confirmed = (
        reference_f1 is not None and measured_f1 is not None and reference_f1 - float(measured_f1) > degradation_tolerance
    )
    return {
        "feature_distribution": feature_reports,
        "distribution_drift_detected": any(item["distribution_drift"] for item in feature_reports.values()),
        "model_degradation": {
            "confirmed": bool(degradation_confirmed),
            "reference_f1": reference_f1,
            "current_f1": float(measured_f1) if measured_f1 is not None else None,
            "assessment": "confirmed from labeled holdout metrics" if degradation_confirmed else "not assessed without a reference and labeled holdout",
        },
        "method": "Population Stability Index; PSI >= 0.2 is a review signal, not proof of model degradation.",
    }