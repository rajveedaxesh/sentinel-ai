from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

METRIC_COLUMNS = [
    "cpu_utilization",
    "memory_utilization",
    "request_rate",
    "api_latency_ms",
    "error_rate",
    "network_traffic_mbps",
]
LABEL_COLUMNS = ["is_anomaly", "anomaly_type"]
REQUIRED_COLUMNS = ["timestamp", "service", *METRIC_COLUMNS, *LABEL_COLUMNS]


def generate_metrics(
    *, seed: int = 42, periods: int = 24 * 14, frequency: str = "h"
) -> pd.DataFrame:
    """Create reproducible synthetic infrastructure metrics, never live telemetry."""
    if periods < 48:
        raise ValueError("periods must be at least 48 to include daily patterns")
    if periods > 24 * 90:
        raise ValueError("periods cannot exceed 90 days of hourly data")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-10-01", periods=periods, freq=frequency, tz="UTC")
    records: list[pd.DataFrame] = []
    profiles = {
        "checkout": (48.0, 56.0, 210.0, 170.0, 0.45, 82.0),
        "catalog": (34.0, 48.0, 320.0, 115.0, 0.22, 126.0),
        "identity": (27.0, 42.0, 145.0, 92.0, 0.18, 58.0),
    }

    for service_index, (service, profile) in enumerate(profiles.items()):
        phase = service_index * 0.7
        hour = timestamps.hour.to_numpy()
        daily = np.sin(2 * np.pi * (hour - 8) / 24 + phase)
        trend = np.linspace(0, 1, periods)
        cpu = profile[0] + 9 * daily + 5 * trend + rng.normal(0, 3, periods)
        memory = profile[1] + 2.5 * trend + 2 * np.sin(np.arange(periods) / 55) + rng.normal(0, 1.4, periods)
        requests = profile[2] + 55 * daily + 12 * trend + rng.normal(0, 13, periods)
        latency = profile[3] - 12 * daily + 8 * trend + rng.normal(0, 8, periods)
        errors = profile[4] + 0.12 * np.maximum(daily, 0) + rng.normal(0, 0.07, periods)
        network = profile[5] + profile[5] * 0.35 * daily + rng.normal(0, 5, periods)
        labels = np.zeros(periods, dtype=bool)
        anomaly_type = np.full(periods, "normal", dtype=object)
        windows = [
            (periods // 5 + service_index * 2, "cpu_spike"),
            (periods // 2 + service_index * 3, "memory_leak"),
            (periods * 3 // 4 + service_index * 2, "latency_spike"),
            (periods * 4 // 5 + service_index, "request_rate_drop"),
            (periods * 9 // 10 + service_index * 2, "error_rate_spike"),
        ]
        for start, kind in windows:
            end = min(start + max(3, periods // 180), periods)
            labels[start:end] = True
            anomaly_type[start:end] = kind
            if kind == "cpu_spike":
                cpu[start:end] += 35
            elif kind == "memory_leak":
                memory[start:end] += np.linspace(8, 28, end - start)
            elif kind == "latency_spike":
                latency[start:end] += 125
            elif kind == "request_rate_drop":
                requests[start:end] *= 0.28
            else:
                errors[start:end] += 2.8

        records.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "service": service,
                    "cpu_utilization": np.clip(cpu, 0, 100),
                    "memory_utilization": np.clip(memory, 0, 100),
                    "request_rate": np.clip(requests, 0, None),
                    "api_latency_ms": np.clip(latency, 1, None),
                    "error_rate": np.clip(errors, 0, 100),
                    "network_traffic_mbps": np.clip(network, 0, None),
                    "is_anomaly": labels,
                    "anomaly_type": anomaly_type,
                }
            )
        )

    return pd.concat(records, ignore_index=True).sort_values(["timestamp", "service"]).reset_index(drop=True)


def save_metrics(data: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path).resolve()
    approved_directory = Path(__file__).resolve().parents[3] / "data"
    if not destination.is_relative_to(approved_directory):
        raise ValueError("generated datasets must be saved within the project data directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(destination, index=False)
    return destination


def load_metrics(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])
