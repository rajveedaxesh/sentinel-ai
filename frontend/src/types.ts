export const metricNames = ['cpu_utilization', 'memory_utilization', 'request_rate', 'api_latency_ms', 'error_rate', 'network_traffic_mbps'] as const
export type MetricName = (typeof metricNames)[number]
export type Severity = 'critical' | 'high' | 'medium' | 'low'
export type AlertStatus = 'open' | 'acknowledged' | 'resolved'

export interface ServiceSummary {
  name: string; last_seen: string; metrics: Record<MetricName, number>; anomaly_count: number; status: 'healthy' | 'watch' | 'critical'
}
export interface MetricPoint { timestamp: string; service: string; metric: MetricName; value: number; anomaly: boolean; anomaly_score: number | null }
export interface ForecastPoint { timestamp: string; metric: MetricName; predicted_value: number; lower_bound: number; upper_bound: number }
export interface AnomalyPoint { timestamp: string; service: string; anomaly_score: number; metrics: Record<MetricName, number> }
export interface Alert {
  id: string; service: string; timestamp: string; severity: Severity; metric: MetricName; observed_value: number; expected_value: number | null
  anomaly_score: number | null; evidence: { metric: string; standardized_deviation: number }[]; investigation_steps: string[]; status: AlertStatus; created_at: string; updated_at: string
}
export interface Evaluation {
  created_at: string; model_version: string; dataset_fingerprint: string
  payload: {
    seed: number; train_rows: number; test_rows: number; threshold: number; precision: number; recall: number; f1: number; average_precision: number
    false_alarms: number; confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
    baseline: { precision: number; recall: number; f1: number }; forecasting: Record<string, { mae: number; rmse: number; n: number }>
    drift: { distribution_drift_detected: boolean; model_degradation: { confirmed: boolean; assessment: string } }
  }
}