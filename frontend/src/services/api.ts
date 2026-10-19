import type { Alert, AnomalyPoint, Evaluation, ForecastPoint, MetricName, MetricPoint, ServiceSummary } from '../types'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers: { 'Content-Type': 'application/json', ...init?.headers } })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(body?.detail ?? `Request failed (${response.status})`)
  }
  return response.json() as Promise<T>
}

export const api = {
  services: () => request<ServiceSummary[]>('/api/v1/services'),
  history: (service: string, metric: MetricName, limit: number) => request<MetricPoint[]>(`/api/v1/metrics/history?service=${encodeURIComponent(service)}&metric=${metric}&limit=${limit}`),
  forecasts: (service: string, metric: MetricName, horizon = 24) => request<ForecastPoint[]>(`/api/v1/forecasts?service=${encodeURIComponent(service)}&metric=${metric}&horizon=${horizon}`),
  anomalies: (service: string) => request<AnomalyPoint[]>(`/api/v1/anomalies?service=${encodeURIComponent(service)}&limit=100`),
  alerts: () => request<{ items: Alert[]; total: number }>('/api/v1/alerts?page_size=100'),
  evaluation: () => request<Evaluation>('/api/v1/evaluation'),
  updateAlert: (id: string, status: 'acknowledged' | 'resolved') => request<Alert>(`/api/v1/alerts/${id}`, { method: 'PATCH', body: JSON.stringify({ status }) }),
}