import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

afterEach(() => { cleanup(); vi.unstubAllGlobals() })
const response = (data: unknown) => ({ ok: true, json: async () => data })

describe('SentinelAI dashboard', () => {
  it('renders backend services and lets the operator change service metric', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.includes('/services')) return response([{ name: 'checkout', last_seen: '2025-01-01T00:00:00Z', metrics: { cpu_utilization: 42, memory_utilization: 51, request_rate: 180, api_latency_ms: 120, error_rate: 0.2, network_traffic_mbps: 72 }, anomaly_count: 1, status: 'watch' }])
      if (url.includes('/alerts')) return response({ items: [], total: 0 })
      if (url.includes('/evaluation')) return response({ created_at: '2025-01-01', model_version: 'isolation-forest-v1', dataset_fingerprint: 'abc123', payload: { seed: 42, train_rows: 500, test_rows: 100, threshold: 0.1, precision: 0.7, recall: 0.8, f1: 0.74, average_precision: 0.8, false_alarms: 3, confusion_matrix: { tn: 80, fp: 3, fn: 2, tp: 15 }, baseline: { precision: 0.4, recall: 0.3, f1: 0.34 }, forecasting: {}, drift: { distribution_drift_detected: true, model_degradation: { confirmed: false, assessment: 'not assessed without a reference and labeled holdout' } } } })
      if (url.includes('/history')) return response([{ timestamp: '2025-01-01T00:00:00Z', service: 'checkout', metric: 'cpu_utilization', value: 42, anomaly: false, anomaly_score: null }])
      if (url.includes('/forecasts') || url.includes('/anomalies')) return response([])
      return response([])
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)
    expect(await screen.findByRole('heading', { name: 'System overview' })).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /checkout/ })).toBeInTheDocument()
    expect(screen.getByText('42.0%')).toBeInTheDocument()
    expect(screen.getByText('Review signal detected')).toBeInTheDocument()
    expect(screen.getByText('not assessed without a reference and labeled holdout')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Memory utilization/ }))
    expect(screen.getByRole('heading', { name: 'Memory utilization' })).toBeInTheDocument()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('metric=memory_utilization'), expect.anything()))
    const historyCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('/history')).length
    await userEvent.click(screen.getByRole('button', { name: 'Refresh dashboard' }))
    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => String(url).includes('/history')).length).toBeGreaterThan(historyCalls))
    expect(screen.getByText(/SIMULATED DATA/)).toBeInTheDocument()
  })
})