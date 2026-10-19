import { useMemo } from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts'
import type { ForecastPoint, MetricName, MetricPoint } from '../types'

const labels: Record<MetricName, string> = {
  cpu_utilization: 'CPU utilization', memory_utilization: 'Memory utilization', request_rate: 'Request rate', api_latency_ms: 'API latency', error_rate: 'Error rate', network_traffic_mbps: 'Network traffic',
}
const units: Record<MetricName, string> = { cpu_utilization: '%', memory_utilization: '%', request_rate: 'req/s', api_latency_ms: 'ms', error_rate: '%', network_traffic_mbps: 'Mbps' }
const formatTime = (value: string) => new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' })

export function MetricChart({ history, forecast, metric, loading }: { history: MetricPoint[]; forecast: ForecastPoint[]; metric: MetricName; loading: boolean }) {
  const chartData = useMemo(() => [
    ...history.map((point) => ({ timestamp: point.timestamp, actual: point.value, anomaly: point.anomaly ? point.value : null, predicted: null as number | null, lower: null as number | null, upper: null as number | null })),
    ...forecast.map((point) => ({ timestamp: point.timestamp, actual: null as number | null, anomaly: null as number | null, predicted: point.predicted_value, lower: point.lower_bound, upper: point.upper_bound })),
  ], [history, forecast])
  return (
    <section className="panel chart-panel" aria-labelledby="trend-title">
      <div className="panel-heading chart-heading"><div><p className="eyebrow">Signal explorer</p><h2 id="trend-title">{labels[metric]}</h2></div><span className="chart-unit">{units[metric]}</span></div>
      <div className="chart-legend" aria-label="Chart legend"><span><i className="legend-dot actual" />Observed</span><span><i className="legend-dot forecast" />Forecast</span><span><i className="legend-dot anomaly" />Anomaly</span><span><i className="legend-band" />Prediction interval</span></div>
      <div className="chart-wrap">
        {loading ? <div className="chart-state">Loading metric history…</div> : chartData.length === 0 ? <div className="chart-state">No observations available.</div> : (
          <ResponsiveContainer width="100%" height="100%"><ComposedChart data={chartData} margin={{ top: 12, right: 8, left: -18, bottom: 4 }}>
            <defs><linearGradient id="intervalFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ef9d64" stopOpacity={0.16} /><stop offset="100%" stopColor="#ef9d64" stopOpacity={0.015} /></linearGradient></defs>
            <CartesianGrid stroke="#293236" strokeDasharray="3 5" vertical={false} />
            <XAxis dataKey="timestamp" tickFormatter={formatTime} minTickGap={42} tick={{ fill: '#869398', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis width={56} tick={{ fill: '#869398', fontSize: 11 }} axisLine={false} tickLine={false} />
            <Tooltip labelFormatter={(value) => new Date(String(value)).toLocaleString()} formatter={(value, name) => [typeof value === 'number' ? `${value.toFixed(2)} ${units[metric]}` : '—', String(name)]} contentStyle={{ background: '#1b2326', border: '1px solid #354247', borderRadius: 5, color: '#eef3f0' }} />
            {history.length > 0 && <ReferenceLine x={history.at(-1)?.timestamp} stroke="#637176" strokeDasharray="4 4" />}
            <Area dataKey="upper" stroke="none" fill="url(#intervalFill)" connectNulls={false} isAnimationActive={false} />
            <Area dataKey="lower" stroke="none" fill="#13191c" fillOpacity={0.92} connectNulls={false} isAnimationActive={false} />
            <Line dataKey="actual" name="Observed" stroke="#62c7ba" strokeWidth={2} dot={false} connectNulls={false} activeDot={{ r: 4, fill: '#62c7ba' }} />
            <Line dataKey="predicted" name="Forecast" stroke="#ef9d64" strokeWidth={2} strokeDasharray="5 4" dot={false} connectNulls={false} />
            <Scatter dataKey="anomaly" name="Anomaly" fill="#ed6a5a" shape="circle" />
          </ComposedChart></ResponsiveContainer>
        )}
      </div>
    </section>
  )
}