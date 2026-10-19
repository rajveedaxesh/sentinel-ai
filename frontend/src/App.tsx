import { useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, Bell, ChevronDown, CircleHelp, Clock3, Cpu, Gauge, Layers3, RefreshCw, Server, ShieldCheck, Signal, Wifi } from 'lucide-react'
import { AlertPanel } from './components/AlertPanel'
import { EvaluationPanel } from './components/EvaluationPanel'
import { MetricChart } from './components/MetricChart'
import { api } from './services/api'
import type { Alert, AlertStatus, AnomalyPoint, Evaluation, ForecastPoint, MetricName, MetricPoint, ServiceSummary, Severity } from './types'

const metrics: { key: MetricName; label: string; unit: string; icon: typeof Cpu }[] = [
  { key: 'cpu_utilization', label: 'CPU utilization', unit: '%', icon: Cpu }, { key: 'memory_utilization', label: 'Memory utilization', unit: '%', icon: Layers3 },
  { key: 'request_rate', label: 'Request rate', unit: 'req/s', icon: Signal }, { key: 'api_latency_ms', label: 'API latency', unit: 'ms', icon: Activity },
  { key: 'error_rate', label: 'Error rate', unit: '%', icon: AlertTriangle }, { key: 'network_traffic_mbps', label: 'Network traffic', unit: 'Mbps', icon: Wifi },
]
const formatValue = (value: number, unit: string) => `${value.toFixed(unit === '%' ? 1 : 0)}${unit === '%' ? '%' : ` ${unit}`}`
const relativeTime = (date: string) => {
  const minutes = Math.max(0, Math.round((Date.now() - new Date(date).getTime()) / 60000))
  return minutes < 60 ? `${minutes}m ago` : minutes < 1440 ? `${Math.floor(minutes / 60)}h ago` : `${Math.floor(minutes / 1440)}d ago`
}

function App() {
  const [services, setServices] = useState<ServiceSummary[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null)
  const [history, setHistory] = useState<MetricPoint[]>([])
  const [forecast, setForecast] = useState<ForecastPoint[]>([])
  const [anomalies, setAnomalies] = useState<AnomalyPoint[]>([])
  const [selectedService, setSelectedService] = useState('')
  const [selectedMetric, setSelectedMetric] = useState<MetricName>('cpu_utilization')
  const [historyWindow, setHistoryWindow] = useState(168)
  const [refreshKey, setRefreshKey] = useState(0)
  const [severityFilter, setSeverityFilter] = useState<Severity | 'all'>('all')
  const [statusFilter, setStatusFilter] = useState<AlertStatus | 'all'>('all')
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null)
  const [loading, setLoading] = useState(true)
  const [chartLoading, setChartLoading] = useState(false)
  const [error, setError] = useState('')
  const [refreshing, setRefreshing] = useState(false)

  async function loadOverview() {
    setError('')
    try {
      const [nextServices, alertData, nextEvaluation] = await Promise.all([api.services(), api.alerts(), api.evaluation()])
      setServices(nextServices); setAlerts(alertData.items); setEvaluation(nextEvaluation)
      setSelectedService((current) => current || nextServices[0]?.name || '')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not connect to SentinelAI API.')
    } finally { setLoading(false); setRefreshing(false) }
  }
  useEffect(() => { void loadOverview() }, [])

  useEffect(() => {
    if (!selectedService) return
    let active = true
    setChartLoading(true)
    Promise.all([api.history(selectedService, selectedMetric, historyWindow), api.forecasts(selectedService, selectedMetric, 24), api.anomalies(selectedService)])
      .then(([nextHistory, nextForecast, nextAnomalies]) => {
        if (!active) return
        setHistory(nextHistory); setForecast(nextForecast); setAnomalies(nextAnomalies)
      }).catch((reason: unknown) => { if (active) setError(reason instanceof Error ? reason.message : 'Could not load selected service data.') })
      .finally(() => { if (active) setChartLoading(false) })
    return () => { active = false }
  }, [selectedService, selectedMetric, historyWindow, refreshKey])

  const activeService = services.find((service) => service.name === selectedService)
  const openCount = alerts.filter((alert) => alert.status === 'open').length
  const criticalCount = alerts.filter((alert) => alert.status !== 'resolved' && (alert.severity === 'critical' || alert.severity === 'high')).length
  const scoreBars = useMemo(() => anomalies.slice(0, 5), [anomalies])
  const maxAnomalyScore = Math.max(...scoreBars.map((item) => item.anomaly_score), 0.0001)

  async function updateAlert(alert: Alert, status: 'acknowledged' | 'resolved') {
    try {
      const updated = await api.updateAlert(alert.id, status)
      setAlerts((current) => current.map((item) => item.id === updated.id ? updated : item)); setSelectedAlert(updated)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not update this alert.') }
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <a className="brand" href="#overview" aria-label="SentinelAI overview"><span className="brand-mark"><Activity size={19} strokeWidth={2.3} /></span><span>sentinel<span className="brand-ai">ai</span><small>OPERATIONS INTELLIGENCE</small></span></a>
      <div className="sidebar-section-label">WORKSPACE</div>
      <a className="nav-link active" href="#overview"><Gauge size={17} /> Overview <span className="nav-indicator" /></a>
      <a className="nav-link" href="#alerts"><Bell size={17} /> Alert queue {openCount > 0 && <span className="nav-count">{openCount}</span>}</a>
      <a className="nav-link" href="#evaluation"><Activity size={17} /> Model health</a>
      <div className="sidebar-bottom"><div className="sidebar-status"><span className="live-dot" /><span>DEMO ENVIRONMENT</span></div><p>Metrics are generated from a reproducible synthetic scenario. No live infrastructure is connected.</p><div className="sidebar-build"><span>MODEL</span><b>{evaluation?.model_version ?? 'Loading'}</b></div></div>
    </aside>

    <main className="main-content" id="overview">
      <header className="topbar"><div className="breadcrumb"><span>Workspace</span><b>/</b><strong>Operations overview</strong></div><div className="topbar-actions">
        <span className="demo-chip"><span className="live-dot" /> SIMULATED DATA</span>
        <button className="refresh-button" onClick={() => { setRefreshing(true); setRefreshKey((current) => current + 1); void loadOverview() }} disabled={refreshing} aria-label="Refresh dashboard"><RefreshCw size={15} className={refreshing ? 'spin' : ''} /><span>Refresh</span></button>
        <button className="help-button" aria-label="About SentinelAI" title="Synthetic monitoring demonstration"><CircleHelp size={18} /></button>
      </div></header>

      <div className="page-content">
        <section className="page-heading"><div><p className="eyebrow">PREDICTIVE OPERATIONS / DEMO</p><h1>System overview</h1><p className="page-subtitle">Behavioral signals across your monitored services.</p></div><div className="data-time"><Clock3 size={14} /><span>Latest observation</span><b>{activeService ? relativeTime(activeService.last_seen) : '—'}</b></div></section>
        {error && <div className="error-banner" role="alert"><AlertTriangle size={16} /><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss error">×</button></div>}
        {loading ? <div className="loading-state"><span className="loader" />Connecting to monitoring data…</div> : services.length === 0 ? <div className="empty-dashboard"><Server size={26} /><h2>No monitoring data</h2><p>Start the backend demo workflow to generate the sample dataset and model outputs.</p></div> : <>
          <section className="service-strip" aria-label="Monitored services">{services.map((service) => <button className={`service-tile ${selectedService === service.name ? 'active' : ''}`} key={service.name} onClick={() => setSelectedService(service.name)}>
            <span className={`service-state ${service.status}`} /><span className="service-name">{service.name}</span><span className="service-meta">{service.status === 'healthy' ? 'Nominal' : service.status === 'watch' ? 'Under observation' : 'Needs attention'}</span><span className="service-anomalies">{service.anomaly_count} <small>signals</small></span>
          </button>)}<div className="service-uptime"><ShieldCheck size={17} /><span>MODEL SCAN</span><b>Chronological holdout</b></div></section>

          <section className="metric-grid" aria-label="Current service metrics">{metrics.map(({ key, label, unit, icon: Icon }, index) => {
            const value = activeService?.metrics[key] ?? 0
            return <button className={`metric-card ${selectedMetric === key ? 'chosen' : ''}`} key={key} onClick={() => setSelectedMetric(key)} aria-pressed={selectedMetric === key}>
              <span className={`metric-icon icon-${index}`}><Icon size={16} /></span><span className="metric-label">{label}</span><strong>{formatValue(value, unit)}</strong><span className="metric-change"><i className="sample-dot" /><span>latest sample</span></span>
            </button>
          })}</section>

          <div className="section-tools"><div className="section-title"><span className="section-index">01</span><span>Signal &amp; forecast</span></div><div className="chart-controls">
            <label htmlFor="service-select">Service</label><div className="select-wrap"><select id="service-select" value={selectedService} onChange={(event) => setSelectedService(event.target.value)}>{services.map((service) => <option key={service.name}>{service.name}</option>)}</select><ChevronDown size={14} /></div>
            <label htmlFor="window-select">History</label><div className="select-wrap"><select id="window-select" value={historyWindow} onChange={(event) => setHistoryWindow(Number(event.target.value))}><option value={72}>72 hours</option><option value={168}>7 days</option><option value={336}>14 days</option></select><ChevronDown size={14} /></div>
          </div></div>
          <div className="analysis-grid"><MetricChart history={history} forecast={forecast} metric={selectedMetric} loading={chartLoading} />
            <section className="panel score-panel" aria-labelledby="score-title"><div className="panel-heading"><div><p className="eyebrow">Detector output</p><h2 id="score-title">Anomaly scores</h2></div><span className="score-info" title="Scores are uncalibrated model outputs, not probabilities">i</span></div>
              <p className="score-description">Out-of-time observations · higher values indicate stronger deviation.</p>
              {scoreBars.length === 0 ? <div className="empty-state">No holdout anomalies for this service.</div> : <div className="score-list">{scoreBars.map((item, index) => {
                const relative = Math.max(8, Math.min(100, (item.anomaly_score / maxAnomalyScore) * 100))
                return <div className="score-item" key={`${item.timestamp}-${index}`}><div className="score-item-top"><span>{new Date(item.timestamp).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit' })}</span><b>{item.anomaly_score.toFixed(3)}</b></div><div className="score-track"><i style={{ width: `${relative}%` }} /></div></div>
              })}</div>}
              <div className="score-foot"><span>DETECTOR</span><b>Isolation Forest</b><span>Threshold</span><b>{evaluation?.payload.threshold.toFixed(3) ?? '—'}</b></div>
            </section>
          </div>

          <div className="section-tools alert-section-heading" id="alerts"><div className="section-title"><span className="section-index">02</span><span>Alert intelligence</span>{criticalCount > 0 && <span className="critical-summary"><i />{criticalCount} elevated</span>}</div><span className="section-note">Derived from model outputs and metric rules</span></div>
          <AlertPanel alerts={alerts} severity={severityFilter} status={statusFilter} selected={selectedAlert} onSeverity={setSeverityFilter} onStatus={setStatusFilter} onSelect={setSelectedAlert} onUpdate={(alert, status) => void updateAlert(alert, status)} />
          <div className="section-tools evaluation-heading" id="evaluation"><div className="section-title"><span className="section-index">03</span><span>Model validation</span></div><span className="section-note">Measured on held-out future observations</span></div>
          <EvaluationPanel evaluation={evaluation} />
        </>}
        <footer className="page-footer"><span>SentinelAI · Predictive Operations</span><span>Demo data only · Scores are not probabilities</span></footer>
      </div>
    </main>
  </div>
}

export default App