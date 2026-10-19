import { ArrowUpRight, Check, CircleCheck, Clock3, X } from 'lucide-react'
import type { Alert, AlertStatus, Severity } from '../types'

const pretty = (value: string) => value.replaceAll('_', ' ')
const timeLabel = (value: string) => new Date(value).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

export function AlertPanel({ alerts, severity, status, selected, onSeverity, onStatus, onSelect, onUpdate }: {
  alerts: Alert[]; severity: Severity | 'all'; status: AlertStatus | 'all'; selected: Alert | null
  onSeverity: (value: Severity | 'all') => void; onStatus: (value: AlertStatus | 'all') => void
  onSelect: (value: Alert | null) => void; onUpdate: (alert: Alert, status: 'acknowledged' | 'resolved') => void
}) {
  const filtered = alerts.filter((alert) => (severity === 'all' || alert.severity === severity) && (status === 'all' || alert.status === status))
  return (
    <section className="panel alerts-panel" aria-labelledby="alerts-title">
      <div className="panel-heading alerts-heading"><div><p className="eyebrow">Triage queue <span className="count-pill">{filtered.length}</span></p><h2 id="alerts-title">Recent alerts</h2></div>
        <div className="filter-row"><label className="sr-only" htmlFor="severity-filter">Filter severity</label><select id="severity-filter" value={severity} onChange={(event) => onSeverity(event.target.value as Severity | 'all')}><option value="all">All severity</option><option value="critical">Critical</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select>
          <label className="sr-only" htmlFor="status-filter">Filter status</label><select id="status-filter" value={status} onChange={(event) => onStatus(event.target.value as AlertStatus | 'all')}><option value="all">All status</option><option value="open">Open</option><option value="acknowledged">Acknowledged</option><option value="resolved">Resolved</option></select></div>
      </div>
      <div className="alert-layout"><div className="alert-list" role="list" aria-label="Alerts">
        {filtered.length === 0 ? <div className="empty-state">No alerts match these filters.</div> : filtered.map((alert) => <button className={`alert-row ${selected?.id === alert.id ? 'selected' : ''}`} key={alert.id} onClick={() => onSelect(alert)} role="listitem">
          <span className={`severity-mark ${alert.severity}`} /><span className="alert-main"><span className="alert-service">{alert.service}<small>{pretty(alert.metric)}</small></span><span className="alert-time">{timeLabel(alert.timestamp)}</span></span>
          <span className={`status-tag ${alert.status}`}>{alert.status}</span><ArrowUpRight size={15} aria-hidden="true" />
        </button>)}
      </div>
        {selected ? <aside className="alert-detail" aria-label="Alert details"><div className="detail-top"><span className={`severity-label ${selected.severity}`}>{selected.severity}</span><button className="icon-button" onClick={() => onSelect(null)} aria-label="Close alert details"><X size={17} /></button></div>
          <h3>{selected.service} · {pretty(selected.metric)}</h3><p className="detail-time"><Clock3 size={13} /> {timeLabel(selected.timestamp)}</p>
          <div className="detail-values"><div><span>Observed</span><strong>{selected.observed_value.toFixed(2)}</strong></div><div><span>Baseline</span><strong>{selected.expected_value?.toFixed(2) ?? '—'}</strong></div><div><span>Model score</span><strong>{selected.anomaly_score?.toFixed(3) ?? '—'}</strong></div></div>
          <p className="detail-caption">Evidence · standardized deviation, not probability</p><ul className="evidence-list">{selected.evidence.map((item) => <li key={item.metric}><span>{pretty(item.metric)}</span><b>{item.standardized_deviation.toFixed(2)}σ</b></li>)}</ul>
          <p className="detail-caption">Suggested investigation</p><ul className="investigation-list">{selected.investigation_steps.map((step) => <li key={step}>{step}</li>)}</ul>
          <div className="detail-actions">{selected.status === 'open' && <button className="secondary-button" onClick={() => onUpdate(selected, 'acknowledged')}><Check size={15} /> Acknowledge</button>}{selected.status !== 'resolved' && <button className="resolve-button" onClick={() => onUpdate(selected, 'resolved')}><CircleCheck size={15} /> Resolve</button>}</div>
        </aside> : <div className="detail-placeholder"><span>SELECT AN ALERT</span><p>Choose a row to inspect evidence and investigation steps.</p></div>}
      </div>
    </section>
  )
}