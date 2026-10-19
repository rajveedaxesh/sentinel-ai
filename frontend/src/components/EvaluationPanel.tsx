import { Activity, Crosshair, Fingerprint, Target } from 'lucide-react'
import type { Evaluation } from '../types'

const percent = (value: number) => `${(value * 100).toFixed(1)}%`

export function EvaluationPanel({ evaluation }: { evaluation: Evaluation | null }) {
  if (!evaluation) return <section className="panel eval-panel"><div className="empty-state">Evaluation results are unavailable.</div></section>
  const { payload } = evaluation
  const metrics = [{ label: 'Precision', value: percent(payload.precision), icon: Target }, { label: 'Recall', value: percent(payload.recall), icon: Crosshair }, { label: 'F1 score', value: percent(payload.f1), icon: Activity }, { label: 'Avg. precision', value: percent(payload.average_precision), icon: Fingerprint }]
  return <section className="panel eval-panel" aria-labelledby="evaluation-title">
    <div className="panel-heading"><div><p className="eyebrow">Chronological holdout</p><h2 id="evaluation-title">Model evaluation</h2></div><span className="model-version">{evaluation.model_version}</span></div>
    <div className="evaluation-grid">{metrics.map(({ label, value, icon: Icon }) => <div className="evaluation-metric" key={label}><Icon size={15} /><span>{label}</span><strong>{value}</strong></div>)}</div>
    <div className="evaluation-foot"><span>{payload.train_rows.toLocaleString()} train · {payload.test_rows.toLocaleString()} holdout</span><span>{payload.false_alarms} false alarms</span><span>Baseline F1 {percent(payload.baseline.f1)}</span></div>
    <div className="matrix-row" aria-label="Confusion matrix"><span>Confusion matrix</span><b>TN {payload.confusion_matrix.tn}</b><b>FP {payload.confusion_matrix.fp}</b><b>FN {payload.confusion_matrix.fn}</b><b>TP {payload.confusion_matrix.tp}</b></div>
    <div className="forecast-errors">{Object.entries(payload.forecasting).map(([metric, result]) => <div key={metric}><span>{metric.replaceAll('_', ' ')} holdout</span><b>MAE {result.mae.toFixed(2)} · RMSE {result.rmse.toFixed(2)}</b></div>)}</div>
    <div className="drift-row"><span>Feature drift</span><b className={payload.drift.distribution_drift_detected ? 'drift-detected' : ''}>{payload.drift.distribution_drift_detected ? 'Review signal detected' : 'No review signal'}</b><span>Model degradation</span><b>{payload.drift.model_degradation.assessment}</b></div>
    <p className="fingerprint">Dataset {evaluation.dataset_fingerprint.slice(0, 16)}… · seed {payload.seed}</p>
  </section>
}