# SentinelAI Model Comparison

This report is generated from the tracked sample CSV by `scripts/compare_models.py`. The machine-readable metrics, thresholds, confusion matrices, label counts, and per-service false-positive diagnostics are in [model-comparison.json](model-comparison.json).

## Method

Dataset fingerprint: `6f4c9c75beca9acffa8fa87640e71b053d5507a210029db114ad931435bfe4db` (seed 42; 1,008 records, 336 hourly timestamps, three services).

The experiment uses global timestamp boundaries, so every service stays on the same side of each split:

| Partition | Records | Time window | Ground-truth anomalies |
| --- | ---: | --- | --- |
| Training | 603 | Oct 1 00:00 through Oct 9 08:00 UTC | CPU spikes and memory leaks |
| Validation | 201 | Oct 9 09:00 through Oct 12 03:00 UTC | Latency spikes only |
| Final test | 204 | Oct 12 04:00 through Oct 14 23:00 UTC | Request-rate drops and error-rate spikes only |

The detector and seasonal robust statistics fit only normal training rows. Candidate thresholds are selected against validation labels, maximizing validation F1, then precision, fewer false alarms, and average precision. The final test is not used for threshold or model selection. All listed test models and the three-sigma baseline use the exact same 204 observations and 18 labels. Average precision uses continuous anomaly scores; confusion matrices and precision/recall/F1 use thresholded flags. “False alarms” is the false-positive count.

The baseline is per-service, per-metric three-standard-deviation scoring, with mean and standard deviation fitted to normal training rows. It is not tuned on validation or test. The seasonal MAD comparison fits per-service/per-hour medians and median absolute deviations on normal training rows, and selects its single score threshold on validation. The seasonal-residual Isolation Forest uses signed per-service/per-hour robust residuals as features; its scaler and detector are fitted to normal training rows only.

## Final Test Results

| Model and validation choice | Precision | Recall | F1 | Average precision | False alarms | Confusion matrix (TN, FP, FN, TP) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Three-sigma baseline, fixed 3σ | 0.938 | 0.833 | **0.882** | 0.896 | **1** | 185, 1, 3, 15 |
| Isolation Forest, pooled, 200 trees | 0.500 | 0.444 | 0.471 | 0.581 | 8 | 178, 8, 10, 8 |
| Isolation Forest, pooled, 500 trees | 0.667 | 0.222 | 0.333 | 0.611 | 2 | 184, 2, 14, 4 |
| Isolation Forest, separate per service | 0.324 | 0.667 | 0.436 | 0.381 | 25 | 161, 25, 6, 12 |
| Isolation Forest on seasonal residual features | 0.313 | 0.278 | 0.294 | 0.319 | 11 | 175, 11, 13, 5 |
| Seasonal robust MAD, selected on validation | 0.833 | 0.833 | 0.833 | **0.944** | 3 | 183, 3, 3, 15 |

For pooled 200-tree Isolation Forest, contamination settings 0.01, 0.03, and 0.10 produce the same validation-calibrated test flags; contamination shifts the estimator score offset, while validation threshold selection compensates for that shift. The 500-tree variant reduces false alarms at the cost of missing most injected test anomalies.

The validation-selected seasonal model has the highest candidate validation F1 (1.000, 0 false alarms), but final test F1 falls to 0.833. It detects all nine error-rate spikes and six of nine request-rate drops, missing three request drops. Its validation period contains only latency spikes, so the result demonstrates a real scenario shift, not a guarantee of generalization. Seasonal residual features did not fix Isolation Forest: their validation F1 was 0.500 and final-test F1 0.294, with 11 false alarms and 13 missed anomalies.

## Existing 80/20 Evaluation

The prior production split uses 804 training and 204 final holdout rows. Re-evaluating its Isolation Forest and baseline on identical holdout rows, with normal-only baseline statistics, gives:

| Model | Precision | Recall | F1 | Average precision | False alarms | Confusion matrix (TN, FP, FN, TP) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Isolation Forest, seed 42, contamination 0.03, threshold 0 | 0.306 | 0.833 | 0.448 | 0.529 | 34 | 152, 34, 3, 15 |
| Three-sigma baseline, fixed 3σ | 0.938 | 0.833 | **0.882** | 0.919 | **1** | 185, 1, 3, 15 |

Of the 34 Isolation Forest false positives, 17 are checkout, 16 catalog, and one identity. The largest per-row univariate standardized deviation is CPU for 21 of them. Twenty-three false positives occur from 10:00 through 16:00 UTC, with six at 12:00. This clusters near the generator's daytime CPU/request cycle peak. These are labeled normal observations, although the pooled detector assigns them unusual multivariate scores; this is evidence consistent with a diurnal-pattern mismatch, not proof of causal attribution. The production holdout has overlapping score ranges: normal scores extend to 0.0418 while anomaly scores begin at -0.0210. A single zero threshold therefore flags many normal operating points.

The synthetic positives are direct, large, mostly single-metric injections. That makes a per-service threshold baseline naturally competitive. The pooled Isolation Forest sees raw service metrics with service/time encodings but does not model each service’s hour-specific normal distribution; ordinary CPU variation is a frequent outlier signal. This explains why simply lowering contamination is not a dependable fix: it trades false alarms for missed request drops, and calibration moves flags around rather than improving the score ranking.

## Promotion Decision

No candidate meets the existing acceptance criteria. The validation-selected seasonal MAD detector has lower untouched-test F1 than the three-sigma baseline (0.833 versus 0.882) and three versus one false alarms at the same recall. The pooled Isolation Forest is substantially worse. No candidate was promoted; the active artifact was not modified. The current active artifact still has a fingerprint from a different dataset version, so normal startup correctly fails closed rather than silently retraining or replacing it.

## Reproduction

```powershell
py -3.11 scripts\compare_models.py --output docs\model-comparison.json
py -3.11 -m pytest backend\tests\test_model_comparison.py -q
```

The experiment report is tied to the fingerprint above. Regenerate it after changing the CSV; do not compare candidates trained on different fingerprints or adjust any threshold after observing the final test.