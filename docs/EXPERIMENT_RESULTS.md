# Synthetic experiment record

Status: **internal review draft — not an approved public performance claim**

## Frozen run

- Synthetic generator seed: `20260915`
- Calendar end: `2026-09-07`
- SKUs: 12
- History: 156 weekly observations per SKU
- Forecast horizon: 8 weeks
- Rolling-origin validation folds: 8
- Isolated test period: 2026-07-20 through 2026-09-07
- Models and feature definitions: repository state at Checkpoint F

No model saw the isolated test observations during fitting or selection.

## Validation results

Macro-average across SKU-level metrics:

| Model | MAE | WAPE | MASE | Bias |
|---|---:|---:|---:|---:|
| Seasonal Naive | 9.0938 | 0.4477 | 1.0221 | 0.3958 |
| ETS | 8.3194 | 0.4073 | 1.0029 | -0.1821 |
| Global HGB | 8.1522 | 0.4090 | 1.1904 | 1.4239 |

Per-SKU validation selection chose Global HGB for 6 SKUs, Seasonal Naive for 4,
and ETS for 2.

## Isolated test results

Aggregated across all 12 SKUs and eight test weeks:

| Approach | MAE | WAPE | Bias |
|---|---:|---:|---:|
| Seasonal Naive | 8.8646 | 0.2575 | 2.1146 |
| ETS | 14.0525 | 0.4082 | 9.7103 |
| Global HGB | 11.4629 | 0.3330 | 6.5527 |
| Validation-selected model per SKU | 13.1449 | 0.3818 | 8.3067 |

Positive bias means average over-forecasting.

## Interpretation

The transparent Seasonal Naive baseline performed best on this particular final
test window. The validation-selected per-SKU combination did not generalize better
than the baseline. This is an important negative result: a more complex forecasting
method is not automatically more accurate, and validation-based selection remains
uncertain when regimes or intermittent spikes shift.

The public Demo may show the verified metrics, but it must not claim that machine
learning improved accuracy. If the portfolio story requires such a claim, the
experiment design and synthetic scenario must be reviewed prospectively rather
than tuned against this held-out test period.

## Reproduction note

The live built-in scenario moves its calendar to the latest completed week to avoid
stale operational dates. Reproduce the frozen numbers by calling:

```python
generate_synthetic_data(end_date="2026-09-07")
```

