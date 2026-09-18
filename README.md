# Demand Forecasting & Inventory Decision Support

A local, reproducible multi-SKU forecasting and inventory decision-support Demo.
It uses synthetic data by default and shows how forecasts become transparent
stock-risk and replenishment signals. It is a portfolio demonstration, not a
production planning system or inventory optimizer.

## Verified example

The bundled synthetic scenario contains 12 SKUs with 156 weekly observations per
SKU. Models are compared through chronological rolling-origin validation, with a
final eight-week period isolated for evaluation. Seasonal Naive performs best on
aggregate WAPE for the current frozen test; that result is retained rather than
tuned away.

## Run locally

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m streamlit run ui/app.py
```

Then open the local URL printed by Streamlit. To configure the navigation link:

```powershell
$env:DEMO_CASE_STUDY_URL = "https://example.com/work/demand-forecasting"
```

The configured value must be an `http` or `https` URL. Invalid values fall back
to the local case-study page.

## User workflow

1. Run the built-in synthetic scenario immediately or upload a weekly CSV.
2. Triage SKUs by stockout risk and select a row for detailed analysis.
3. Compare the selected forecast with history and empirical uncertainty ranges.
4. Inspect chronological validation metrics and demand diagnostics.
5. Vary lead time, target service level, or annual demand growth.
6. Review the reproducible reorder point, timing, quantity, and assumptions.

Uploaded CSV content is processed in memory in the running Python process. Its
forecast result is cached only in the current Streamlit session, cleared when the
upload is removed or the built-in scenario is selected, and is not placed in the
cross-user Streamlit data cache. The application does not write uploaded content
to disk or intentionally transmit it to another service.

## CSV contract

Exactly five columns are accepted:

| Column | Rule |
|---|---|
| `date` | ISO date for a Monday; consecutive weekly calendar |
| `sku_id` | Non-empty identifier |
| `demand` | Finite, non-negative integer |
| `current_on_hand` | Finite, non-negative integer; constant within each SKU |
| `lead_time_weeks` | Integer from 1 through 12; constant within each SKU |

All SKUs must share one calendar. Each SKU needs 80–260 observations, with no
more than 100 SKUs per upload. Missing weeks are rejected rather than silently
interpreted as zero demand.

## Forecasting design

- Weekly data, eight-week public forecast, and 13-week internal planning forecast.
- Seasonal Naive baseline using the observation from 52 weeks earlier.
- Additive damped ETS with fixed, documented smoothing parameters.
- Global histogram gradient boosting using lag 1/2/4/8/13/26/52, shifted rolling
  mean and standard deviation, calendar seasonality, trend, and SKU indicators.
- Up to eight rolling-origin validation folds followed by an untouched final
  eight-week test. An 80-week minimum history uses two folds so lag features still
  have at least 52 prior observations.
- Per-SKU selection by validation MASE, with a 2% simplicity tie-break.
- MAE, WAPE, MASE, and signed forecast bias. MAPE is deliberately omitted because
  zeros and intermittent demand make it unstable.
- Model-agnostic 80% and 95% forecast ranges from validation errors. These are
  empirical uncertainty summaries, not probabilistic guarantees.

Every lag and rolling feature is built from shifted history. Encoders, model fits,
and error calibration use only information available before each forecast origin.

## Inventory decision logic

The first version assumes weekly review, no open purchase orders, and inventory
position equal to current on-hand inventory.

```text
lead-time demand = sum of scenario-adjusted forecasts over L weeks
safety stock = max(empirical cumulative-error quantile, z × cumulative-error σ)
reorder point = lead-time demand + safety stock
order-up-to level = demand over L + 1 review week + safety stock
recommended quantity = ceil(max(0, order-up-to level - projected inventory position))
```

Overlapping cumulative-error windows provide 40 observations for the default
four-week lead time in the built-in scenario. The interface shows the observation
count and warns below 20. Lead times beyond the eight-week validation horizon use
a labeled circular-block approximation.

“Overstock risk” means inventory exceeds the near-term order-up-to level. It is not
an economic optimum because holding costs, shelf life, order minimums, supplier
constraints, and service penalties are not modeled.

## Project structure

```text
src/demand_forecasting/  validation, synthetic data, models, evaluation, inventory
ui/app.py                local Streamlit application
ui/app.css               portfolio-aligned presentation layer
ui/wireframe.*           approved Checkpoint D review artifact
tests/                   leakage, reproducibility, validation, and formula tests
docs/                    experiment record and website integration contract
```

Primary Python entry points:

- `demand_forecasting.service.run_forecasting_core`
- `demand_forecasting.inventory.build_inventory_decisions`

## Verification

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

See `docs/EXPERIMENT_RESULTS.md` for the frozen synthetic evaluation. No accuracy
improvement claim should be published from those results without separate review.

## Non-goals and publication status

This version has no ERP integration, persistent storage, real-time data, deep
learning, purchase-order submission, or mathematical inventory optimizer. It
contains no client data or client code.

The Demo has no public deployment yet. Streamlit Community Cloud is the proposed
host, subject to separate staging and publication approval. The code is licensed
separately under the MIT License below.

## Streamlit Community Cloud staging configuration

The root `requirements.txt` installs this package and its dependencies from
`pyproject.toml`. Use `ui/app.py` as the Community Cloud entrypoint and select
Python 3.12 in the deployment settings. The root `.streamlit/config.toml` is in
the location Community Cloud expects and limits individual uploads to 10 MB.

Set the approved website return URL as `DEMO_CASE_STUDY_URL` in the hosted
environment after a portfolio staging URL exists. A private repository, staging
deployment, and final public URL still require separate approval.

## License

This project is available under the [MIT License](LICENSE).
