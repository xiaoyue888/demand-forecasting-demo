# Website integration contract

Status: **deployed, connected, and verified**

## Portfolio metadata

- Title: **Demand Forecasting & Inventory Decision Support**
- Primary category: **AI & Machine Learning**
- Secondary tags: Demand Forecasting, Time Series, Inventory Planning,
  Explainable ML, Supply Chain
- One-line summary: **Forecast weekly multi-SKU demand, quantify uncertainty, and
  translate predictions into transparent inventory-risk and replenishment signals.**
- Core flow: Historical sales → Demand forecast → Inventory risk → Replenishment
  recommendation

Do not publish an accuracy-improvement percentage. The frozen test shows the
Seasonal Naive baseline outperforming the more complex candidates on aggregate
WAPE for that specific synthetic test period.

## Runtime boundary

The Demo is a stateful Streamlit application backed by a Python process. It remains
separate from the static Portfolio deployment and is linked rather than embedded.
Uploaded CSV content is processed in memory, limited by the public data contract,
and never placed in the cross-user Streamlit cache.

The local case-study fallback is
`http://127.0.0.1:4173/demand-forecasting.html`. The hosted runtime sets
`DEMO_CASE_STUDY_URL` to the public Portfolio case page.

## Public integration

- Live Demo: `https://xiaoyue-demand-forecasting.streamlit.app/`
- Portfolio case study: `https://xiaoyue-portfolio.pages.dev/demand-forecasting.html`
- The Portfolio opens the hosted Demo as an external application.
- The hosted Demo opens the matching Portfolio case page at the top level.
- Both public endpoints and the two-way navigation have been verified.

The selected Streamlit Community Cloud service may sleep when idle. That hosting
behavior does not change the model, data, or privacy boundary.

## Visual handoff

The application carries:

- XZ navigation mark and Back to case study link.
- Warm off-white `#fafaf7` background.
- Light-blue `#4b9ee8` primary accent.
- Jost headings, Inter body text, and JetBrains Mono technical labels.
- Thin rules, square controls, and no decorative imagery.
- Minimal Streamlit toolbar configuration.

No raster assets are required. Forecast charts are generated at runtime.

## Publication boundary

The repository contains only synthetic data and newly written portfolio code.
Do not add client data, logos, endorsement claims, or unsupported performance
language. GitHub repository visibility and website source visibility remain
separate decisions from the already approved public application deployment.
