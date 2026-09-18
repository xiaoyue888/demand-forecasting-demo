# Website integration contract

Status: **connected to the local portfolio prototype; public hosting remains unapproved**

## Portfolio metadata

- Title: **Demand Forecasting & Inventory Decision Support**
- Primary category: **AI & Machine Learning**
- Secondary tags: Demand Forecasting, Time Series, Inventory Planning,
  Explainable ML, Supply Chain
- One-line summary: **Forecast weekly multi-SKU demand, quantify uncertainty, and
  translate predictions into transparent inventory-risk and replenishment signals.**
- Core flow: Historical sales → Demand forecast → Inventory risk → Replenishment
  recommendation

Do not publish an accuracy-improvement percentage. The frozen test currently shows
the Seasonal Naive baseline outperforming the more complex candidates.

## Runtime boundary

The Demo is currently a stateful Streamlit application backed by a Python process.
It is not a static bundle and cannot be copied into the website's existing `dist/`
directory. Public integration needs a separately approved Python-capable hosting
decision or a separately scoped browser-side rewrite.

The website should link to the approved Demo URL rather than iframe it until the
chosen host's embedding, cookies, loading, and security behavior are verified.

Set `DEMO_CASE_STUDY_URL` in the Demo runtime to the final website case-page URL.
The local default is `http://127.0.0.1:4173/demand-forecasting.html`.

## Visual handoff

The local application already carries:

- XZ navigation mark and Back to case study link.
- Warm off-white `#fafaf7` background.
- Light-blue `#4b9ee8` primary accent.
- Jost headings, Inter body text, and JetBrains Mono technical labels.
- Thin rules, square controls, and no decorative imagery.
- Minimal Streamlit toolbar configuration.

No raster assets are required. Forecast charts are generated at runtime.

## Local integration completed

- Added the project card and case page using the approved metadata.
- Linked the case page to the local Demo at `http://127.0.0.1:8767/`.
- Configured the Demo's default back-link to the local case page.

## Required work before public release

1. Link the case page to the final hosted Demo URL.
2. Configure the hosted Demo's `DEMO_CASE_STUDY_URL` back-link.
3. Add a visible loading expectation if the selected Python host sleeps when idle.
4. Verify desktop/mobile navigation and accessibility in the hosted context.
5. Review the experiment record before exposing any metric or performance copy.

Do not copy client data, add logos, claim endorsement, or replace the disclosed
planning assumptions with marketing language.

## Deployment decision still required

Before publication, Xiaoyue must choose and approve one of these scopes:

- Keep Streamlit and select a Python host, accepting process cost and possible
  sleep/wake latency.
- Rewrite forecasting for a browser/static architecture, with a separate technical
  feasibility and model-equivalence review.

No hosting provider, public license, GitHub repository, deployment, or website edit
is authorized by this document.
