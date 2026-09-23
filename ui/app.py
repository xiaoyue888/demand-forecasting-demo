from __future__ import annotations

import hashlib
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demand_forecasting import (
    DataValidationError,
    PlanningScenario,
    build_inventory_decisions,
    diagnose_demand,
    generate_synthetic_data,
    run_forecasting_core,
    validate_demand_data,
)
from demand_forecasting.web import case_study_url

MODEL_LABELS = {
    "seasonal_naive": "Seasonal Naive",
    "ets": "ETS",
    "global_hgb": "Global HGB",
}


st.set_page_config(
    page_title="Demand Forecasting & Inventory Decision Support",
    page_icon="↗",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(f"<style>{Path(__file__).with_name('app.css').read_text()}</style>", unsafe_allow_html=True)
case_study_link = escape(case_study_url(), quote=True)


@st.cache_data(show_spinner=False)
def compute_synthetic_forecasting_core():
    """Cache only the public synthetic scenario across sessions."""
    return run_forecasting_core(generate_synthetic_data())


def compute_uploaded_forecasting_core(data: pd.DataFrame):
    """Reuse uploaded-data results only inside the current Streamlit session."""
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(data, index=True).values.tobytes()
    ).hexdigest()
    cached = st.session_state.get("_uploaded_forecasting_core")
    if not cached or cached["fingerprint"] != fingerprint:
        cached = {
            "fingerprint": fingerprint,
            "result": run_forecasting_core(data),
        }
        st.session_state["_uploaded_forecasting_core"] = cached
    return cached["result"]


def risk_badge(level: str) -> str:
    return f'<span class="risk-badge {escape(level)}">{escape(level.title())}</span>'


def metric_value(value: float, *, percentage: bool = False) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.1%}" if percentage else f"{value:.2f}"


def inventory_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <span>{escape(label)}</span>
          <strong>{escape(value)}</strong>
          <small>{escape(note)}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <nav class="portfolio-nav">
      <span class="portfolio-mark">XZ</span>
      <a href="{case_study_link}" target="_blank" rel="noopener noreferrer">← Back to case study</a>
    </nav>
    <header class="app-header">
      <div>
        <p>AI &amp; Machine Learning · Supply Chain</p>
        <h1>Demand Forecasting &amp; Inventory Decision Support</h1>
      </div>
      <div class="scenario-note"><strong>Scenario analysis</strong><br>
      Planning estimates, not guaranteed business outcomes.</div>
    </header>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    source_col, lead_col, service_col, growth_col = st.columns([1.8, 1, 1, 1])
    with source_col:
        st.markdown("#### Dataset")
        source = st.radio(
            "Dataset source",
            ["Built-in synthetic scenario", "Upload CSV"],
            horizontal=True,
            label_visibility="collapsed",
        )
        uploaded_file = None
        if source == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Weekly CSV",
                type=["csv"],
                help="Processed in memory for this session only.",
            )
    with lead_col:
        st.markdown("#### Lead time")
        lead_time = st.slider("Supplier lead time", 1, 12, 4, format="%d weeks", label_visibility="collapsed")
    with service_col:
        st.markdown("#### Service level")
        service_level_percent = st.slider(
            "Target service level",
            80,
            99,
            95,
            1,
            format="%d%%",
            label_visibility="collapsed",
        )
        service_level = service_level_percent / 100
    with growth_col:
        st.markdown("#### Demand growth")
        annual_growth_percent = st.slider("Annual demand growth", -50, 100, 0, 5, format="%d%%", label_visibility="collapsed")

if source == "Built-in synthetic scenario":
    st.session_state.pop("_uploaded_forecasting_core", None)
    data = generate_synthetic_data()
    source_note = (
        f"Synthetic · 12 SKUs · 156 weeks · as of {data['date'].max():%d %b %Y}"
    )
elif uploaded_file is None:
    st.session_state.pop("_uploaded_forecasting_core", None)
    st.info("Choose a weekly CSV to begin. Required fields: date, sku_id, demand, current_on_hand, lead_time_weeks.")
    st.stop()
else:
    try:
        raw_data = pd.read_csv(uploaded_file)
        data = validate_demand_data(raw_data)
        source_note = f"Uploaded in memory · {data['sku_id'].nunique()} SKUs · {data['date'].nunique()} weeks"
    except (DataValidationError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error("The CSV needs attention before forecasting can run.")
        if isinstance(exc, DataValidationError):
            for issue in exc.issues:
                st.markdown(f"- **{escape(issue.code.replace('_', ' ').title())}:** {escape(issue.message)}")
        else:
            st.write("The file could not be parsed as a UTF-8 comma-separated CSV.")
        st.stop()

with st.status("Running chronological backtests…", expanded=False) as run_status:
    try:
        if source == "Built-in synthetic scenario":
            core = compute_synthetic_forecasting_core()
        else:
            core = compute_uploaded_forecasting_core(data)
        scenario = PlanningScenario(
            service_level=service_level,
            annual_demand_growth=annual_growth_percent / 100,
            lead_time_weeks=lead_time,
        )
        inventory = build_inventory_decisions(
            data,
            core.planning_forecast,
            core.validation_predictions,
            core.selections,
            scenario=scenario,
        )
    except (ValueError, RuntimeError) as exc:
        run_status.update(label="The analysis could not be completed", state="error")
        st.error(str(exc))
        st.stop()
    run_status.update(label="Analysis ready", state="complete")

decisions = inventory.decisions.copy()
risk_order = {"high": 0, "watch": 1, "low": 2}
decisions["risk_order"] = decisions["stockout_risk"].map(risk_order)
decisions = decisions.sort_values(["risk_order", "sku_id"], ignore_index=True)
high_count = int((decisions["stockout_risk"] == "high").sum())
watch_count = int((decisions["stockout_risk"] == "watch").sum())
attention_count = high_count + watch_count

st.markdown(
    f"""
    <div class="overview-strip">
      <div><span>{escape(source_note)}</span><strong>{attention_count} of {len(decisions)} SKUs need attention</strong></div>
      <div class="overview-counts">
        <span class="high"><b>{high_count}</b> High</span>
        <span class="watch"><b>{watch_count}</b> Watch</span>
        <span class="low"><b>{len(decisions) - attention_count}</b> Low</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

selector_col, workspace_col = st.columns([0.9, 3.1], gap="large")
with selector_col:
    st.markdown("### SKUs")
    st.caption("Ordered by current stockout risk")
    if "sku_picker" not in st.session_state or st.session_state.sku_picker not in decisions["sku_id"].tolist():
        st.session_state.sku_picker = decisions.iloc[0]["sku_id"]
    st.markdown("#### Attention list")
    attention_table = decisions[["sku_id", "stockout_risk"]].rename(
        columns={"sku_id": "SKU", "stockout_risk": "Risk"}
    )
    attention_table["Risk"] = attention_table["Risk"].str.upper()
    selection_event = st.dataframe(
        attention_table,
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key="attention_table",
    )
    if selection_event.selection.rows:
        clicked_sku = attention_table.iloc[selection_event.selection.rows[0]]["SKU"]
        if clicked_sku != st.session_state.sku_picker:
            st.session_state.sku_picker = clicked_sku
    selected_sku = st.session_state.sku_picker

    st.markdown("#### CSV contract")
    st.code("date\nsku_id\ndemand\ncurrent_on_hand\nlead_time_weeks", language=None)
    st.caption("Monday dates · 80–260 weeks per SKU · maximum 100 SKUs")

with workspace_col:
    decision = decisions.loc[decisions["sku_id"] == selected_sku].iloc[0]
    sku_history = data[data["sku_id"] == selected_sku].sort_values("date")
    diagnostics = diagnose_demand(sku_history["demand"])
    sku_future = core.future_forecast[core.future_forecast["sku_id"] == selected_sku].sort_values("horizon_step")
    fold_count = int(core.validation_predictions["fold"].nunique())

    st.markdown(
        f"""
        <div class="selection-heading">
          <div><span>Selected SKU</span><h2>{escape(selected_sku)}</h2></div>
          <dl>
            <div><dt>Pattern</dt><dd>{escape(str(diagnostics['pattern']))}</dd></div>
            <div><dt>Selected model</dt><dd>{escape(MODEL_LABELS.get(str(decision['selected_model']), str(decision['selected_model'])))}</dd></div>
            <div><dt>Forecast horizon</dt><dd>8 weeks</dd></div>
          </dl>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Historical demand & forecast")
    st.caption("Weekly units with empirical 80% and 95% uncertainty ranges")
    display_history = sku_history.tail(52)
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=pd.concat([sku_future["date"], sku_future["date"].iloc[::-1]]),
            y=pd.concat([sku_future["upper_95"], sku_future["lower_95"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(102, 160, 204, .16)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name="95% range",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=pd.concat([sku_future["date"], sku_future["date"].iloc[::-1]]),
            y=pd.concat([sku_future["upper_80"], sku_future["lower_80"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(73, 137, 188, .24)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name="80% range",
        )
    )
    chart.add_trace(go.Scatter(x=display_history["date"], y=display_history["demand"], mode="lines", name="Actual", line={"color": "#26282b", "width": 2}))
    chart.add_trace(go.Scatter(x=sku_future["date"], y=sku_future["forecast"], mode="lines+markers", name="Forecast", line={"color": "#23658f", "width": 2, "dash": "dash"}))
    chart.add_vline(x=sku_history["date"].max(), line_dash="dot", line_color="#81868d")
    chart.update_layout(
        height=360,
        margin={"l": 8, "r": 8, "t": 16, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        yaxis={"title": "Units", "gridcolor": "#e0e1e2", "rangemode": "tozero"},
        xaxis={"gridcolor": "rgba(0,0,0,0)"},
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Intervals are empirical ranges calibrated from rolling validation errors. "
        "They communicate uncertainty but are not probabilistic guarantees."
    )

    risk_col, action_col = st.columns(2, gap="medium")
    with risk_col:
        st.markdown(
            f'<div class="section-kicker">Inventory risk</div><h3>{risk_badge(decision["stockout_risk"])} Stockout risk</h3>',
            unsafe_allow_html=True,
        )
        card_a, card_b = st.columns(2)
        with card_a:
            inventory_card("On hand", f"{int(decision['on_hand'])} units")
            inventory_card("Safety stock", f"{decision['safety_stock']:.1f} units")
        with card_b:
            inventory_card("Lead-time demand", f"{decision['lead_time_demand']:.1f} units")
            inventory_card("Stock coverage", f"{decision['stock_coverage_weeks']:.1f} weeks")
        coverage_gap = lead_time - float(decision["stock_coverage_weeks"])
        if coverage_gap > 0:
            st.error(
                f"Coverage is {coverage_gap:.1f} weeks shorter than the {lead_time}-week lead time; "
                "expected supply may arrive after inventory is depleted."
            )
        else:
            st.success(
                f"Coverage exceeds the {lead_time}-week lead time by {-coverage_gap:.1f} weeks "
                "under the point forecast."
            )
        depletion = decision["expected_depletion_date"]
        if pd.isna(depletion):
            st.info("Inventory is not expected to deplete within the supplied planning horizon.")
        else:
            st.warning(f"Expected depletion: week of {pd.Timestamp(depletion):%d %b %Y}")

    with action_col:
        st.markdown('<div class="section-kicker">Recommended action</div>', unsafe_allow_html=True)
        st.markdown(f"### {escape(str(decision['reorder_status']).title())}")
        st.markdown(
            f'<div class="order-quantity"><strong>{int(decision["recommended_order_quantity"])}</strong><span>suggested units</span></div>',
            unsafe_allow_html=True,
        )
        reorder_date = decision["recommended_reorder_date"]
        date_text = "Not within planning horizon" if pd.isna(reorder_date) else f"{pd.Timestamp(reorder_date):%d %b %Y}"
        inventory_card("Recommended reorder date", date_text)
        inventory_card("Reorder point", f"{decision['reorder_point']:.1f} units")

    with st.expander("How the recommendation was calculated"):
        st.markdown(
            f"""
            - **Lead-time demand:** sum of the next {lead_time} scenario-adjusted weekly forecasts.
            - **Safety stock:** the larger of the {service_level:.0%} empirical cumulative-error quantile and a `z × σL` variance floor.
            - **Calibration sample:** {int(decision['safety_stock_observations'])} cumulative-error observations; empirical component {decision['safety_stock_empirical_quantile']:.1f} units, parametric floor {decision['safety_stock_parametric_floor']:.1f} units.
            - **Reorder point:** lead-time demand + safety stock.
            - **Order-up-to level:** demand over lead time + one review week + safety stock.
            - **Inventory position:** current on-hand only; open purchase orders are not included.
            - **Calibration:** {decision['calibration_method']}.
            """
        )
        if bool(decision["calibration_warning"]):
            st.warning(
                "Fewer than 20 cumulative-error observations support this safety-stock estimate. "
                "Treat it as a low-confidence scenario result."
            )
        elif float(decision["safety_stock"]) == 0:
            st.info(
                "Safety stock is zero because both the empirical under-forecast quantile and "
                "the variance-based floor are zero for this calibration sample."
            )

    st.markdown("### Backtest comparison")
    st.caption(f"{fold_count} chronological rolling-origin validation windows; the final 8-week test remains isolated.")
    metrics = core.validation_metrics[core.validation_metrics["sku_id"] == selected_sku].copy()
    metrics["Model"] = metrics["model"].map(MODEL_LABELS).fillna(metrics["model"])
    metrics["MAE"] = metrics["mae"].map(metric_value)
    metrics["WAPE"] = metrics["wape"].map(lambda value: metric_value(value, percentage=True))
    metrics["MASE"] = metrics["mase"].map(metric_value)
    metrics["Bias"] = metrics["bias"].map(metric_value)
    metrics["Selection"] = np.where(metrics["model"] == decision["selected_model"], "Selected", "")
    metrics = metrics.sort_values("model", key=lambda values: values.map({"seasonal_naive": 0, "ets": 1, "global_hgb": 2}))
    st.dataframe(
        metrics[["Model", "MAE", "WAPE", "MASE", "Bias", "Selection"]],
        hide_index=True,
        width="stretch",
    )

    diag_a, diag_b, diag_c, diag_d = st.columns(4)
    variability = float(diagnostics["coefficient_of_variation"])
    annual_correlation = float(diagnostics["seasonal_correlation"])
    recent_trend = float(diagnostics["recent_weekly_trend"])
    variability_reading = "relatively stable" if variability < 0.35 else "moderate variability" if variability < 0.60 else "high variability"
    correlation_reading = (
        "strong yearly recurrence"
        if np.isfinite(annual_correlation) and annual_correlation >= 0.50
        else "some yearly recurrence"
        if np.isfinite(annual_correlation) and annual_correlation >= 0.25
        else "weak or unavailable recurrence"
    )
    diag_a.metric(
        "Zero-demand weeks",
        metric_value(float(diagnostics["zero_share"]), percentage=True),
        help="Share of observed weeks with demand equal to zero.",
    )
    diag_b.metric(
        "Demand variability (CV)",
        metric_value(variability),
        help="Coefficient of variation: standard deviation divided by mean demand.",
    )
    diag_c.metric(
        "52-week correlation",
        metric_value(annual_correlation),
        help="Correlation between weekly demand and demand 52 weeks earlier.",
    )
    diag_d.metric(
        "Recent trend",
        f"{recent_trend:+.2f} units/week",
        help="Linear slope fitted to the latest 13 observed weeks.",
    )
    st.caption(
        f"Interpretation: demand is {variability_reading}; {correlation_reading}; "
        f"the recent slope is {recent_trend:+.2f} units per week."
    )

    with st.expander("Decision assumptions and limitations"):
        st.markdown(
            """
            - Inventory is reviewed weekly and no open purchase orders are modeled.
            - The replenishment rule is transparent planning logic, not an optimization solver.
            - No holding costs, shelf life, supplier constraints, or order minimums are included.
            - Longer lead-time uncertainty beyond the validation horizon uses a disclosed circular-block approximation.
            - Uploaded data stays in this running process and is not intentionally persisted or transmitted.
            """
        )

st.markdown(
    "<footer>Built from synthetic or user-supplied weekly data · Forecasts support decisions; they do not guarantee outcomes.</footer>",
    unsafe_allow_html=True,
)
