"""Transparent inventory-risk and replenishment planning logic.

This module is intentionally a decision-support layer, not an inventory optimizer.
It assumes one current on-hand snapshot, no open purchase orders, and weekly review.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlanningScenario:
    """Small set of approved what-if assumptions."""

    service_level: float = 0.95
    annual_demand_growth: float = 0.0
    lead_time_weeks: int | None = None
    review_period_weeks: int = 1

    def __post_init__(self) -> None:
        if not 0.80 <= self.service_level <= 0.99:
            raise ValueError("service_level must be between 0.80 and 0.99.")
        if not -0.50 <= self.annual_demand_growth <= 1.00:
            raise ValueError("annual_demand_growth must be between -0.50 and 1.00.")
        if self.lead_time_weeks is not None and not 1 <= self.lead_time_weeks <= 12:
            raise ValueError("lead_time_weeks must be between 1 and 12.")
        if self.review_period_weeks != 1:
            raise ValueError("The first version uses a fixed one-week review period.")


@dataclass(frozen=True)
class InventoryDecisionResult:
    decisions: pd.DataFrame
    projections: pd.DataFrame


@dataclass(frozen=True)
class SafetyStockEstimate:
    quantity: float
    empirical_quantile: float
    parametric_floor: float
    cumulative_error_std: float
    observation_count: int
    method: str


def _apply_growth(forecast: pd.Series, steps: pd.Series, annual_growth: float) -> pd.Series:
    growth_factor = np.power(1.0 + annual_growth, steps.to_numpy(dtype=float) / 52.0)
    return pd.Series(forecast.to_numpy(dtype=float) * growth_factor, index=forecast.index)


def _selected_validation_errors(
    validation_predictions: pd.DataFrame,
    selections: pd.DataFrame,
    sku_id: str,
) -> pd.DataFrame:
    selected_model = selections.loc[selections["sku_id"] == sku_id, "selected_model"]
    if len(selected_model) != 1:
        raise ValueError(f"Expected exactly one selected model for {sku_id}.")
    rows = validation_predictions[
        (validation_predictions["sku_id"] == sku_id)
        & (validation_predictions["model"] == selected_model.iloc[0])
    ].copy()
    if rows.empty:
        raise ValueError(f"No validation errors are available for {sku_id}.")
    return rows.sort_values(["fold", "horizon_step"])


def empirical_cumulative_error_quantile(
    errors: pd.DataFrame,
    *,
    lead_time_weeks: int,
    service_level: float,
) -> tuple[float, str]:
    """Estimate cumulative lead-time under-forecast error without randomness.

    For lead times within the validation horizon, each fold contributes its direct
    cumulative error. For a longer lead time, circular starts within each fold are
    used as a transparent block-bootstrap approximation. The approximation is
    explicitly returned so a later interface can disclose it.
    """

    fold_lengths = errors.groupby("fold").size()
    shortest_fold = int(fold_lengths.min())
    cumulative_samples: list[float] = []

    if lead_time_weeks <= shortest_fold:
        for _, fold in errors.groupby("fold", sort=True):
            ordered = fold.sort_values("horizon_step")["error"].to_numpy(dtype=float)
            for start in range(len(ordered) - lead_time_weeks + 1):
                cumulative_samples.append(float(ordered[start : start + lead_time_weeks].sum()))
        method = "overlapping rolling-origin cumulative errors"
    else:
        for _, fold in errors.groupby("fold", sort=True):
            ordered = fold.sort_values("horizon_step")["error"].to_numpy(dtype=float)
            for start in range(len(ordered)):
                positions = np.arange(start, start + lead_time_weeks) % len(ordered)
                cumulative_samples.append(float(ordered[positions].sum()))
        method = "circular block approximation beyond validation horizon"

    quantile = float(np.quantile(np.asarray(cumulative_samples), service_level))
    return max(0.0, quantile), method


def estimate_safety_stock(
    errors: pd.DataFrame,
    *,
    lead_time_weeks: int,
    service_level: float,
) -> SafetyStockEstimate:
    """Combine the empirical quantile with a disclosed normal-variance floor."""

    fold_lengths = errors.groupby("fold").size()
    shortest_fold = int(fold_lengths.min())
    samples: list[float] = []
    if lead_time_weeks <= shortest_fold:
        for _, fold in errors.groupby("fold", sort=True):
            ordered = fold.sort_values("horizon_step")["error"].to_numpy(dtype=float)
            for start in range(len(ordered) - lead_time_weeks + 1):
                samples.append(float(ordered[start : start + lead_time_weeks].sum()))
        method = "overlapping rolling-origin cumulative errors"
    else:
        for _, fold in errors.groupby("fold", sort=True):
            ordered = fold.sort_values("horizon_step")["error"].to_numpy(dtype=float)
            for start in range(len(ordered)):
                positions = np.arange(start, start + lead_time_weeks) % len(ordered)
                samples.append(float(ordered[positions].sum()))
        method = "circular block approximation beyond validation horizon"

    sample_array = np.asarray(samples, dtype=float)
    empirical = max(0.0, float(np.quantile(sample_array, service_level)))
    cumulative_std = float(sample_array.std(ddof=1)) if len(sample_array) > 1 else 0.0
    z_score = NormalDist().inv_cdf(service_level)
    parametric_floor = max(0.0, z_score * cumulative_std)
    quantity = max(empirical, parametric_floor)
    return SafetyStockEstimate(
        quantity=quantity,
        empirical_quantile=empirical,
        parametric_floor=parametric_floor,
        cumulative_error_std=cumulative_std,
        observation_count=len(sample_array),
        method=f"max(empirical quantile, z × cumulative-error σ); {method}",
    )


def build_inventory_decisions(
    history: pd.DataFrame,
    future_forecast: pd.DataFrame,
    validation_predictions: pd.DataFrame,
    selections: pd.DataFrame,
    *,
    scenario: PlanningScenario | None = None,
) -> InventoryDecisionResult:
    """Translate selected forecasts into reproducible weekly inventory decisions."""

    scenario = scenario or PlanningScenario()
    summaries: list[dict[str, object]] = []
    projections: list[pd.DataFrame] = []
    last_history_date = pd.Timestamp(history["date"].max())

    for sku_id, history_group in history.groupby("sku_id", sort=True):
        sku_id = str(sku_id)
        snapshots = history_group[["current_on_hand", "lead_time_weeks"]].drop_duplicates()
        if len(snapshots) != 1:
            raise ValueError(f"Inventory snapshot is inconsistent for {sku_id}.")
        on_hand = int(snapshots.iloc[0]["current_on_hand"])
        stored_lead_time = int(snapshots.iloc[0]["lead_time_weeks"])
        lead_time = scenario.lead_time_weeks or stored_lead_time
        review_period = scenario.review_period_weeks
        required_steps = lead_time + review_period

        selected_model_row = selections.loc[selections["sku_id"] == sku_id, "selected_model"]
        if len(selected_model_row) != 1:
            raise ValueError(f"Expected exactly one selected model for {sku_id}.")
        selected_model = selected_model_row.iloc[0]
        forecast_mask = future_forecast["sku_id"] == sku_id
        if "model" in future_forecast.columns:
            forecast_mask &= future_forecast["model"] == selected_model
        sku_forecast = future_forecast[forecast_mask].sort_values("horizon_step").copy()
        if len(sku_forecast) < required_steps:
            raise ValueError(
                f"{sku_id} needs {required_steps} forecast weeks for lead time plus review, "
                f"but only {len(sku_forecast)} were supplied."
            )

        sku_forecast["scenario_forecast"] = _apply_growth(
            sku_forecast["forecast"],
            sku_forecast["horizon_step"],
            scenario.annual_demand_growth,
        )
        errors = _selected_validation_errors(validation_predictions, selections, sku_id)
        safety_estimate = estimate_safety_stock(
            errors,
            lead_time_weeks=lead_time,
            service_level=scenario.service_level,
        )
        safety_stock = safety_estimate.quantity

        demand = sku_forecast["scenario_forecast"].to_numpy(dtype=float)
        lead_time_demand = float(demand[:lead_time].sum())
        reorder_point = lead_time_demand + safety_stock
        order_up_to = float(demand[:required_steps].sum()) + safety_stock

        starts = on_hand - np.concatenate(([0.0], np.cumsum(demand[:-1])))
        ends = starts - demand
        sku_forecast["projected_start_inventory"] = starts
        sku_forecast["projected_end_inventory"] = ends
        projections.append(sku_forecast)

        depletion_rows = sku_forecast.loc[sku_forecast["projected_end_inventory"] < 0, "date"]
        depletion_date = pd.Timestamp(depletion_rows.iloc[0]) if not depletion_rows.empty else pd.NaT

        if on_hand < lead_time_demand:
            stockout_risk = "high"
        elif on_hand < reorder_point:
            stockout_risk = "watch"
        else:
            stockout_risk = "low"

        excess_units = max(0.0, on_hand - order_up_to)
        overstock_risk = "watch" if excess_units > 0 else "low"

        reorder_index: int | None = None
        max_review_index = len(demand) - required_steps
        for index in range(max_review_index + 1):
            future_lead_demand = float(demand[index : index + lead_time].sum())
            future_reorder_point = future_lead_demand + safety_stock
            if starts[index] <= future_reorder_point:
                reorder_index = index
                break

        if reorder_index is None:
            reorder_date = pd.NaT
            recommended_quantity = 0
            reorder_status = "not within supplied planning horizon"
        else:
            reorder_date = last_history_date if reorder_index == 0 else pd.Timestamp(
                sku_forecast.iloc[reorder_index]["date"]
            )
            target = float(demand[reorder_index : reorder_index + required_steps].sum()) + safety_stock
            recommended_quantity = int(np.ceil(max(0.0, target - starts[reorder_index])))
            reorder_status = "reorder now" if reorder_index == 0 else "future reorder"

        mean_near_term = float(demand[: min(8, len(demand))].mean())
        stock_coverage_weeks = on_hand / mean_near_term if mean_near_term > 0 else np.inf
        summaries.append(
            {
                "sku_id": sku_id,
                "selected_model": selected_model,
                "on_hand": on_hand,
                "lead_time_weeks": lead_time,
                "service_level": scenario.service_level,
                "annual_demand_growth": scenario.annual_demand_growth,
                "lead_time_demand": lead_time_demand,
                "safety_stock": safety_stock,
                "reorder_point": reorder_point,
                "order_up_to_level": order_up_to,
                "stock_coverage_weeks": stock_coverage_weeks,
                "stockout_risk": stockout_risk,
                "overstock_risk": overstock_risk,
                "excess_units": excess_units,
                "expected_depletion_date": depletion_date,
                "recommended_reorder_date": reorder_date,
                "recommended_order_quantity": recommended_quantity,
                "reorder_status": reorder_status,
                "safety_stock_observations": safety_estimate.observation_count,
                "safety_stock_empirical_quantile": safety_estimate.empirical_quantile,
                "safety_stock_parametric_floor": safety_estimate.parametric_floor,
                "calibration_method": safety_estimate.method,
                "calibration_warning": safety_estimate.observation_count < 20,
                "assumption_note": (
                    "Scenario analysis: weekly review, no open purchase orders, "
                    "no holding-cost or shelf-life optimization."
                ),
            }
        )

    return InventoryDecisionResult(
        decisions=pd.DataFrame(summaries).sort_values("sku_id", ignore_index=True),
        projections=pd.concat(projections, ignore_index=True),
    )
