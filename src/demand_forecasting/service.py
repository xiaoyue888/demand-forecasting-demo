"""One stable computational entry point for later inventory and UI checkpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import validate_demand_data
from .evaluation import (
    evaluate_fixed_window,
    metric_table,
    rolling_origin_predictions,
    select_models,
)
from .models import model_factories


@dataclass(frozen=True)
class ForecastingCoreResult:
    validation_predictions: pd.DataFrame
    validation_metrics: pd.DataFrame
    selections: pd.DataFrame
    test_predictions: pd.DataFrame
    test_metrics: pd.DataFrame
    future_forecast: pd.DataFrame
    planning_forecast: pd.DataFrame


def _selected_rows(predictions: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    selected = predictions.merge(selections, on="sku_id", how="left", validate="many_to_one")
    return selected[selected["model"] == selected["selected_model"]].copy()


def _add_intervals(
    future: pd.DataFrame,
    validation: pd.DataFrame,
    selections: pd.DataFrame,
) -> pd.DataFrame:
    selected_errors = _selected_rows(validation, selections)
    result = future.copy()
    for level in (0.80, 0.95):
        quantiles = (
            selected_errors.assign(abs_error=selected_errors["error"].abs())
            .groupby(["sku_id", "horizon_step"])["abs_error"]
            .quantile(level)
            .rename("radius")
            .reset_index()
        )
        result = result.merge(quantiles, on=["sku_id", "horizon_step"], how="left")
        label = int(level * 100)
        result[f"lower_{label}"] = np.maximum(0.0, result["forecast"] - result["radius"])
        result[f"upper_{label}"] = result["forecast"] + result["radius"]
        result = result.drop(columns="radius")
    return result


def run_forecasting_core(
    frame: pd.DataFrame,
    *,
    horizon: int = 8,
    folds: int = 8,
    planning_horizon: int = 13,
) -> ForecastingCoreResult:
    """Validate, backtest, select, test once, and forecast the next eight weeks."""

    if planning_horizon < horizon:
        raise ValueError("planning_horizon cannot be shorter than the displayed forecast horizon.")
    data = validate_demand_data(frame)
    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    test_dates = dates[-horizon:]
    pretest = data[data["date"] < test_dates[0]].copy()
    test_actual = data[data["date"].isin(test_dates)].copy()

    maximum_folds = max(1, (len(dates) - horizon - 52) // horizon)
    effective_folds = min(folds, maximum_folds)
    validation_predictions = rolling_origin_predictions(
        data,
        horizon=horizon,
        folds=effective_folds,
        reserve_test=horizon,
    )
    validation_metrics = metric_table(validation_predictions, pretest)
    selections = select_models(validation_metrics)

    test_predictions_all = evaluate_fixed_window(pretest, test_actual, horizon=horizon)
    test_predictions = _selected_rows(test_predictions_all, selections)
    test_metrics = metric_table(test_predictions, pretest)

    forecasts: list[pd.DataFrame] = []
    for model_name, factory in model_factories().items():
        predicted = factory().fit(data).predict(planning_horizon)
        predicted["model"] = model_name
        forecasts.append(predicted)
    future_all = pd.concat(forecasts, ignore_index=True)
    planning_forecast = _selected_rows(future_all, selections)
    future = planning_forecast[planning_forecast["horizon_step"] <= horizon].copy()
    future = _add_intervals(future, validation_predictions, selections)

    return ForecastingCoreResult(
        validation_predictions=validation_predictions,
        validation_metrics=validation_metrics,
        selections=selections,
        test_predictions=test_predictions,
        test_metrics=test_metrics,
        future_forecast=future,
        planning_forecast=planning_forecast,
    )
