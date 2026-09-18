"""Chronological backtesting, metrics, and validation-based selection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import model_factories


MODEL_ORDER = {"seasonal_naive": 0, "ets": 1, "global_hgb": 2}


def rolling_origin_predictions(
    data: pd.DataFrame,
    *,
    horizon: int = 8,
    folds: int = 3,
    reserve_test: int = 8,
) -> pd.DataFrame:
    """Run validation folds entirely before the isolated final test period."""

    dates = pd.DatetimeIndex(sorted(data["date"].unique()))
    first_validation_end = len(dates) - reserve_test - (folds - 1) * horizon
    if first_validation_end - horizon < 52:
        raise ValueError("Not enough history for the requested rolling-origin design.")

    results: list[pd.DataFrame] = []
    for fold in range(folds):
        validation_end = first_validation_end + fold * horizon
        validation_dates = dates[validation_end - horizon : validation_end]
        train = data[data["date"] < validation_dates[0]]
        actual = data[data["date"].isin(validation_dates)][["date", "sku_id", "demand"]]
        for model_name, factory in model_factories().items():
            predicted = factory().fit(train).predict(horizon)
            merged = predicted.merge(
                actual,
                on=["date", "sku_id"],
                how="inner",
                validate="one_to_one",
            )
            merged["model"] = model_name
            merged["fold"] = fold + 1
            merged["error"] = merged["demand"] - merged["forecast"]
            results.append(merged)
    return pd.concat(results, ignore_index=True)


def _seasonal_scale(values: np.ndarray, season_length: int = 52) -> float:
    if len(values) <= season_length:
        return float("nan")
    return float(np.mean(np.abs(values[season_length:] - values[:-season_length])))


def metric_table(predictions: pd.DataFrame, scale_history: pd.DataFrame) -> pd.DataFrame:
    scales = {
        str(sku_id): _seasonal_scale(group.sort_values("date")["demand"].to_numpy(dtype=float))
        for sku_id, group in scale_history.groupby("sku_id", sort=True)
    }
    rows: list[dict[str, object]] = []
    for (model, sku_id), group in predictions.groupby(["model", "sku_id"], sort=True):
        error = group["demand"].to_numpy(dtype=float) - group["forecast"].to_numpy(dtype=float)
        actual = group["demand"].to_numpy(dtype=float)
        abs_error = np.abs(error)
        scale = scales[str(sku_id)]
        rows.append(
            {
                "model": model,
                "sku_id": sku_id,
                "mae": float(abs_error.mean()),
                "wape": float(abs_error.sum() / actual.sum()) if actual.sum() > 0 else np.nan,
                "mase": float(abs_error.mean() / scale) if np.isfinite(scale) and scale > 0 else np.nan,
                "bias": float((-error).mean()),
                "observations": len(group),
            }
        )
    return pd.DataFrame(rows)


def select_models(metrics: pd.DataFrame, tolerance: float = 0.02) -> pd.DataFrame:
    """Select per SKU, preferring simpler models within 2% of the best score."""

    selections: list[dict[str, object]] = []
    for sku_id, group in metrics.groupby("sku_id", sort=True):
        usable = group[np.isfinite(group["mase"])].copy()
        score_column = "mase"
        if usable.empty:
            usable = group[np.isfinite(group["mae"])].copy()
            score_column = "mae"
        best = float(usable[score_column].min())
        threshold = best * (1 + tolerance)
        eligible = usable[usable[score_column] <= threshold].copy()
        eligible["complexity"] = eligible["model"].map(MODEL_ORDER)
        chosen = eligible.sort_values(["complexity", score_column]).iloc[0]
        selections.append(
            {
                "sku_id": sku_id,
                "selected_model": chosen["model"],
                "selection_metric": score_column,
                "selection_score": float(chosen[score_column]),
            }
        )
    return pd.DataFrame(selections)


def evaluate_fixed_window(
    train: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    results: list[pd.DataFrame] = []
    actual_values = actual[["date", "sku_id", "demand"]]
    for model_name, factory in model_factories().items():
        predicted = factory().fit(train).predict(horizon)
        merged = predicted.merge(
            actual_values,
            on=["date", "sku_id"],
            how="inner",
            validate="one_to_one",
        )
        merged["model"] = model_name
        merged["error"] = merged["demand"] - merged["forecast"]
        results.append(merged)
    return pd.concat(results, ignore_index=True)
