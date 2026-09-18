"""Small, purposeful forecasting model set approved at Checkpoint A."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class PanelForecaster(Protocol):
    name: str

    def fit(self, panel: pd.DataFrame) -> "PanelForecaster": ...

    def predict(self, horizon: int) -> pd.DataFrame: ...


def _future_rows(history: pd.DataFrame, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    last_date = history["date"].max()
    rows: list[dict[str, object]] = []
    for sku_id, values in predictions.items():
        for step, value in enumerate(values, start=1):
            rows.append(
                {
                    "date": last_date + pd.Timedelta(weeks=step),
                    "sku_id": sku_id,
                    "horizon_step": step,
                    "forecast": max(0.0, float(value)),
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "sku_id"], ignore_index=True)


@dataclass
class SeasonalNaiveForecaster:
    season_length: int = 52
    fallback_window: int = 4
    name: str = "seasonal_naive"

    def fit(self, panel: pd.DataFrame) -> "SeasonalNaiveForecaster":
        self.history = panel.sort_values(["date", "sku_id"]).copy()
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        predictions: dict[str, np.ndarray] = {}
        for sku_id, group in self.history.groupby("sku_id", sort=True):
            values = group.sort_values("date")["demand"].to_numpy(dtype=float)
            if len(values) >= self.season_length:
                forecast = np.resize(values[-self.season_length :], horizon)
            else:
                forecast = np.repeat(np.mean(values[-self.fallback_window :]), horizon)
            predictions[str(sku_id)] = forecast
        return _future_rows(self.history, predictions)


@dataclass
class ETSForecaster:
    season_length: int = 52
    name: str = "ets"

    def fit(self, panel: pd.DataFrame) -> "ETSForecaster":
        self.history = panel.sort_values(["date", "sku_id"]).copy()
        self.fitted: dict[str, object] = {}
        for sku_id, group in self.history.groupby("sku_id", sort=True):
            values = group.sort_values("date")["demand"].to_numpy(dtype=float)
            seasonal = "add" if len(values) >= 2 * self.season_length else None
            try:
                model = ExponentialSmoothing(
                    values,
                    trend="add",
                    damped_trend=True,
                    seasonal=seasonal,
                    seasonal_periods=self.season_length if seasonal else None,
                    initialization_method="estimated",
                )
                fit_options: dict[str, float | bool] = {
                    "optimized": False,
                    "remove_bias": True,
                    "smoothing_level": 0.25,
                    "smoothing_trend": 0.05,
                    "damping_trend": 0.95,
                }
                if seasonal:
                    fit_options["smoothing_seasonal"] = 0.10
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    self.fitted[str(sku_id)] = model.fit(**fit_options)
            except (ValueError, np.linalg.LinAlgError):
                self.fitted[str(sku_id)] = None
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        predictions: dict[str, np.ndarray] = {}
        for sku_id, group in self.history.groupby("sku_id", sort=True):
            fitted = self.fitted[str(sku_id)]
            if fitted is None:
                values = group.sort_values("date")["demand"].to_numpy(dtype=float)
                forecast = np.repeat(np.mean(values[-4:]), horizon)
            else:
                forecast = np.asarray(fitted.forecast(horizon), dtype=float)
            predictions[str(sku_id)] = forecast
        return _future_rows(self.history, predictions)


LAGS = (1, 2, 4, 8, 13, 26, 52)


def _feature_row(values: list[float], date: pd.Timestamp, sku_id: str) -> dict[str, object]:
    row: dict[str, object] = {
        "sku_id": sku_id,
        "trend": date.toordinal() / 7.0,
        "week_sin": np.sin(2 * np.pi * date.isocalendar().week / 52),
        "week_cos": np.cos(2 * np.pi * date.isocalendar().week / 52),
    }
    for lag in LAGS:
        row[f"lag_{lag}"] = values[-lag]
    for window in (4, 13):
        tail = np.asarray(values[-window:], dtype=float)
        row[f"mean_{window}"] = float(tail.mean())
        row[f"std_{window}"] = float(tail.std(ddof=0))
    return row


@dataclass
class GlobalHGBForecaster:
    random_state: int = 20260915
    name: str = "global_hgb"

    def fit(self, panel: pd.DataFrame) -> "GlobalHGBForecaster":
        self.history = panel.sort_values(["date", "sku_id"]).copy()
        rows: list[dict[str, object]] = []
        targets: list[float] = []
        for sku_id, group in self.history.groupby("sku_id", sort=True):
            ordered = group.sort_values("date")
            values = ordered["demand"].astype(float).tolist()
            dates = ordered["date"].tolist()
            for position in range(max(LAGS), len(values)):
                rows.append(_feature_row(values[:position], dates[position], str(sku_id)))
                targets.append(values[position])
        raw = pd.DataFrame(rows)
        encoded = pd.get_dummies(raw, columns=["sku_id"], dtype=float)
        self.feature_columns = encoded.columns.tolist()
        self.model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_iter=60,
            max_leaf_nodes=16,
            l2_regularization=1.0,
            random_state=self.random_state,
        )
        self.model.fit(encoded, np.asarray(targets))
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        histories = {
            str(sku_id): group.sort_values("date")["demand"].astype(float).tolist()
            for sku_id, group in self.history.groupby("sku_id", sort=True)
        }
        predictions = {sku_id: [] for sku_id in histories}
        last_date = self.history["date"].max()
        for step in range(1, horizon + 1):
            date = last_date + pd.Timedelta(weeks=step)
            raw = pd.DataFrame(
                [_feature_row(values, date, sku_id) for sku_id, values in histories.items()]
            )
            encoded = pd.get_dummies(raw, columns=["sku_id"], dtype=float)
            encoded = encoded.reindex(columns=self.feature_columns, fill_value=0.0)
            values = np.maximum(0.0, self.model.predict(encoded))
            for (sku_id, history), value in zip(histories.items(), values, strict=True):
                history.append(float(value))
                predictions[sku_id].append(float(value))
        arrays = {sku_id: np.asarray(values) for sku_id, values in predictions.items()}
        return _future_rows(self.history, arrays)


def model_factories() -> dict[str, type[SeasonalNaiveForecaster] | type[ETSForecaster] | type[GlobalHGBForecaster]]:
    return {
        "seasonal_naive": SeasonalNaiveForecaster,
        "ets": ETSForecaster,
        "global_hgb": GlobalHGBForecaster,
    }
