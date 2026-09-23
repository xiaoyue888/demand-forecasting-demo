from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from demand_forecasting import PlanningScenario, build_inventory_decisions
from demand_forecasting.inventory import empirical_cumulative_error_quantile, estimate_safety_stock


def _history(on_hand: int = 25, lead_time: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-01-06", periods=4, freq="W-MON"),
            "sku_id": ["SKU_A"] * 4,
            "demand": [10] * 4,
            "current_on_hand": [on_hand] * 4,
            "lead_time_weeks": [lead_time] * 4,
        }
    )


def _forecast(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-02-03", periods=len(values), freq="W-MON"),
            "sku_id": ["SKU_A"] * len(values),
            "horizon_step": range(1, len(values) + 1),
            "forecast": values,
            "model": ["seasonal_naive"] * len(values),
        }
    )


def _validation(fold_errors: list[list[float]]) -> pd.DataFrame:
    rows = []
    for fold, errors in enumerate(fold_errors, start=1):
        for step, error in enumerate(errors, start=1):
            forecast = 10.0
            rows.append(
                {
                    "date": pd.Timestamp("2024-01-01") + timedelta(weeks=step),
                    "sku_id": "SKU_A",
                    "horizon_step": step,
                    "forecast": forecast,
                    "demand": forecast + error,
                    "model": "seasonal_naive",
                    "fold": fold,
                    "error": error,
                }
            )
    return pd.DataFrame(rows)


def _selections() -> pd.DataFrame:
    return pd.DataFrame({"sku_id": ["SKU_A"], "selected_model": ["seasonal_naive"]})


def test_hand_calculated_reorder_date_quantity_and_depletion() -> None:
    result = build_inventory_decisions(
        _history(on_hand=25, lead_time=2),
        _forecast([10, 10, 10, 10, 10]),
        _validation([[0, 0], [0, 0], [0, 0]]),
        _selections(),
    )
    decision = result.decisions.iloc[0]
    assert decision["lead_time_demand"] == 20
    assert decision["safety_stock"] == 0
    assert decision["reorder_point"] == 20
    assert decision["order_up_to_level"] == 30
    assert decision["recommended_reorder_date"] == pd.Timestamp("2025-02-10")
    assert decision["recommended_order_quantity"] == 15
    assert decision["expected_depletion_date"] == pd.Timestamp("2025-02-17")
    assert decision["stockout_risk"] == "low"


def test_safety_stock_uses_cumulative_underforecast_error() -> None:
    errors = _validation([[2, 3], [4, 1], [3, 2]])
    stock, method = empirical_cumulative_error_quantile(
        errors,
        lead_time_weeks=2,
        service_level=0.95,
    )
    assert stock == 5
    assert method == "overlapping rolling-origin cumulative errors"


def test_service_level_lead_time_and_growth_are_directionally_consistent() -> None:
    validation = _validation([[1, 1, 1, 1], [2, 2, 2, 2], [5, 5, 5, 5]])
    forecast = _forecast([10] * 8)
    low_service = build_inventory_decisions(
        _history(on_hand=5, lead_time=2),
        forecast,
        validation,
        _selections(),
        scenario=PlanningScenario(service_level=0.80),
    ).decisions.iloc[0]
    high_service = build_inventory_decisions(
        _history(on_hand=5, lead_time=2),
        forecast,
        validation,
        _selections(),
        scenario=PlanningScenario(service_level=0.99),
    ).decisions.iloc[0]
    longer_lead = build_inventory_decisions(
        _history(on_hand=5, lead_time=2),
        forecast,
        validation,
        _selections(),
        scenario=PlanningScenario(service_level=0.99, lead_time_weeks=4),
    ).decisions.iloc[0]
    growth = build_inventory_decisions(
        _history(on_hand=5, lead_time=2),
        forecast,
        validation,
        _selections(),
        scenario=PlanningScenario(service_level=0.99, annual_demand_growth=0.50),
    ).decisions.iloc[0]

    assert high_service["safety_stock"] >= low_service["safety_stock"]
    assert longer_lead["reorder_point"] >= high_service["reorder_point"]
    assert growth["recommended_order_quantity"] >= high_service["recommended_order_quantity"]


def test_long_lead_time_is_labeled_as_approximation() -> None:
    errors = _validation([[1, 2], [2, 3], [3, 4]])
    stock, method = empirical_cumulative_error_quantile(
        errors,
        lead_time_weeks=4,
        service_level=0.95,
    )
    assert stock >= 0
    assert method == "circular block approximation beyond validation horizon"


def test_safety_stock_uses_parametric_floor_and_reports_sample_count() -> None:
    errors = _validation([[-4, -1, -3, -2], [-2, -5, -1, -4], [-3, -1, -6, -2]])
    low = estimate_safety_stock(errors, lead_time_weeks=2, service_level=0.80)
    high = estimate_safety_stock(errors, lead_time_weeks=2, service_level=0.99)
    assert low.empirical_quantile == 0
    assert low.observation_count == 9
    assert low.quantity == low.parametric_floor
    assert high.quantity > low.quantity > 0


def test_insufficient_forecast_is_rejected_without_extrapolation() -> None:
    with pytest.raises(ValueError, match="needs 5 forecast weeks"):
        build_inventory_decisions(
            _history(on_hand=20, lead_time=4),
            _forecast([10, 10, 10, 10]),
            _validation([[0, 0], [0, 0], [0, 0]]),
            _selections(),
        )


def test_zero_demand_has_infinite_coverage_without_invalid_values() -> None:
    result = build_inventory_decisions(
        _history(on_hand=10, lead_time=2),
        _forecast([0, 0, 0]),
        _validation([[0, 0], [0, 0], [0, 0]]),
        _selections(),
    )
    assert np.isinf(result.decisions.iloc[0]["stock_coverage_weeks"])
    assert result.decisions.iloc[0]["overstock_risk"] == "watch"
