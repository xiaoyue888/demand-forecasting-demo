import numpy as np
import pandas as pd

from demand_forecasting import build_inventory_decisions, generate_synthetic_data, run_forecasting_core
from demand_forecasting.evaluation import rolling_origin_predictions
from demand_forecasting.models import GlobalHGBForecaster, SeasonalNaiveForecaster


def test_seasonal_naive_uses_only_prior_year_values() -> None:
    data = generate_synthetic_data().query("sku_id == 'SKU_01'").copy()
    train = data.iloc[:-8]
    predicted = SeasonalNaiveForecaster().fit(train).predict(8)
    expected = train["demand"].to_numpy()[-52:-44]
    np.testing.assert_allclose(predicted["forecast"], expected)


def test_global_features_do_not_use_current_target() -> None:
    data = generate_synthetic_data().query("sku_id in ['SKU_01', 'SKU_02']").copy()
    cutoff = data["date"].max() - pd.Timedelta(weeks=8)
    train = data[data["date"] <= cutoff]
    model_a = GlobalHGBForecaster().fit(train)
    prediction_a = model_a.predict(2)

    changed_future = data.copy()
    changed_future.loc[changed_future["date"] > cutoff, "demand"] = 999_999
    model_b = GlobalHGBForecaster().fit(changed_future[changed_future["date"] <= cutoff])
    prediction_b = model_b.predict(2)
    pd.testing.assert_frame_equal(prediction_a, prediction_b)


def test_future_changes_do_not_change_earlier_validation_predictions() -> None:
    data = generate_synthetic_data()
    original = rolling_origin_predictions(data, horizon=8, folds=1, reserve_test=8)
    changed = data.copy()
    final_dates = sorted(changed["date"].unique())[-8:]
    changed.loc[changed["date"].isin(final_dates), "demand"] = 999_999
    altered = rolling_origin_predictions(changed, horizon=8, folds=1, reserve_test=8)
    pd.testing.assert_frame_equal(original, altered)


def test_core_outputs_are_complete_nonnegative_and_reproducible() -> None:
    data = generate_synthetic_data(periods=80)
    first = run_forecasting_core(data)
    second = run_forecasting_core(data)

    assert len(first.selections) == 12
    assert len(first.test_predictions) == 12 * 8
    assert len(first.future_forecast) == 12 * 8
    assert len(first.planning_forecast) == 12 * 13
    numeric = first.future_forecast[["forecast", "lower_80", "upper_80", "lower_95", "upper_95"]]
    assert np.isfinite(numeric.to_numpy()).all()
    assert (numeric >= 0).all().all()
    assert (first.future_forecast["lower_95"] <= first.future_forecast["forecast"]).all()
    assert (first.future_forecast["forecast"] <= first.future_forecast["upper_95"]).all()
    pd.testing.assert_frame_equal(first.selections, second.selections)
    pd.testing.assert_frame_equal(first.future_forecast, second.future_forecast)
    decisions = build_inventory_decisions(
        data,
        first.planning_forecast,
        first.validation_predictions,
        first.selections,
    )
    assert len(decisions.decisions) == 12
    assert decisions.decisions["recommended_order_quantity"].ge(0).all()
