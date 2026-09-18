import pandas as pd
import pytest

from demand_forecasting import DataValidationError, generate_synthetic_data, validate_demand_data


def test_synthetic_data_is_valid_and_aligned() -> None:
    data = generate_synthetic_data()
    validated = validate_demand_data(data)
    assert validated["sku_id"].nunique() == 12
    assert validated.groupby("sku_id").size().eq(156).all()


def test_validation_rejects_missing_week_instead_of_assuming_zero() -> None:
    data = generate_synthetic_data().drop(index=25).reset_index(drop=True)
    with pytest.raises(DataValidationError) as exc_info:
        validate_demand_data(data)
    assert any(issue.code == "calendar_gap" for issue in exc_info.value.issues)


def test_validation_rejects_inconsistent_snapshot() -> None:
    data = generate_synthetic_data()
    data.loc[data["sku_id"] == "SKU_01", "current_on_hand"] = range(156)
    with pytest.raises(DataValidationError) as exc_info:
        validate_demand_data(data)
    assert any(issue.code == "inconsistent_snapshot" for issue in exc_info.value.issues)


def test_validation_rejects_unknown_columns() -> None:
    data = generate_synthetic_data().assign(customer_name="private")
    with pytest.raises(DataValidationError) as exc_info:
        validate_demand_data(data)
    assert any(issue.code == "unexpected_columns" for issue in exc_info.value.issues)


def test_synthetic_generation_is_reproducible() -> None:
    pd.testing.assert_frame_equal(generate_synthetic_data(), generate_synthetic_data())

