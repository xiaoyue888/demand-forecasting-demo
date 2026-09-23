"""Validation for the deliberately small public CSV contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "date",
    "sku_id",
    "demand",
    "current_on_hand",
    "lead_time_weeks",
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


class DataValidationError(ValueError):
    """Raised with all actionable contract violations found in a dataset."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


def _integer_series(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce")
    return bool(numeric.notna().all() and np.isfinite(numeric).all() and (numeric % 1 == 0).all())


def validate_demand_data(
    frame: pd.DataFrame,
    *,
    min_history: int = 80,
    max_skus: int = 100,
    max_history: int = 260,
) -> pd.DataFrame:
    """Validate and normalize weekly multi-SKU demand without silently imputing gaps."""

    issues: list[ValidationIssue] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    unexpected = [column for column in frame.columns if column not in REQUIRED_COLUMNS]
    if missing:
        issues.append(ValidationIssue("missing_columns", f"Missing required columns: {missing}."))
    if unexpected:
        issues.append(ValidationIssue("unexpected_columns", f"Unexpected columns: {unexpected}."))
    if issues:
        raise DataValidationError(issues)

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    parsed_dates = pd.to_datetime(data["date"], errors="coerce")
    if parsed_dates.isna().any():
        rows = (parsed_dates[parsed_dates.isna()].index + 2).tolist()
        issues.append(ValidationIssue("invalid_date", f"Invalid date values on CSV rows {rows}."))
    else:
        data["date"] = parsed_dates.dt.normalize()
        bad_weekdays = data.index[data["date"].dt.weekday != 0].tolist()
        if bad_weekdays:
            rows = [row + 2 for row in bad_weekdays]
            issues.append(ValidationIssue("non_monday_date", f"Dates must be Mondays; check CSV rows {rows}."))

    normalized_skus = data["sku_id"].astype("string").str.strip()
    invalid_skus = normalized_skus.isna() | normalized_skus.eq("")
    if invalid_skus.any():
        rows = (invalid_skus[invalid_skus].index + 2).tolist()
        issues.append(ValidationIssue("invalid_sku", f"SKU identifiers are blank on CSV rows {rows}."))
    data["sku_id"] = normalized_skus

    for column, lower, upper in (
        ("demand", 0, None),
        ("current_on_hand", 0, None),
        ("lead_time_weeks", 1, 12),
    ):
        if not _integer_series(data[column]):
            issues.append(ValidationIssue("invalid_integer", f"{column} must contain finite integers only."))
            continue
        data[column] = pd.to_numeric(data[column]).astype("int64")
        invalid = data[column] < lower
        if upper is not None:
            invalid |= data[column] > upper
        if invalid.any():
            rows = (invalid[invalid].index + 2).tolist()
            bounds = f"{lower} through {upper}" if upper is not None else f"at least {lower}"
            issues.append(ValidationIssue("out_of_range", f"{column} must be {bounds}; check CSV rows {rows}."))

    if issues:
        raise DataValidationError(issues)

    duplicates = data.duplicated(["sku_id", "date"], keep=False)
    if duplicates.any():
        pairs = data.loc[duplicates, ["sku_id", "date"]].astype(str).drop_duplicates().values.tolist()
        issues.append(ValidationIssue("duplicate_period", f"Duplicate SKU/date pairs: {pairs}."))

    sku_count = data["sku_id"].nunique()
    if sku_count > max_skus:
        issues.append(ValidationIssue("too_many_skus", f"Found {sku_count} SKUs; maximum is {max_skus}."))

    calendars: dict[str, tuple[pd.Timestamp, ...]] = {}
    for sku_id, group in data.groupby("sku_id", sort=True):
        ordered = group.sort_values("date")
        count = len(ordered)
        if count < min_history:
            issues.append(ValidationIssue("short_history", f"{sku_id} has {count} weeks; minimum is {min_history}."))
        if count > max_history:
            issues.append(ValidationIssue("long_history", f"{sku_id} has {count} weeks; maximum is {max_history}."))
        expected = pd.date_range(ordered["date"].min(), ordered["date"].max(), freq="W-MON")
        actual = pd.DatetimeIndex(ordered["date"])
        if not actual.equals(expected):
            missing_dates = expected.difference(actual).strftime("%Y-%m-%d").tolist()
            issues.append(
                ValidationIssue(
                    "calendar_gap",
                    f"{sku_id} must have consecutive weekly observations; missing {missing_dates[:8]}.",
                )
            )
        for column in ("current_on_hand", "lead_time_weeks"):
            if ordered[column].nunique() != 1:
                issues.append(ValidationIssue("inconsistent_snapshot", f"{column} must be constant for {sku_id}."))
        calendars[str(sku_id)] = tuple(actual)

    if calendars:
        reference = next(iter(calendars.values()))
        misaligned = [sku for sku, calendar in calendars.items() if calendar != reference]
        if misaligned:
            issues.append(ValidationIssue("misaligned_calendar", f"All SKUs must share one calendar; check {misaligned}."))

    if issues:
        raise DataValidationError(issues)

    return data.sort_values(["date", "sku_id"], ignore_index=True)

