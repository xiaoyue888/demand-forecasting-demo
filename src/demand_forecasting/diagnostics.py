"""Small, descriptive demand-pattern diagnostics for the interface."""

from __future__ import annotations

import numpy as np
import pandas as pd


def diagnose_demand(values: pd.Series, season_length: int = 52) -> dict[str, float | str]:
    demand = values.to_numpy(dtype=float)
    mean = float(demand.mean())
    zero_share = float(np.mean(demand == 0))
    coefficient_of_variation = float(demand.std(ddof=0) / mean) if mean > 0 else 0.0
    seasonal_correlation = (
        float(np.corrcoef(demand[season_length:], demand[:-season_length])[0, 1])
        if len(demand) > season_length and demand.std() > 0
        else np.nan
    )
    recent = demand[-min(13, len(demand)) :]
    recent_trend = float(np.polyfit(np.arange(len(recent)), recent, 1)[0]) if len(recent) > 1 else 0.0

    if zero_share >= 0.20:
        pattern = "Intermittent"
    elif coefficient_of_variation >= 0.60:
        pattern = "Volatile"
    elif np.isfinite(seasonal_correlation) and seasonal_correlation >= 0.35 and abs(recent_trend) >= 0.20:
        pattern = "Trend + seasonal"
    elif np.isfinite(seasonal_correlation) and seasonal_correlation >= 0.35:
        pattern = "Seasonal"
    elif abs(recent_trend) >= 0.20:
        pattern = "Trending"
    else:
        pattern = "Stable"

    return {
        "pattern": pattern,
        "zero_share": zero_share,
        "coefficient_of_variation": coefficient_of_variation,
        "seasonal_correlation": seasonal_correlation,
        "recent_weekly_trend": recent_trend,
    }

