"""Deterministic, public synthetic demand scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import validate_demand_data


def generate_synthetic_data(
    seed: int = 20260915,
    periods: int = 156,
    *,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create aligned weekly histories covering several recognizable demand regimes."""

    if periods < 80:
        raise ValueError("Synthetic histories must contain at least 80 weeks.")
    rng = np.random.default_rng(seed)
    if end_date is None:
        today = pd.Timestamp.today().normalize()
        end = today - pd.Timedelta(days=today.weekday() + 7)
    else:
        end = pd.Timestamp(end_date).normalize()
        end -= pd.Timedelta(days=end.weekday())
    dates = pd.date_range(end=end, periods=periods, freq="W-MON")
    records: list[dict[str, object]] = []

    for index in range(12):
        week = np.arange(periods, dtype=float)
        base = 18.0 + 4.0 * index
        annual = (5.0 + index % 4) * np.sin(2 * np.pi * week / 52 + index / 3)
        trend = (0.04 + 0.015 * (index % 3)) * week if index in {4, 5, 6} else 0.0
        noise = rng.normal(0, 2.5 + index * 0.25, periods)
        demand = base + annual + trend + noise

        if index in {7, 8}:
            active = rng.random(periods) < (0.42 if index == 7 else 0.25)
            demand = active * rng.gamma(shape=2.2, scale=base / 2.2, size=periods)
        elif index in {9, 10}:
            spikes = rng.random(periods) < 0.07
            demand = demand + spikes * rng.integers(25, 70, periods)
        elif index == 11:
            demand = np.full(periods, 26.0) + rng.normal(0, 0.8, periods)

        demand = np.maximum(0, np.rint(demand)).astype(int)
        lead_time = 1 + (index * 3) % 8
        recent_mean = max(1.0, float(np.mean(demand[-13:])))
        on_hand = int(np.ceil(recent_mean * (lead_time + (index % 4) / 2 + 0.5)))
        sku_id = f"SKU_{index + 1:02d}"
        records.extend(
            {
                "date": date,
                "sku_id": sku_id,
                "demand": int(quantity),
                "current_on_hand": on_hand,
                "lead_time_weeks": lead_time,
            }
            for date, quantity in zip(dates, demand, strict=True)
        )

    return validate_demand_data(pd.DataFrame.from_records(records))
