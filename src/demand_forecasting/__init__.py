"""Leakage-safe forecasting core for the portfolio demo."""

from .contracts import DataValidationError, validate_demand_data
from .diagnostics import diagnose_demand
from .inventory import InventoryDecisionResult, PlanningScenario, build_inventory_decisions
from .service import ForecastingCoreResult, run_forecasting_core
from .synthetic import generate_synthetic_data

__all__ = [
    "DataValidationError",
    "ForecastingCoreResult",
    "InventoryDecisionResult",
    "PlanningScenario",
    "build_inventory_decisions",
    "diagnose_demand",
    "generate_synthetic_data",
    "run_forecasting_core",
    "validate_demand_data",
]
