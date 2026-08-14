"""
Unit Tests for ModelSelector.
"""

import pandas as pd
import pytest

from src.modeling.selector import ModelSelector, BestModelResult
from src.utils.exceptions import ModelSelectionError


def test_model_selector_deterministic_ranking():
    """Verify deterministic ranking with controlled synthetic model metrics."""
    synthetic_results = {
        "random_forest": {
            "model_name": "Random Forest Regressor",
            "model_instance": "dummy_rf_instance",
            "metrics": {"MAE": 5.2, "RMSE": 8.1, "R2": 0.85, "R": 0.92},
        },
        "xgboost": {
            "model_name": "XGBoost Regressor",
            "model_instance": "dummy_xgb_instance",
            "metrics": {"MAE": 4.1, "RMSE": 6.3, "R2": 0.91, "R": 0.95},
        },
        "linear_regression": {
            "model_name": "Linear Regression",
            "model_instance": "dummy_lr_instance",
            "metrics": {"MAE": 8.9, "RMSE": 12.4, "R2": 0.71, "R": 0.84},
        },
    }

    selector = ModelSelector(primary_metric="RMSE", secondary_metric="MAE", tertiary_metric="R")
    best_result = selector.select_best_model(synthetic_results)

    assert isinstance(best_result, BestModelResult)
    assert best_result.best_model_key == "xgboost"
    assert best_result.best_model_name == "XGBoost Regressor"
    assert best_result.best_model_instance == "dummy_xgb_instance"
    assert best_result.best_model_metrics["RMSE"] == 6.3

    # Check ranked comparison table
    comp_table = best_result.comparison_table
    assert len(comp_table) == 3
    assert comp_table.iloc[0]["model_key"] == "xgboost"
    assert comp_table.iloc[1]["model_key"] == "random_forest"
    assert comp_table.iloc[2]["model_key"] == "linear_regression"


def test_model_selector_custom_primary_metric_r2():
    """Verify selecting best model using R2 (higher is better)."""
    synthetic_results = {
        "model_a": {
            "model_name": "Model A",
            "model_instance": "inst_a",
            "metrics": {"MAE": 6.0, "RMSE": 9.0, "R2": 0.80, "R": 0.89},
        },
        "model_b": {
            "model_name": "Model B",
            "model_instance": "inst_b",
            "metrics": {"MAE": 5.0, "RMSE": 8.5, "R2": 0.95, "R": 0.97},
        },
    }

    selector = ModelSelector(primary_metric="R2")
    best_result = selector.select_best_model(synthetic_results)

    assert best_result.best_model_key == "model_b"
    assert best_result.best_model_metrics["R2"] == 0.95


def test_model_selector_empty_results_raises_exception():
    """Verify empty dictionary raises ModelSelectionError."""
    selector = ModelSelector()
    with pytest.raises(ModelSelectionError):
        selector.select_best_model({})
