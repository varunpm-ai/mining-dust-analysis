"""
Unit Tests for ModelTrainer Engine.
"""

import pandas as pd
import pytest

from src.modeling.trainer import ModelTrainer
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df, create_synthetic_operational_df


def test_train_all_models_pm10_legacy():
    """Verify training all 7 regression models using cross-validation on target PM10."""
    df = create_synthetic_legacy_df(num_rows=40)
    trainer = ModelTrainer()

    report = trainer.train_models_cv(df, target_column="PM10", n_splits=3, job_id="job-train-1")

    assert report["target"] == "PM10"
    assert "comparison_df" in report
    assert "models" in report

    comp_df = report["comparison_df"]
    assert len(comp_df) == 7  # All 7 models trained
    assert "MAE" in comp_df.columns
    assert "RMSE" in comp_df.columns
    assert "R2" in comp_df.columns
    assert "R" in comp_df.columns

    # Verify each model key present in models dictionary
    for model_key in [
        "linear_regression",
        "decision_tree",
        "random_forest",
        "bagging",
        "svr",
        "xgboost",
        "voting",
    ]:
        assert model_key in report["models"]
        model_data = report["models"][model_key]
        assert hasattr(model_data["model_instance"], "predict")
        assert "metrics" in model_data
        assert "oof_predictions" in model_data


def test_train_all_models_pm25_operational():
    """Verify training all regression models on target PM2.5 with operational dataset."""
    df = create_synthetic_operational_df(num_rows=40)
    trainer = ModelTrainer()

    report = trainer.train_models_cv(df, target_column="PM2.5", n_splits=3, job_id="job-train-2")

    assert report["target"] == "PM2.5"
    assert len(report["comparison_df"]) == 7
