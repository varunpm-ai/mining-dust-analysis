"""
Unit Tests for ChartGenerator.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.reporting.chart_generator import ChartGenerator
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_chart_generator_distributions(tmp_path):
    """Verify plotting numerical distributions saves PNG file."""
    df = create_synthetic_legacy_df(num_rows=30)
    chart_gen = ChartGenerator(output_dir=tmp_path)
    file_path = chart_gen.plot_numerical_distributions(df)

    assert Path(file_path).exists()
    assert file_path.endswith(".png")


def test_chart_generator_seasonal_and_location(tmp_path):
    """Verify plotting seasonal and location boxplots."""
    df = create_synthetic_legacy_df(num_rows=30)
    chart_gen = ChartGenerator(output_dir=tmp_path)

    seasonal_path = chart_gen.plot_seasonal_analysis(df, target="PM10")
    location_path = chart_gen.plot_location_analysis(df, target="PM10")

    assert Path(seasonal_path).exists()
    assert Path(location_path).exists()


def test_chart_generator_regression_diagnostics_charts(tmp_path):
    """Verify plotting predicted vs actual and residual diagnostic charts."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    y_pred = np.array([52.0, 58.0, 71.0, 79.0, 92.0])

    chart_gen = ChartGenerator(output_dir=tmp_path)
    pred_path = chart_gen.plot_predicted_vs_actual(y_true, y_pred, model_name="XGBoost", target="PM10")
    resid_path = chart_gen.plot_residual_analysis(y_true, y_pred, model_name="XGBoost", target="PM10")

    assert Path(pred_path).exists()
    assert Path(resid_path).exists()
