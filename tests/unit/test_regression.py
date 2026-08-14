"""
Unit Tests for StatisticalRegressionAnalyzer (OLS).
"""

import pandas as pd
import pytest

from src.statistics.regression import StatisticalRegressionAnalyzer
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df, create_synthetic_operational_df


def test_fit_ols_pm10_legacy():
    """Verify OLS regression analysis on target PM10 with legacy features."""
    df = create_synthetic_legacy_df(num_rows=50)
    analyzer = StatisticalRegressionAnalyzer()

    report = analyzer.fit_ols(df, target_column="PM10", job_id="job-ols-1")

    assert report["target"] == "PM10"
    assert report["n_observations"] == 50
    assert "r2" in report
    assert "r2_adjusted" in report
    assert "coefficients" in report

    coef_df = report["coefficients"]
    assert "variable" in coef_df.columns
    assert "coefficient" in coef_df.columns
    assert "std_error" in coef_df.columns
    assert "t_statistic" in coef_df.columns
    assert "p_value" in coef_df.columns
    assert "ci_lower_95" in coef_df.columns
    assert "ci_upper_95" in coef_df.columns

    # Verify diagnostic dictionary present
    assert "diagnostics" in report
    assert "durbin_watson" in report["diagnostics"]


def test_fit_ols_pm25_operational():
    """Verify OLS regression analysis on target PM2.5 with operational features."""
    df = create_synthetic_operational_df(num_rows=50)
    analyzer = StatisticalRegressionAnalyzer()

    report = analyzer.fit_ols(
        df,
        target_column="PM2.5",
        feature_columns=["Temperature", "Relative_humidity", "Road_width", "Distance"],
        job_id="job-ols-2",
    )

    assert report["target"] == "PM2.5"
    assert report["r2"] is not None
    assert any(report["coefficients"]["variable"] == "Road_width")
