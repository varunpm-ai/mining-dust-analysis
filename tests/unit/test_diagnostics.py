"""
Unit Tests for VIF Calculator and Regression Diagnostics.
"""

import pandas as pd
import pytest
import statsmodels.api as sm

from src.statistics.diagnostics import VarianceInflationFactorCalculator, RegressionDiagnostics
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_vif_calculation():
    """Verify Variance Inflation Factor (VIF) calculation across numerical features."""
    df = create_synthetic_legacy_df(num_rows=40)
    features = df[["Temperature", "Relative_humidity", "RF-CUMULATIVE", "Wind_speed"]]

    vif_df = VarianceInflationFactorCalculator.compute_vif(features)
    assert not vif_df.empty
    assert "feature" in vif_df.columns
    assert "vif" in vif_df.columns
    assert "multicollinearity_risk" in vif_df.columns
    assert len(vif_df) == 4


def test_regression_diagnostics():
    """Verify residual diagnostics extraction on fitted OLS model."""
    df = create_synthetic_legacy_df(num_rows=40)
    X = sm.add_constant(df[["Temperature", "Relative_humidity", "RF-CUMULATIVE", "Wind_speed"]])
    y = df["PM10"]

    model_fit = sm.OLS(y, X).fit()
    diag = RegressionDiagnostics.evaluate_residuals(model_fit)

    assert "durbin_watson" in diag
    assert "jarque_bera_stat" in diag
    assert "jarque_bera_pvalue" in diag
    assert "omnibus_stat" in diag
    assert "condition_number" in diag
    assert isinstance(diag["r2"], float)
