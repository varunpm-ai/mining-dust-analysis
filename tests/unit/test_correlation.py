"""
Unit Tests for CorrelationAnalyzer.
"""

import pandas as pd
import pytest

from src.statistics.correlation import CorrelationAnalyzer
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_pairwise_correlation_calculation():
    """Verify pairwise Pearson and Spearman correlation calculation for target PM10."""
    df = create_synthetic_legacy_df(num_rows=40)
    analyzer = CorrelationAnalyzer()
    corr_df = analyzer.analyze_pairwise(df, target="PM10", job_id="job-corr-1")

    assert corr_df is not None
    assert not corr_df.empty
    assert "pearson_r" in corr_df.columns
    assert "spearman_rho" in corr_df.columns
    assert "pearson_pvalue" in corr_df.columns
    assert "spearman_pvalue" in corr_df.columns
    assert any(corr_df["feature"] == "Temperature")


def test_full_correlation_matrices():
    """Verify full matrix computation for Pearson and Spearman coefficients."""
    df = create_synthetic_legacy_df(num_rows=30)
    analyzer = CorrelationAnalyzer()
    p_mat, p_pval, s_mat, s_pval = analyzer.compute_correlation_matrices(df)

    assert p_mat.shape == (8, 8)  # 6 features (including YEAR, WEEK) + 2 targets
    assert s_mat.shape == (8, 8)
    assert p_mat.loc["PM10", "PM10"] == 1.0
    assert s_mat.loc["PM2.5", "PM2.5"] == 1.0
