"""
Unit Tests for HypothesisTester (ANOVA & Kruskal-Wallis).
"""

import pandas as pd
import pytest

from src.statistics.hypothesis import HypothesisTester
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_shapiro_normality_check():
    """Verify Shapiro-Wilk normality check behavior."""
    df = create_synthetic_legacy_df(num_rows=30)
    tester = HypothesisTester()
    stat, p_val, is_normal = tester.check_normality(df["PM10"])

    assert isinstance(stat, float)
    assert isinstance(p_val, float)
    assert isinstance(is_normal, bool)


def test_group_comparison_season_pm10():
    """Verify group comparison (ANOVA or Kruskal-Wallis) across SEASON categories."""
    df = create_synthetic_legacy_df(num_rows=50)
    tester = HypothesisTester()
    report = tester.run_group_comparison(df, group_col="SEASON", target_col="PM10", job_id="job-hyp-1")

    assert report["group_column"] == "SEASON"
    assert report["target_column"] == "PM10"
    assert report["test_applied"] in ("One-Way ANOVA", "Kruskal-Wallis H-Test")
    assert "statistic" in report
    assert "p_value" in report
    assert len(report["group_summaries"]) > 1


def test_group_comparison_location_pm25():
    """Verify group comparison across LOCATION categories."""
    df = create_synthetic_legacy_df(num_rows=50)
    tester = HypothesisTester()
    report = tester.run_group_comparison(df, group_col="LOCATION", target_col="PM2.5", job_id="job-hyp-2")

    assert report["group_column"] == "LOCATION"
    assert report["target_column"] == "PM2.5"
    assert "statistic" in report
