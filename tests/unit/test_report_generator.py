"""
Unit Tests for Executive ReportGenerator.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.reporting.chart_generator import ChartGenerator
from src.reporting.report_generator import ReportGenerator
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_generate_html_report(tmp_path):
    """Verify generating complete executive HTML report file."""
    df = create_synthetic_legacy_df(num_rows=20)
    chart_gen = ChartGenerator(output_dir=tmp_path / "charts")
    dist_chart = chart_gen.plot_numerical_distributions(df)

    quality_report = {
        "initial_rows": 20,
        "duplicate_rows_removed": 0,
        "final_rows": 20,
        "final_cols": 11,
    }

    corr_df = pd.DataFrame({
        "feature": ["Temperature", "Relative_humidity"],
        "pearson_r": [0.25, -0.15],
        "pearson_pvalue": [0.01, 0.04],
    })

    model_df = pd.DataFrame({
        "model_name": ["Random Forest", "XGBoost"],
        "RMSE": [12.4, 10.2],
        "MAE": [9.1, 7.8],
        "R2": [0.72, 0.81],
    })

    report_gen = ReportGenerator(output_dir=tmp_path)
    html_path = report_gen.generate_html_report(
        quality_report=quality_report,
        correlation_df=corr_df,
        model_comparison_df=model_df,
        chart_paths={"Distributions": dist_chart},
        winning_model_name="XGBoost Regressor",
        winning_metric_val=10.2,
        job_id="job-report-1",
    )

    assert Path(html_path).exists()
    content = Path(html_path).read_text(encoding="utf-8")
    assert "Mine Dust Analysis & Predictive Modeling Executive Report" in content
    assert "job-report-1" in content
    assert "XGBoost Regressor" in content
    assert "data:image/png;base64," in content  # Embedded base64 image
