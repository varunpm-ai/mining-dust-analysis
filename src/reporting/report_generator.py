"""
Executive HTML Report Generator for Mine Dust Analysis Platform.

Generates standalone, beautifully styled executive reports for non-technical mining clients.
Includes inline embedded charts, statistical summary tables, and ML performance metrics.
"""

import base64
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from jinja2 import Template

from src.utils.exceptions import ReportGenerationError
from src.utils.logger import setup_logger

logger = setup_logger("reporting.report_generator")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mine Dust Analysis & Predictive Modeling Executive Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            color: #2c3e50;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 1100px;
            margin: 30px auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .header {
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 28px;
        }
        .header .meta {
            font-size: 14px;
            color: #7f8c8d;
        }
        h2 {
            color: #2c3e50;
            border-left: 4px solid #3498db;
            padding-left: 12px;
            margin-top: 35px;
            font-size: 20px;
        }
        p, li {
            color: #34495e;
            font-size: 15px;
        }
        .card {
            background: #ebf5fb;
            border: 1px solid #d6eaf8;
            border-radius: 6px;
            padding: 20px;
            margin: 20px 0;
        }
        .card-title {
            font-weight: bold;
            color: #1b4f72;
            margin-bottom: 8px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 14px;
        }
        th, td {
            text-align: left;
            padding: 10px 14px;
            border-bottom: 1px solid #e2e8f0;
        }
        th {
            background-color: #f1f5f9;
            color: #1e293b;
            font-weight: 600;
        }
        tr:hover {
            background-color: #f8fafc;
        }
        .chart-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 25px 0;
        }
        .chart-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 15px;
            text-align: center;
        }
        .chart-card img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            font-size: 13px;
            color: #94a3b8;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⛏️ Mine Dust Analysis & Predictive Modeling Executive Report</h1>
            <div class="meta">
                <strong>Platform Version:</strong> {{ version }} | 
                <strong>Execution Job ID:</strong> {{ job_id }} | 
                <strong>Report Generated:</strong> {{ timestamp }}
            </div>
        </div>

        <div class="card">
            <div class="card-title">Executive Summary</div>
            <p>This report details the automated data quality validation, statistical regression analysis, and machine learning model evaluations conducted on the uploaded mining operational dataset. All transformations were performed deterministically without silent data deletion.</p>
        </div>

        <h2>1. Data Ingestion & Quality Audit</h2>
        <table>
            <tr><th>Metric</th><th>Ingested Value</th></tr>
            <tr><td>Total Initial Dataset Rows</td><td>{{ quality_report.initial_rows }}</td></tr>
            <tr><td>Duplicate Records Removed</td><td>{{ quality_report.duplicate_rows_removed }}</td></tr>
            <tr><td>Final Validated Rows</td><td>{{ quality_report.final_rows }}</td></tr>
            <tr><td>Final Column Count</td><td>{{ quality_report.final_cols }}</td></tr>
        </table>

        <h2>2. Statistical Correlation & Regression Analysis</h2>
        <p>Statistical relationships between weather conditions, operational parameters, and dust concentrations (PM10 / PM2.5):</p>
        {% if correlation_html %}
            {{ correlation_html }}
        {% endif %}

        <h2>3. Visual Exploratory Analysis</h2>
        <div class="chart-grid">
            {% for chart_name, chart_uri in charts.items() %}
                <div class="chart-card">
                    <img src="{{ chart_uri }}" alt="{{ chart_name }}">
                </div>
            {% endfor %}
        </div>

        <h2>4. Machine Learning Model Comparisons</h2>
        <p>Multiple regression algorithms were evaluated using cross-validation methodology to select the winning model:</p>
        {% if model_comparison_html %}
            {{ model_comparison_html }}
        {% endif %}

        <div class="card" style="background: #e8f8f5; border-color: #a3e4d7;">
            <div class="card-title" style="color: #117a65;">Best Performing Model Selected</div>
            <p><strong>Winning Model:</strong> {{ winning_model_name }}</p>
            <p><strong>Primary Selection Metric ({{ primary_metric }}):</strong> {{ winning_metric_val }}</p>
        </div>

        <h2>5. Operational Recommendations for Mine Management</h2>
        <ul>
            <li><strong>Dust Mitigation:</strong> Implement active water spraying on Haul Roads during dry summer months when wind speeds exceed thresholds.</li>
            <li><strong>Monitoring:</strong> Maintain continuous monitoring at high-risk locations identified in the spatial breakdown.</li>
            <li><strong>Model Deployment:</strong> Deploy the versioned winning model artifact for daily operational forecasting.</li>
        </ul>

        <div class="footer">
            Mine Dust Analysis Production MLOps Platform v{{ version }} &copy; Automated Report Generation System
        </div>
    </div>
</body>
</html>
"""


class ReportGenerator:
    """
    Automated HTML report generator integrating pipeline metadata, statistical summary tables, and embedded charts.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _file_to_base64(self, file_path: str) -> str:
        """Convert an image file on disk to base64 Data URI string for HTML embedding."""
        path = Path(file_path)
        if not path.exists():
            return ""
        with open(path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode("utf-8")
        ext = path.suffix.lower().replace(".", "")
        mime = "image/png" if ext == "png" else "image/jpeg"
        return f"data:{mime};base64,{encoded}"

    def generate_html_report(
        self,
        quality_report: Dict[str, Any],
        correlation_df: Optional[pd.DataFrame] = None,
        model_comparison_df: Optional[pd.DataFrame] = None,
        chart_paths: Optional[Dict[str, str]] = None,
        winning_model_name: str = "XGBoost Regressor",
        winning_metric_val: float = 0.0,
        primary_metric: str = "RMSE",
        job_id: str = "N/A",
    ) -> str:
        """
        Generate and save complete executive HTML report file.

        Returns:
            str: Path to generated HTML report file.
        """
        try:
            # Base64 encode charts for embedded single-file HTML portability
            embedded_charts: Dict[str, str] = {}
            if chart_paths:
                for name, path_str in chart_paths.items():
                    uri = self._file_to_base64(path_str)
                    if uri:
                        embedded_charts[name] = uri

            # Convert DataFrames to HTML tables
            corr_html = correlation_df.to_html(index=False, classes="table") if correlation_df is not None and not correlation_df.empty else ""
            model_html = model_comparison_df.to_html(index=False, classes="table") if model_comparison_df is not None and not model_comparison_df.empty else ""

            import datetime
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            template = Template(HTML_TEMPLATE)
            rendered_html = template.render(
                version="2.0.0",
                job_id=job_id,
                timestamp=timestamp_str,
                quality_report=quality_report,
                correlation_html=corr_html,
                model_comparison_html=model_html,
                charts=embedded_charts,
                winning_model_name=winning_model_name,
                winning_metric_val=f"{winning_metric_val:.4f}",
                primary_metric=primary_metric,
            )

            report_file = self.output_dir / f"executive_report_{job_id}.html"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(rendered_html)

            logger.info(f"Generated executive HTML report at {report_file}", extra={"job_id": job_id})
            return str(report_file.resolve())
        except Exception as e:
            logger.error(f"HTML report generation failed: {e}", exc_info=True, extra={"job_id": job_id})
            raise ReportGenerationError(
                message=f"Failed to render HTML report template: {e}",
                user_message="Executive report generation failed.",
                details={"error": str(e)},
            )
