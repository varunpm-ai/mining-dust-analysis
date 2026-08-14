"""
Chart Generator Module for Mine Dust Analysis Platform.

Generates high-resolution publication-quality matplotlib/seaborn charts
for exploratory data analysis, statistical relationships, model comparisons,
and residual diagnostics.
"""

import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger("reporting.chart_generator")

# Configure clean aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 14,
})


class ChartGenerator:
    """
    Generates and saves visual charts for reporting and dashboard presentation.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("artifacts/reports/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_fig(self, fig: plt.Figure, filename: str) -> str:
        """Save matplotlib figure to disk and return absolute filepath."""
        filepath = self.output_dir / filename
        fig.tight_layout()
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return str(filepath.resolve())

    def plot_numerical_distributions(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
        """Generate histogram & KDE distribution grid for numerical features."""
        target_cols = columns or ["Temperature", "Relative_humidity", "RF-CUMULATIVE", "Wind_speed", "PM10", "PM2.5"]
        valid_cols = [c for c in target_cols if c in df.columns]

        if not valid_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, "No numerical data available", ha="center", va="center")
            return self._save_fig(fig, "distributions.png")

        n_cols = 3
        n_rows = (len(valid_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = np.array(axes).flatten()

        for idx, col in enumerate(valid_cols):
            ax = axes[idx]
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#1f77b4", edgecolor="black", alpha=0.6)
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        # Hide unused subplots
        for i in range(len(valid_cols), len(axes)):
            axes[i].set_visible(False)

        return self._save_fig(fig, "numerical_distributions.png")

    def plot_seasonal_analysis(self, df: pd.DataFrame, target: str = "PM10") -> str:
        """Generate boxplot of target concentration by Season."""
        fig, ax = plt.subplots(figsize=(8, 5))
        if "SEASON" in df.columns and target in df.columns:
            sns.boxplot(data=df, x="SEASON", y=target, ax=ax, palette="Blues_d")
            sns.stripplot(data=df, x="SEASON", y=target, ax=ax, color="black", alpha=0.3, jitter=0.2, size=3)
            ax.set_title(f"Seasonal Analysis of {target} Dust Concentration")
            ax.set_xlabel("Season")
            ax.set_ylabel(f"{target} Concentration (µg/m³)")
        else:
            ax.text(0.5, 0.5, f"SEASON or {target} column missing", ha="center", va="center")

        return self._save_fig(fig, f"seasonal_{target.lower()}.png")

    def plot_location_analysis(self, df: pd.DataFrame, target: str = "PM10") -> str:
        """Generate boxplot of target concentration by Location."""
        fig, ax = plt.subplots(figsize=(9, 5))
        if "LOCATION" in df.columns and target in df.columns:
            sns.boxplot(data=df, x="LOCATION", y=target, ax=ax, palette="Purples_d")
            sns.stripplot(data=df, x="LOCATION", y=target, ax=ax, color="black", alpha=0.3, jitter=0.2, size=3)
            ax.set_title(f"Location Analysis of {target} Dust Concentration")
            ax.set_xlabel("Monitoring Location")
            ax.set_ylabel(f"{target} Concentration (µg/m³)")
            plt.xticks(rotation=15)
        else:
            ax.text(0.5, 0.5, f"LOCATION or {target} column missing", ha="center", va="center")

        return self._save_fig(fig, f"location_{target.lower()}.png")

    def plot_monthly_trends(self, df: pd.DataFrame, target: str = "PM10") -> str:
        """Generate monthly mean trends line chart."""
        fig, ax = plt.subplots(figsize=(9, 4))
        if "MONTH" in df.columns and target in df.columns:
            monthly_agg = df.groupby("MONTH")[target].mean().reset_index()
            sns.barplot(data=monthly_agg, x="MONTH", y=target, ax=ax, palette="Viridis")
            ax.set_title(f"Monthly Average {target} Concentration")
            ax.set_xlabel("Month")
            ax.set_ylabel(f"Mean {target} (µg/m³)")
            plt.xticks(rotation=30)
        else:
            ax.text(0.5, 0.5, f"MONTH or {target} column missing", ha="center", va="center")

        return self._save_fig(fig, f"monthly_{target.lower()}.png")

    def plot_weather_vs_target(self, df: pd.DataFrame, weather_col: str, target_col: str = "PM10") -> str:
        """Generate scatter plot with trendline between a weather feature and target."""
        fig, ax = plt.subplots(figsize=(7, 5))
        if weather_col in df.columns and target_col in df.columns:
            sns.regplot(
                data=df,
                x=weather_col,
                y=target_col,
                ax=ax,
                scatter_kws={"alpha": 0.4, "color": "#2c3e50"},
                line_kws={"color": "#e74c3c", "linewidth": 2},
            )
            ax.set_title(f"Relationship: {weather_col} vs {target_col}")
            ax.set_xlabel(weather_col)
            ax.set_ylabel(f"{target_col} (µg/m³)")
        else:
            ax.text(0.5, 0.5, f"{weather_col} or {target_col} missing", ha="center", va="center")

        clean_col = weather_col.lower().replace("-", "_").replace(" ", "_")
        return self._save_fig(fig, f"weather_{clean_col}_vs_{target_col.lower()}.png")

    def plot_correlation_heatmap(self, corr_df: pd.DataFrame, title: str = "Correlation Heatmap") -> str:
        """Generate heatmap visualization of correlation matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        if not corr_df.empty:
            sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=True, square=True)
            ax.set_title(title)
        else:
            ax.text(0.5, 0.5, "Empty correlation matrix", ha="center", va="center")

        return self._save_fig(fig, "correlation_heatmap.png")

    def plot_model_comparison(self, metrics_df: pd.DataFrame, primary_metric: str = "RMSE") -> str:
        """Generate bar chart comparing model metrics."""
        fig, ax = plt.subplots(figsize=(9, 5))
        if not metrics_df.empty and "model_name" in metrics_df.columns and primary_metric in metrics_df.columns:
            sorted_df = metrics_df.sort_values(by=primary_metric, ascending=True)
            sns.barplot(data=sorted_df, x="model_name", y=primary_metric, ax=ax, palette="Blues_r")
            ax.set_title(f"Model Comparison by {primary_metric} (Lower is Better)")
            ax.set_xlabel("Regression Model")
            ax.set_ylabel(primary_metric)
            plt.xticks(rotation=25)
            
            # Annotate bar values
            for p in ax.patches:
                height = p.get_height()
                if not np.isnan(height):
                    ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2., height),
                                ha="center", va="bottom", fontsize=8, xytext=(0, 3), textcoords="offset points")
        else:
            ax.text(0.5, 0.5, "Model metrics unavailable", ha="center", va="center")

        return self._save_fig(fig, "model_comparison.png")

    def plot_predicted_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, target: str) -> str:
        """Generate scatter plot of predicted vs actual values with 45-degree reference line."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_true, y_pred, alpha=0.5, color="#2980b9", edgecolors="none")

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal 1:1 Line", linewidth=2)

        ax.set_title(f"Predicted vs Actual ({model_name} - {target})")
        ax.set_xlabel(f"Actual {target} (µg/m³)")
        ax.set_ylabel(f"Predicted {target} (µg/m³)")
        ax.legend()

        return self._save_fig(fig, f"pred_vs_act_{model_name.lower().replace(' ', '_')}_{target.lower()}.png")

    def plot_residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str, target: str) -> str:
        """Generate residual scatter plot and residual distribution histogram."""
        residuals = y_true - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.5, color="#27ae60")
        ax1.axhline(0, color="red", linestyle="--", linewidth=1.5)
        ax1.set_title(f"Residuals vs Predicted ({model_name})")
        ax1.set_xlabel("Predicted Values")
        ax1.set_ylabel("Residuals")

        # Residual Distribution
        sns.histplot(residuals, kde=True, ax=ax2, color="#8e44ad")
        ax2.set_title("Residual Distribution")
        ax2.set_xlabel("Residual")

        return self._save_fig(fig, f"residuals_{model_name.lower().replace(' ', '_')}_{target.lower()}.png")

    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str) -> str:
        """Generate horizontal bar plot of feature importances or coefficients."""
        fig, ax = plt.subplots(figsize=(8, 5))
        if not importance_df.empty and "feature" in importance_df.columns and "importance" in importance_df.columns:
            sorted_df = importance_df.sort_values(by="importance", ascending=True)
            sns.barplot(data=sorted_df, x="importance", y="feature", ax=ax, palette="teal")
            ax.set_title(f"Feature Importance ({model_name})")
            ax.set_xlabel("Importance Score")
            ax.set_ylabel("Feature")
        else:
            ax.text(0.5, 0.5, "Feature importance data unavailable", ha="center", va="center")

        return self._save_fig(fig, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
