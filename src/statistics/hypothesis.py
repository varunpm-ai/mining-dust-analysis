"""
Hypothesis Testing Module for Mine Dust Data Analysis.

Implements Shapiro-Wilk normality tests, Levene's variance homogeneity tests,
One-Way ANOVA, and non-parametric Kruskal-Wallis H-tests across categorical groupings.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import shapiro, levene, f_oneway, kruskal

from src.utils.exceptions import StatisticalAnalysisError
from src.utils.logger import setup_logger

logger = setup_logger("statistics.hypothesis")


class HypothesisTester:
    """
    Automated statistical hypothesis tester (ANOVA & Kruskal-Wallis tests).
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def check_normality(self, series: pd.Series, sample_size: int = 500) -> Tuple[float, float, bool]:
        """
        Shapiro-Wilk test for normality.

        Returns:
            Tuple[float, float, bool]: (statistic, p_value, is_normal)
        """
        clean_series = series.dropna()
        if len(clean_series) < 3:
            return 0.0, 1.0, False

        # Subsample if large dataset to avoid Shapiro-Wilk N > 5000 limitation
        if len(clean_series) > sample_size:
            clean_series = clean_series.sample(n=sample_size, random_state=42)

        stat, p = shapiro(clean_series)
        is_normal = bool(p >= self.alpha)
        return float(round(stat, 4)), float(round(p, 6)), is_normal

    def run_group_comparison(
        self,
        df: pd.DataFrame,
        group_col: str,
        target_col: str,
        job_id: str = "N/A",
    ) -> Dict[str, Any]:
        """
        Compare target distributions across categorical groups (e.g. SEASON or LOCATION).
        Automatically evaluates normality/homoscedasticity and applies ANOVA or Kruskal-Wallis.

        Args:
            df (pd.DataFrame): Dataset DataFrame.
            group_col (str): Categorical grouping column (e.g. "SEASON" or "LOCATION").
            target_col (str): Continuous numerical target (e.g. "PM10" or "PM2.5").
            job_id (str): Execution job ID.

        Returns:
            Dict[str, Any]: Structured test report with test name, statistic, p-value, group stats.
        """
        extra_ctx = {"job_id": job_id}

        if group_col not in df.columns or target_col not in df.columns:
            raise StatisticalAnalysisError(
                message=f"Specified columns '{group_col}' or '{target_col}' not found in DataFrame.",
                user_message=f"Missing columns for group comparison: '{group_col}', '{target_col}'.",
            )

        valid_df = df[[group_col, target_col]].dropna()
        groups = [group[target_col].values for _, group in valid_df.groupby(group_col) if len(group) >= 2]

        if len(groups) < 2:
            raise StatisticalAnalysisError(
                message=f"Insufficient group categories in '{group_col}' for hypothesis testing.",
                user_message=f"Grouping column '{group_col}' must contain at least 2 categories.",
            )

        # Calculate group summary statistics
        group_summaries = (
            valid_df.groupby(group_col)[target_col]
            .agg(["count", "mean", "std", "median"])
            .reset_index()
            .to_dict(orient="records")
        )

        # 1. Check Normality of overall target
        norm_stat, norm_p, is_normal = self.check_normality(valid_df[target_col])

        # 2. Check Homogeneity of Variance (Levene Test)
        lev_stat, lev_p = levene(*groups)
        is_homogenous = bool(lev_p >= self.alpha)

        # 3. Select appropriate hypothesis test
        # If normal and equal variance -> One-Way ANOVA; else Kruskal-Wallis H-test
        if is_normal and is_homogenous:
            test_name = "One-Way ANOVA"
            stat, p_val = f_oneway(*groups)
        else:
            test_name = "Kruskal-Wallis H-Test"
            stat, p_val = kruskal(*groups)

        is_significant = bool(p_val < self.alpha)

        result = {
            "group_column": group_col,
            "target_column": target_col,
            "test_applied": test_name,
            "statistic": float(round(stat, 4)),
            "p_value": float(round(p_val, 6)),
            "is_significant": is_significant,
            "normality_pvalue": norm_p,
            "is_normal": is_normal,
            "homogeneity_pvalue": float(round(lev_p, 6)),
            "is_homogenous": is_homogenous,
            "group_summaries": group_summaries,
        }

        logger.info(
            f"Executed hypothesis test ({test_name}) on '{target_col}' by '{group_col}': p-value = {p_val:.6f}",
            extra=extra_ctx,
        )

        return result
