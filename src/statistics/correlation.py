"""
Correlation Analysis Module for Mine Dust Engine.

Computes Pearson and Spearman correlation matrices, pair-wise correlation tables,
and statistical significance p-values.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.utils.exceptions import StatisticalAnalysisError
from src.utils.logger import setup_logger

logger = setup_logger("statistics.correlation")


class CorrelationAnalyzer:
    """
    Computes Pearson and Spearman correlation metrics and significance.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha (float): Significance threshold level (default 0.05).
        """
        self.alpha = alpha

    def analyze_pairwise(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        target: str = "PM10",
        job_id: str = "N/A",
    ) -> pd.DataFrame:
        """
        Compute pairwise Pearson and Spearman correlations between features and a target column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            features (List[str], optional): List of numeric feature names.
            target (str): Target column name e.g. "PM10" or "PM2.5".
            job_id (str): Execution job ID.

        Returns:
            pd.DataFrame: Structured correlation table with r, rho, p-values, and significance flags.
        """
        extra_ctx = {"job_id": job_id}
        if target not in df.columns:
            raise StatisticalAnalysisError(
                message=f"Target column '{target}' missing for correlation analysis.",
                user_message=f"Target column '{target}' is missing from the dataset.",
            )

        if not features:
            numeric_df = df.select_dtypes(include=[np.number])
            features = [c for c in numeric_df.columns if c != target]

        if not features:
            raise StatisticalAnalysisError(
                message="No numerical feature columns available for correlation analysis.",
                user_message="No numerical features found in dataset to evaluate correlation.",
            )

        results: List[Dict[str, Any]] = []
        target_series = df[target].dropna()

        for feat in features:
            if feat not in df.columns:
                continue

            # Drop missing pairs for valid calculation
            paired = df[[feat, target]].dropna()
            if len(paired) < 3:
                continue

            # Pearson correlation
            pr_val, pr_p = pearsonr(paired[feat], paired[target])
            # Spearman rank correlation
            sr_val, sr_p = spearmanr(paired[feat], paired[target])

            results.append({
                "feature": feat,
                "target": target,
                "pearson_r": float(round(pr_val, 4)),
                "pearson_pvalue": float(round(pr_p, 6)),
                "pearson_significant": bool(pr_p < self.alpha),
                "spearman_rho": float(round(sr_val, 4)),
                "spearman_pvalue": float(round(sr_p, 6)),
                "spearman_significant": bool(sr_p < self.alpha),
                "n_obs": len(paired),
            })

        result_df = pd.DataFrame(results)
        logger.info(f"Computed pairwise correlation for target '{target}' across {len(results)} features.", extra=extra_ctx)
        return result_df

    def compute_correlation_matrices(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Compute full Pearson and Spearman correlation and p-value matrices.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (pearson_matrix, pearson_p_matrix, spearman_matrix, spearman_p_matrix)
        """
        if columns:
            target_cols = [c for c in columns if c in df.columns]
            sub_df = df[target_cols].select_dtypes(include=[np.number]).dropna()
        else:
            sub_df = df.select_dtypes(include=[np.number]).dropna()

        cols = sub_df.columns
        n = len(cols)

        p_matrix = np.zeros((n, n))
        p_pvalues = np.zeros((n, n))
        s_matrix = np.zeros((n, n))
        s_pvalues = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    p_matrix[i, j] = 1.0
                    s_matrix[i, j] = 1.0
                    p_pvalues[i, j] = 0.0
                    s_pvalues[i, j] = 0.0
                else:
                    r, p = pearsonr(sub_df.iloc[:, i], sub_df.iloc[:, j])
                    rho, sp = spearmanr(sub_df.iloc[:, i], sub_df.iloc[:, j])
                    p_matrix[i, j] = r
                    p_pvalues[i, j] = p
                    s_matrix[i, j] = rho
                    s_pvalues[i, j] = sp

        pearson_df = pd.DataFrame(p_matrix, index=cols, columns=cols).round(4)
        pearson_p_df = pd.DataFrame(p_pvalues, index=cols, columns=cols).round(6)
        spearman_df = pd.DataFrame(s_matrix, index=cols, columns=cols).round(4)
        spearman_p_df = pd.DataFrame(s_pvalues, index=cols, columns=cols).round(6)

        return pearson_df, pearson_p_df, spearman_df, spearman_p_df
