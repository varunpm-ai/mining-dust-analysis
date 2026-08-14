"""
Statistical Multiple Linear Regression (MLR) Analysis Engine.

Extracts OLS regression parameters, confidence intervals, p-values, VIFs,
and comprehensive diagnostics into structured DataFrames and dictionaries.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.statistics.diagnostics import VarianceInflationFactorCalculator, RegressionDiagnostics
from src.utils.exceptions import StatisticalAnalysisError
from src.utils.logger import setup_logger

logger = setup_logger("statistics.regression")


class StatisticalRegressionAnalyzer:
    """
    Fits Multiple Linear Regression models for statistical hypothesis evaluation and inference.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def fit_ols(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        job_id: str = "N/A",
    ) -> Dict[str, Any]:
        """
        Fit statsmodels OLS model and compute structured regression analysis results.

        Args:
            df (pd.DataFrame): Dataset DataFrame.
            target_column (str): Target column name ("PM10" or "PM2.5").
            feature_columns (List[str], optional): Predictor feature names. Defaults to numerical features.
            job_id (str): Pipeline execution job ID.

        Returns:
            Dict[str, Any]: Comprehensive statistical regression report containing:
                - target: Target column name
                - r2: Coefficient of determination
                - r2_adjusted: Adjusted R-squared
                - f_statistic: F-statistic value
                - f_pvalue: F-statistic p-value
                - coefficients: DataFrame of coefficients, SE, t-values, p-values, 95% CIs, and VIFs
                - vif_summary: VIF DataFrame
                - diagnostics: Residual diagnostics dictionary
        """
        extra_ctx = {"job_id": job_id}

        if target_column not in df.columns:
            raise StatisticalAnalysisError(
                message=f"Target column '{target_column}' missing for OLS regression.",
                user_message=f"Target column '{target_column}' is missing.",
            )

        # Select predictor features
        if feature_columns:
            valid_feats = [c for c in feature_columns if c in df.columns]
        else:
            default_feats = ["Temperature", "Relative_humidity", "RF-CUMULATIVE", "Wind_speed"]
            valid_feats = [c for c in default_feats if c in df.columns]

        if not valid_feats:
            # Fallback to all numeric columns except targets
            numeric_df = df.select_dtypes(include=[np.number])
            valid_feats = [c for c in numeric_df.columns if c != target_column]

        if not valid_feats:
            raise StatisticalAnalysisError(
                message="No valid numerical predictor features available for regression.",
                user_message="No numeric predictor variables found for statistical regression.",
            )

        # Prepare clean X and y arrays
        data_clean = df[valid_feats + [target_column]].dropna()
        if len(data_clean) < len(valid_feats) + 2:
            raise StatisticalAnalysisError(
                message=f"Insufficient clean observations ({len(data_clean)}) for MLR fitting.",
                user_message="Insufficient clean data points to perform statistical regression.",
            )

        X = data_clean[valid_feats]
        y = data_clean[target_column]

        # Add constant term for statsmodels OLS
        X_const = sm.add_constant(X)

        try:
            model_fit = sm.OLS(y, X_const).fit()
        except Exception as e:
            logger.error(f"OLS fitting failed: {e}", exc_info=True, extra=extra_ctx)
            raise StatisticalAnalysisError(
                message=f"Statsmodels OLS execution error: {e}",
                user_message="Multiple linear regression calculation failed.",
                details={"error": str(e)},
            )

        # Compute VIF summary table
        vif_df = VarianceInflationFactorCalculator.compute_vif(X)

        # Build structured Coefficient DataFrame
        conf_int = model_fit.conf_int(alpha=self.alpha)
        coef_records = []

        for term in model_fit.params.index:
            coef_val = float(round(model_fit.params[term], 4))
            p_val = float(round(model_fit.pvalues[term], 6))
            se = float(round(model_fit.bse[term], 4))
            t_stat = float(round(model_fit.tvalues[term], 4))
            ci_lower = float(round(conf_int.loc[term, 0], 4))
            ci_upper = float(round(conf_int.loc[term, 1], 4))

            # Retrieve VIF if term is a feature
            vif_val = None
            if term != "const" and not vif_df.empty:
                vif_match = vif_df[vif_df["feature"] == term]
                if not vif_match.empty:
                    vif_val = float(vif_match["vif"].values[0])

            coef_records.append({
                "variable": term,
                "coefficient": coef_val,
                "std_error": se,
                "t_statistic": t_stat,
                "p_value": p_val,
                "statistically_significant": bool(p_val < self.alpha),
                "ci_lower_95": ci_lower,
                "ci_upper_95": ci_upper,
                "vif": vif_val,
            })

        coefficients_df = pd.DataFrame(coef_records)

        # Compute residual diagnostics
        diagnostics = RegressionDiagnostics.evaluate_residuals(model_fit)

        report = {
            "target": target_column,
            "n_observations": int(model_fit.nobs),
            "r2": float(round(model_fit.rsquared, 4)),
            "r2_adjusted": float(round(model_fit.rsquared_adj, 4)),
            "f_statistic": float(round(model_fit.fvalue, 4)) if model_fit.fvalue is not None else None,
            "f_pvalue": float(round(model_fit.f_pvalue, 6)) if model_fit.f_pvalue is not None else None,
            "coefficients": coefficients_df,
            "vif_summary": vif_df,
            "diagnostics": diagnostics,
        }

        logger.info(
            f"Fitted OLS model for target '{target_column}' (R² = {report['r2']:.4f}, Adj R² = {report['r2_adjusted']:.4f})",
            extra=extra_ctx,
        )

        return report
