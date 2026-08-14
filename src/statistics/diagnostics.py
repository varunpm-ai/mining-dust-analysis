"""
Regression Diagnostics and Multicollinearity (VIF) Module.

Calculates Variance Inflation Factors (VIF), residual autocorrelation (Durbin-Watson),
residual normality (Jarque-Bera, Omnibus), and condition numbers.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera, omni_normtest

from src.utils.logger import setup_logger

logger = setup_logger("statistics.diagnostics")


class VarianceInflationFactorCalculator:
    """
    Computes Variance Inflation Factor (VIF) for numerical feature matrices to identify multicollinearity.
    """

    @staticmethod
    def compute_vif(X: pd.DataFrame, add_constant: bool = True) -> pd.DataFrame:
        """
        Compute VIF values for all features in X.

        Args:
            X (pd.DataFrame): Numerical feature matrix.
            add_constant (bool): Whether to prepend a constant term column.

        Returns:
            pd.DataFrame: DataFrame containing feature names, VIF scores, and risk flags.
        """
        num_X = X.select_dtypes(include=[np.number]).dropna()
        if num_X.empty or num_X.shape[1] == 0:
            return pd.DataFrame(columns=["feature", "vif", "risk"])

        if add_constant and "const" not in num_X.columns:
            X_calc = sm.add_constant(num_X)
        else:
            X_calc = num_X.copy()

        vif_data = []
        for i, col in enumerate(X_calc.columns):
            if col == "const":
                continue
            try:
                vif_val = variance_inflation_factor(X_calc.values, i)
                if np.isnan(vif_val) or np.isinf(vif_val):
                    vif_val = 999.9
            except Exception:
                vif_val = 999.9

            risk = "Low"
            if vif_val > 10.0:
                risk = "High"
            elif vif_val > 5.0:
                risk = "Moderate"

            vif_data.append({
                "feature": col,
                "vif": float(round(vif_val, 3)),
                "multicollinearity_risk": risk,
            })

        return pd.DataFrame(vif_data)


class RegressionDiagnostics:
    """
    Computes comprehensive diagnostic metrics on fitted statsmodels OLS models.
    """

    @staticmethod
    def evaluate_residuals(model: Any) -> Dict[str, Any]:
        """
        Extract residual diagnostics from a fitted statsmodels OLS model instance.

        Args:
            model: Fitted statsmodels OLS result object.

        Returns:
            Dict[str, Any]: Residual diagnostics summary dictionary.
        """
        residuals = model.resid

        # Durbin-Watson Autocorrelation
        dw_stat = float(durbin_watson(residuals))

        # Jarque-Bera Normality Test
        jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(residuals)

        # Omnibus Normality Test
        omni_stat, omni_pvalue = omni_normtest(residuals)

        # Condition Number
        cond_no = float(np.linalg.cond(model.model.exog)) if hasattr(model.model, "exog") else None

        diagnostics = {
            "r2": float(round(model.rsquared, 4)),
            "r2_adjusted": float(round(model.rsquared_adj, 4)),
            "f_statistic": float(round(model.fvalue, 4)) if model.fvalue is not None else None,
            "f_pvalue": float(round(model.f_pvalue, 6)) if model.f_pvalue is not None else None,
            "aic": float(round(model.aic, 2)),
            "bic": float(round(model.bic, 2)),
            "durbin_watson": float(round(dw_stat, 3)),
            "jarque_bera_stat": float(round(jb_stat, 4)),
            "jarque_bera_pvalue": float(round(jb_pvalue, 6)),
            "omnibus_stat": float(round(omni_stat, 4)),
            "omnibus_pvalue": float(round(omni_pvalue, 6)),
            "skewness": float(round(skew, 4)),
            "kurtosis": float(round(kurtosis, 4)),
            "condition_number": float(round(cond_no, 2)) if cond_no else None,
        }

        return diagnostics
