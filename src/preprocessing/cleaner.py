"""
Data Cleaning, Imputation, Outlier Audit, and Quality Reporting Module.

Implements Rule 5: Never silently corrupt data or delete legitimate environmental outliers.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.configuration import get_config, AppConfig
from src.utils.exceptions import PreprocessingError
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing.cleaner")


class DataCleaner:
    """
    Production data cleaner for mine dust datasets.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()

    def clean(
        self,
        df: pd.DataFrame,
        job_id: str = "N/A",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean dataset, impute missing values, detect outliers, and build quality report.

        Args:
            df (pd.DataFrame): Ingested, schema-validated DataFrame.
            job_id (str): Pipeline execution job ID.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Cleaned DataFrame, Data Quality Report)
        """
        extra_ctx = {"job_id": job_id}
        cleaned_df = df.copy()
        
        quality_report: Dict[str, Any] = {
            "job_id": job_id,
            "initial_rows": len(df),
            "initial_cols": len(df.columns),
            "duplicate_rows_removed": 0,
            "missing_values_before": {},
            "missing_values_after": {},
            "coerced_numeric_values": {},
            "outliers_detected": {},
            "transformations": [],
        }

        # 1. Deduplication Check
        initial_count = len(cleaned_df)
        cleaned_df.drop_duplicates(inplace=True)
        duplicates_removed = initial_count - len(cleaned_df)
        quality_report["duplicate_rows_removed"] = duplicates_removed

        if duplicates_removed > 0:
            quality_report["transformations"].append({
                "operation": "remove_duplicates",
                "affected_rows": duplicates_removed,
                "reason": f"Removed {duplicates_removed} exact duplicate rows from dataset.",
            })
            logger.info(f"Removed {duplicates_removed} duplicate rows.", extra=extra_ctx)

        # Record missing values before cleaning
        quality_report["missing_values_before"] = cleaned_df.isna().sum().to_dict()

        # 2. Identify Numeric & Categorical Columns from Schema Config
        numeric_cols = self._get_numeric_columns(cleaned_df)
        categorical_cols = self._get_categorical_columns(cleaned_df)

        # 3. Numeric Coersion and Cleaning
        for col in numeric_cols:
            if col in cleaned_df.columns:
                before_nans = cleaned_df[col].isna().sum()
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
                after_nans = cleaned_df[col].isna().sum()
                coerced_count = after_nans - before_nans

                if coerced_count > 0:
                    quality_report["coerced_numeric_values"][col] = int(coerced_count)
                    quality_report["transformations"].append({
                        "column": col,
                        "operation": "coerce_numeric",
                        "affected_rows": int(coerced_count),
                        "reason": f"Coerced {coerced_count} non-numeric values to NaN in column '{col}'.",
                    })

        # 4. Outlier Detection & Audit (RULE 5: DO NOT DELETE OUTLIERS)
        for col in numeric_cols:
            if col in cleaned_df.columns:
                series = cleaned_df[col].dropna()
                if len(series) > 0:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = series[(series < lower_bound) | (series > upper_bound)]
                    
                    quality_report["outliers_detected"][col] = {
                        "count": len(outliers),
                        "lower_bound": float(round(lower_bound, 2)),
                        "upper_bound": float(round(upper_bound, 2)),
                        "retained": True,  # Explicitly log that outliers are retained
                    }
                    if len(outliers) > 0:
                        logger.info(
                            f"Detected {len(outliers)} outliers in '{col}' [bounds: {lower_bound:.1f}, {upper_bound:.1f}]. Retaining all intact.",
                            extra=extra_ctx,
                        )

        # 5. Missing Value Imputation
        for col in numeric_cols:
            if col in cleaned_df.columns:
                nans_count = cleaned_df[col].isna().sum()
                if nans_count > 0:
                    # Target columns: drop rows with missing targets
                    if col in self.config.target_columns:
                        cleaned_df.dropna(subset=[col], inplace=True)
                        quality_report["transformations"].append({
                            "column": col,
                            "operation": "drop_missing_target_rows",
                            "affected_rows": int(nans_count),
                            "reason": f"Dropped {nans_count} rows missing target value '{col}'.",
                        })
                    else:
                        median_val = cleaned_df[col].median()
                        if pd.isna(median_val):
                            median_val = 0.0
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        quality_report["transformations"].append({
                            "column": col,
                            "operation": "impute_numeric_median",
                            "affected_rows": int(nans_count),
                            "imputed_value": float(round(median_val, 4)),
                            "reason": f"Imputed {nans_count} missing values in '{col}' using median ({median_val:.2f}).",
                        })

        # 6. Categorical Cleaning & Imputation
        for col in categorical_cols:
            if col in cleaned_df.columns:
                series = cleaned_df[col].astype(str).str.strip().str.upper()
                series = series.replace(["NAN", "NONE", "NULL", "<NA>", ""], np.nan)

                nans_count = series.isna().sum()
                if nans_count > 0:
                    mode_val = series.dropna().mode()
                    fill_val = mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
                    series = series.fillna(fill_val)
                    quality_report["transformations"].append({
                        "column": col,
                        "operation": "impute_categorical_mode",
                        "affected_rows": int(nans_count),
                        "imputed_value": fill_val,
                        "reason": f"Imputed {nans_count} missing categorical values in '{col}' with mode '{fill_val}'.",
                    })
                cleaned_df[col] = series

        quality_report["missing_values_after"] = cleaned_df.isna().sum().to_dict()
        quality_report["final_rows"] = len(cleaned_df)
        quality_report["final_cols"] = len(cleaned_df.columns)

        logger.info(
            f"Data cleaning completed. Rows: {quality_report['initial_rows']} -> {quality_report['final_rows']}, "
            f"Transformations applied: {len(quality_report['transformations'])}",
            extra=extra_ctx,
        )

        return cleaned_df, quality_report

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Determine numeric column names from schema configuration and DataFrame types."""
        configured_numerics = ["Temperature", "Relative_humidity", "RF-CUMULATIVE", "Wind_speed"]
        configured_numerics.extend(self.config.target_columns)
        
        # Add operational numeric features
        for f in self.config.operational_features:
            if f.get("type") in ("numeric", "integer"):
                configured_numerics.append(f["name"])

        # Retain columns present in df
        present_numerics = [c for c in configured_numerics if c in df.columns]
        
        # Also include any actual float/int columns in df
        auto_numerics = df.select_dtypes(include=[np.number]).columns.tolist()
        return list(dict.fromkeys(present_numerics + auto_numerics))

    def _get_categorical_columns(self, df: pd.DataFrame) -> List[str]:
        """Determine categorical column names from schema configuration."""
        configured_categoricals = ["SEASON", "LOCATION", "MONTH"]
        for f in self.config.operational_features:
            if f.get("type") == "categorical":
                configured_categoricals.append(f["name"])

        return [c for c in configured_categoricals if c in df.columns]
