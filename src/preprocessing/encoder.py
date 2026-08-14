"""
Categorical Encoding Module for Mine Dust Features.

Supports One-Hot Encoding with unknown handle safeguards and feature tracking.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from src.utils.logger import setup_logger

logger = setup_logger("preprocessing.encoder")


class CategoricalEncoder:
    """
    Encoder for converting categorical mining variables (SEASON, LOCATION, etc.) into numeric matrices.
    """

    def __init__(self, categorical_columns: Optional[List[str]] = None, drop_first: bool = False):
        """
        Args:
            categorical_columns (List[str], optional): Explicit list of categorical column names.
            drop_first (bool): Whether to drop the first dummy category (useful for linear models to prevent multicollinearity).
        """
        self.categorical_columns = categorical_columns or ["SEASON", "LOCATION", "MONTH"]
        self.drop_first = drop_first
        self.encoder: Optional[OneHotEncoder] = None
        self.encoded_feature_names: List[str] = []
        self.fitted_cols: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        job_id: str = "N/A",
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fit encoder on DataFrame categorical features and transform.

        Args:
            df (pd.DataFrame): Cleaned DataFrame.
            job_id (str): Pipeline execution job ID.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: (Encoded DataFrame, Encoding Manifest)
        """
        extra_ctx = {"job_id": job_id}
        self.fitted_cols = [c for c in self.categorical_columns if c in df.columns]

        if not self.fitted_cols:
            logger.info("No categorical columns found for encoding.", extra=extra_ctx)
            return df.copy(), {"encoded_columns": [], "new_feature_names": []}

        # Initialize Scikit-Learn OneHotEncoder
        drop_param = "first" if self.drop_first else None
        self.encoder = OneHotEncoder(
            sparse_output=False,
            drop=drop_param,
            handle_unknown="ignore" if drop_param is None else "error",
        )

        cat_matrix = df[self.fitted_cols].astype(str)
        encoded_array = self.encoder.fit_transform(cat_matrix)
        
        # Retrieve feature names
        self.encoded_feature_names = list(self.encoder.get_feature_names_out(self.fitted_cols))

        encoded_cat_df = pd.DataFrame(
            encoded_array,
            columns=self.encoded_feature_names,
            index=df.index,
        )

        # Combine non-categorical columns with newly encoded categorical columns
        non_cat_df = df.drop(columns=self.fitted_cols)
        result_df = pd.concat([non_cat_df, encoded_cat_df], axis=1)

        manifest = {
            "encoded_columns": self.fitted_cols,
            "new_feature_names": self.encoded_feature_names,
            "total_encoded_features": len(self.encoded_feature_names),
        }

        logger.info(
            f"Encoded categorical features {self.fitted_cols} -> {len(self.encoded_feature_names)} dummy columns.",
            extra=extra_ctx,
        )

        return result_df, manifest

    def transform(self, df: pd.DataFrame, job_id: str = "N/A") -> pd.DataFrame:
        """
        Transform new data using fitted OneHotEncoder instance.
        """
        extra_ctx = {"job_id": job_id}

        if self.encoder is None or not self.fitted_cols:
            return df.copy()

        present_cols = [c for c in self.fitted_cols if c in df.columns]
        if not present_cols:
            return df.copy()

        cat_matrix = df[present_cols].astype(str)
        encoded_array = self.encoder.transform(cat_matrix)

        encoded_cat_df = pd.DataFrame(
            encoded_array,
            columns=self.encoded_feature_names,
            index=df.index,
        )

        non_cat_df = df.drop(columns=present_cols, errors="ignore")
        result_df = pd.concat([non_cat_df, encoded_cat_df], axis=1)

        logger.info(f"Transformed test/validation dataset with fitted OneHotEncoder.", extra=extra_ctx)
        return result_df
