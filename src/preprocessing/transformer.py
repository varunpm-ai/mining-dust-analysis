"""
Feature Scaling and Matrix Preparation Module.

Prepares X (features) and y (target) arrays and applies standard feature scaling.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.utils.configuration import get_config, AppConfig
from src.utils.exceptions import PreprocessingError
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing.transformer")


class FeatureTransformer:
    """
    Handles feature matrix extraction and scaling transformations.
    """

    def __init__(self, scaling_method: str = "standard"):
        """
        Args:
            scaling_method (str): 'standard' (StandardScaler) or 'minmax' (MinMaxScaler) or None.
        """
        self.scaling_method = scaling_method
        self.scaler: Optional[Any] = None
        self.feature_columns: List[str] = []

    def prepare_x_y(
        self,
        df: pd.DataFrame,
        target_column: str,
        feature_columns: Optional[List[str]] = None,
        job_id: str = "N/A",
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract X (features) and y (target) from dataset.

        Args:
            df (pd.DataFrame): Encoded preprocessed dataset.
            target_column (str): Target column name e.g. "PM10" or "PM2.5".
            feature_columns (List[str], optional): Explicit feature columns to select.
            job_id (str): Pipeline execution job ID.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        extra_ctx = {"job_id": job_id}

        if target_column not in df.columns:
            raise PreprocessingError(
                message=f"Target column '{target_column}' missing from DataFrame for matrix extraction.",
                user_message=f"Target column '{target_column}' is missing.",
            )

        y = df[target_column]

        if feature_columns:
            self.feature_columns = [c for c in feature_columns if c in df.columns]
        else:
            # Exclude all target columns from features
            config = get_config()
            exclude_targets = config.target_columns
            self.feature_columns = [c for c in df.columns if c not in exclude_targets]

        if not self.feature_columns:
            raise PreprocessingError(
                message="No feature columns found after removing target columns.",
                user_message="DataFrame contains no feature variables for model training.",
            )

        X = df[self.feature_columns]
        logger.info(f"Extracted X shape: {X.shape}, y shape: {y.shape} for target '{target_column}'", extra=extra_ctx)
        return X, y

    def fit_scale(
        self,
        X: pd.DataFrame,
        job_id: str = "N/A",
    ) -> pd.DataFrame:
        """
        Fit scaler on X and return scaled DataFrame.
        """
        extra_ctx = {"job_id": job_id}
        if self.scaling_method is None or self.scaling_method.lower() == "none":
            return X.copy()

        if self.scaling_method.lower() == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method.lower() == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise PreprocessingError(
                message=f"Unsupported scaling method: {self.scaling_method}",
                user_message="Invalid feature scaling configuration.",
            )

        scaled_array = self.scaler.fit_transform(X)
        scaled_df = pd.DataFrame(scaled_array, columns=X.columns, index=X.index)

        logger.info(f"Applied {self.scaling_method} scaling to feature matrix.", extra=extra_ctx)
        return scaled_df

    def transform_scale(self, X: pd.DataFrame, job_id: str = "N/A") -> pd.DataFrame:
        """
        Scale test/validation X using fitted scaler instance.
        """
        if self.scaler is None:
            return X.copy()

        scaled_array = self.scaler.transform(X)
        return pd.DataFrame(scaled_array, columns=X.columns, index=X.index)
