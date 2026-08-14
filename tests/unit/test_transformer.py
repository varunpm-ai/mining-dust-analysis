"""
Unit Tests for FeatureTransformer.
"""

import pandas as pd
import pytest

from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.transformer import FeatureTransformer
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_prepare_x_y_matrix_extraction():
    """Verify separating X (features) and y (target) for model training."""
    df = create_synthetic_legacy_df(num_rows=25)
    encoder = CategoricalEncoder()
    encoded_df, _ = encoder.fit_transform(df)

    transformer = FeatureTransformer()
    X, y = transformer.prepare_x_y(encoded_df, target_column="PM10", job_id="job-tr-1")

    assert "PM10" not in X.columns
    assert "PM2.5" not in X.columns  # Excluded secondary target
    assert y.name == "PM10"
    assert len(X) == 25
    assert len(y) == 25


def test_standard_feature_scaling():
    """Verify StandardScaler normalizes features with mean ~ 0 and std ~ 1."""
    df = create_synthetic_legacy_df(num_rows=50)
    encoder = CategoricalEncoder()
    encoded_df, _ = encoder.fit_transform(df)

    transformer = FeatureTransformer(scaling_method="standard")
    X, _ = transformer.prepare_x_y(encoded_df, target_column="PM10")
    scaled_X = transformer.fit_scale(X, job_id="job-tr-scale")

    assert scaled_X.shape == X.shape
    # Check mean close to 0 for numerical columns
    assert abs(scaled_X["Temperature"].mean()) < 1e-5
