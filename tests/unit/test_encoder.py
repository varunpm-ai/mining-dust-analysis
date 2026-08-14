"""
Unit Tests for CategoricalEncoder.
"""

import pandas as pd
import pytest

from src.preprocessing.encoder import CategoricalEncoder
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_fit_transform_one_hot_encoding():
    """Verify fitting and transforming categorical features into one-hot dummy matrix."""
    df = create_synthetic_legacy_df(num_rows=20)
    encoder = CategoricalEncoder(categorical_columns=["SEASON", "LOCATION"])

    encoded_df, manifest = encoder.fit_transform(df, job_id="job-enc-1")

    assert "SEASON" not in encoded_df.columns
    assert "LOCATION" not in encoded_df.columns
    assert any(col.startswith("SEASON_") for col in encoded_df.columns)
    assert any(col.startswith("LOCATION_") for col in encoded_df.columns)
    assert manifest["total_encoded_features"] > 0
    assert "PM10" in encoded_df.columns  # Target retained


def test_transform_unseen_data():
    """Verify transforming test/validation dataset using fitted CategoricalEncoder."""
    train_df = create_synthetic_legacy_df(num_rows=20)
    test_df = create_synthetic_legacy_df(num_rows=10)

    encoder = CategoricalEncoder(categorical_columns=["SEASON", "LOCATION"])
    train_encoded, _ = encoder.fit_transform(train_df, job_id="job-enc-train")
    test_encoded = encoder.transform(test_df, job_id="job-enc-test")

    assert list(train_encoded.columns) == list(test_encoded.columns)
    assert len(test_encoded) == 10
