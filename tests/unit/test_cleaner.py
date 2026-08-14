"""
Unit Tests for DataCleaner and Quality Reporter.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df


def test_cleaner_deduplication():
    """Verify exact duplicate rows are identified and removed."""
    df = create_synthetic_legacy_df(num_rows=20)
    # Duplicate row 0
    duplicate_row = df.iloc[[0]].copy()
    df_dup = pd.concat([df, duplicate_row], ignore_index=True)

    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean(df_dup, job_id="job-clean-1")

    assert len(cleaned_df) == 20
    assert report["duplicate_rows_removed"] == 1
    assert any(t["operation"] == "remove_duplicates" for t in report["transformations"])


def test_cleaner_numeric_string_coersion():
    """Verify non-numeric strings in numeric columns are coerced to NaN and imputed."""
    df = create_synthetic_legacy_df(num_rows=15)
    df["Temperature"] = df["Temperature"].astype(object)
    df.loc[2, "Temperature"] = "invalid_string"

    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean(df, job_id="job-clean-2")

    assert report["coerced_numeric_values"]["Temperature"] == 1
    assert pd.notna(cleaned_df.loc[2, "Temperature"])  # Imputed median value
    assert isinstance(cleaned_df.loc[2, "Temperature"], (float, np.floating))


def test_cleaner_outliers_retained_rule5():
    """
    Verify Rule 5: Legitimate environmental outliers are detected and logged in the report
    but NEVER deleted from the cleaned dataset.
    """
    df = create_synthetic_legacy_df(num_rows=30)
    # Inject environmental extreme value in PM10
    df.loc[5, "PM10"] = 999.0

    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean(df, job_id="job-clean-3")

    # Outlier must be detected in report
    assert "PM10" in report["outliers_detected"]
    assert report["outliers_detected"]["PM10"]["count"] >= 1
    assert report["outliers_detected"]["PM10"]["retained"] is True

    # Outlier row MUST NOT be deleted from cleaned dataset
    assert 999.0 in cleaned_df["PM10"].values
    assert len(cleaned_df) == 30


def test_cleaner_categorical_imputation():
    """Verify missing categorical values are imputed using mode value."""
    df = create_synthetic_legacy_df(num_rows=25)
    df.loc[3, "LOCATION"] = np.nan

    cleaner = DataCleaner()
    cleaned_df, report = cleaner.clean(df, job_id="job-clean-4")

    assert pd.notna(cleaned_df.loc[3, "LOCATION"])
    assert any(t["operation"] == "impute_categorical_mode" for t in report["transformations"])
