"""
Unit Tests for Data Loader (CSV & Excel Ingestion).
"""

import pytest
import pandas as pd

from src.ingestion.loader import DataLoader
from tests.fixtures.synthetic_datasets import (
    create_synthetic_legacy_df,
    create_synthetic_operational_df,
    create_synthetic_csv_bytes,
    create_synthetic_multirow_header_excel_bytes,
)


def test_load_csv_dataset():
    """Verify loading synthetic CSV dataset."""
    df_orig = create_synthetic_legacy_df(num_rows=30)
    csv_bytes = create_synthetic_csv_bytes(df_orig)

    loader = DataLoader()
    loaded_df, audit_manifest = loader.load(csv_bytes, file_type="csv", job_id="job-csv-1")

    assert loaded_df is not None
    assert len(loaded_df) == 30
    assert "PM10" in loaded_df.columns
    assert "PM2.5" in loaded_df.columns
    assert audit_manifest["raw_rows"] == 30
    assert audit_manifest["format"] == "csv"


def test_load_complex_excel_dataset():
    """Verify loading synthetic multi-row header Excel with numeric strings and duplicate columns."""
    excel_bytes = create_synthetic_multirow_header_excel_bytes()

    loader = DataLoader()
    loaded_df, audit_manifest = loader.load(excel_bytes, file_type="excel", job_id="job-excel-1")

    assert loaded_df is not None
    assert len(loaded_df) == 4
    assert audit_manifest["format"] == "excel"

    # Check numeric coersion of " 30.1 " string and "31,5" comma decimal
    assert "Temperature" in loaded_df.columns
    assert loaded_df["Temperature"].dtype in (float, "float64")
    assert loaded_df["Temperature"].iloc[3] == 31.5

    # Check duplicate PM10 column resolution
    assert "PM10" in loaded_df.columns
    assert "PM10_2" in loaded_df.columns

    # Verify Unnamed empty column was dropped
    assert not any("Unnamed" in str(col) for col in loaded_df.columns)


def test_load_operational_features_dataset():
    """Verify loading synthetic dataset with new operational features."""
    df_orig = create_synthetic_operational_df(num_rows=25)
    csv_bytes = create_synthetic_csv_bytes(df_orig)

    loader = DataLoader()
    loaded_df, audit_manifest = loader.load(csv_bytes, file_type="csv")

    assert "Road_width" in loaded_df.columns
    assert "Production_ROM" in loaded_df.columns
    assert "Production_OB" in loaded_df.columns
    assert "Vehicle_capacity" in loaded_df.columns
