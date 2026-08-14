"""
End-to-End Ingestion & Validation Integration Tests (Phase 16).
Tests dataset uploads, merged headers, schema validation edge cases, comma decimals, and safe error handling.
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from api.main import app
from src.ingestion.loader import DatasetLoader
from src.ingestion.schema_validator import SchemaValidator
from src.preprocessing.cleaner import DataCleaner
from tests.fixtures.synthetic_datasets import (
    create_synthetic_csv,
    create_synthetic_xlsx,
    create_synthetic_merged_header_xlsx,
    create_invalid_schema_csv,
    create_missing_pm10_csv,
    create_missing_pm2_5_csv,
    create_missing_weather_csv,
    create_duplicate_columns_csv,
    create_invalid_numeric_values_csv,
    create_empty_dataset_csv,
)

client = TestClient(app)


def test_csv_and_xlsx_api_upload_e2e(tmp_path):
    """Test REST API upload endpoint for both CSV and XLSX files."""
    csv_file = create_synthetic_csv(tmp_path / "test.csv")
    xlsx_file = create_synthetic_xlsx(tmp_path / "test.xlsx")

    # CSV Upload
    with open(csv_file, "rb") as f:
        res_csv = client.post("/datasets/upload", files={"file": ("test.csv", f, "text/csv")})
    assert res_csv.status_code == 200
    assert res_csv.json()["dataset_hash"] is not None

    # XLSX Upload
    with open(xlsx_file, "rb") as f:
        res_xlsx = client.post("/datasets/upload", files={"file": ("test.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert res_xlsx.status_code == 200
    assert res_xlsx.json()["dataset_hash"] is not None


def test_merged_excel_headers_parsing_e2e(tmp_path):
    """Test ingestion loader handling of real-world 2-row merged Excel headers."""
    xlsx_file = create_synthetic_merged_header_xlsx(tmp_path / "merged.xlsx")
    loader = DatasetLoader()
    df_raw, quality_report = loader.load_dataset(xlsx_file)

    assert df_raw is not None
    assert "PM10" in df_raw.columns
    assert "Road_width" in df_raw.columns
    assert quality_report.row_count == 3


def test_invalid_schema_and_empty_dataset_api_responses(tmp_path):
    """Test safe API error handling for empty files and invalid schemas."""
    empty_file = create_empty_dataset_csv(tmp_path / "empty.csv")
    invalid_file = create_invalid_schema_csv(tmp_path / "invalid.csv")

    # Empty File API Upload
    with open(empty_file, "rb") as f:
        res_empty = client.post("/datasets/upload", files={"file": ("empty.csv", f, "text/csv")})
    assert res_empty.status_code == 400
    assert res_empty.json()["code"] == "DATA_INGESTION_ERROR"

    # Invalid Schema Pipeline Run
    loader = DatasetLoader()
    df_raw, _ = loader.load_dataset(invalid_file)
    validator = SchemaValidator()

    with pytest.raises(Exception):
        validator.validate_schema(df_raw, target_columns=["PM10"])


def test_missing_targets_and_weather_features(tmp_path):
    """Test schema validation behavior when target or weather columns are omitted."""
    csv_no_pm10 = create_missing_pm10_csv(tmp_path / "no_pm10.csv")
    csv_no_pm25 = create_missing_pm2_5_csv(tmp_path / "no_pm25.csv")
    csv_no_weather = create_missing_weather_csv(tmp_path / "no_weather.csv")

    loader = DatasetLoader()
    validator = SchemaValidator()

    # Missing PM10
    df_no_pm10, _ = loader.load_dataset(csv_no_pm10)
    with pytest.raises(Exception):
        validator.validate_schema(df_no_pm10, target_columns=["PM10"])

    # Valid validation when targeting PM2.5 on PM2.5-only dataset
    df_no_pm10_valid = validator.validate_schema(df_no_pm10, target_columns=["PM2.5"])
    assert "PM2.5" in df_no_pm10_valid.columns

    # Missing PM2.5 target validation
    df_no_pm25, _ = loader.load_dataset(csv_no_pm25)
    df_no_pm25_valid = validator.validate_schema(df_no_pm25, target_columns=["PM10"])
    assert "PM10" in df_no_pm25_valid.columns


def test_duplicate_columns_and_invalid_numeric_cleaning(tmp_path):
    """Test deduplication of column names and conversion of comma decimals ("31,5" -> 31.5)."""
    csv_dups = create_duplicate_columns_csv(tmp_path / "dups.csv")
    csv_invalid_num = create_invalid_numeric_values_csv(tmp_path / "invalid_num.csv")

    loader = DatasetLoader()
    cleaner = DataCleaner()

    # Deduplication
    df_dups, _ = loader.load_dataset(csv_dups)
    assert len(df_dups.columns) == len(set(df_dups.columns))

    # Comma decimals & string conversion
    df_num_raw, _ = loader.load_dataset(csv_invalid_num)
    df_num_clean, report = cleaner.clean(df_num_raw)

    assert df_num_clean["Temperature"].iloc[0] == 31.5
    assert df_num_clean["PM10"].iloc[0] == 115.4
    assert df_num_clean["PM2.5"].iloc[0] == 55.2
