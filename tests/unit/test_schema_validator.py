"""
Unit Tests for Schema Validator.
"""

import pytest
import pandas as pd

from src.ingestion.schema_validator import SchemaValidator
from src.utils.exceptions import SchemaValidationError
from tests.fixtures.synthetic_datasets import (
    create_synthetic_legacy_df,
    create_synthetic_operational_df,
)


def test_schema_validator_success_legacy():
    """Verify valid legacy schema dataset passes schema validation."""
    df = create_synthetic_legacy_df(num_rows=20)
    validator = SchemaValidator()
    is_valid, report = validator.validate(df, job_id="job-val-1")

    assert is_valid is True
    assert "PM10" in report["targets_found"]
    assert "PM2.5" in report["targets_found"]
    assert len(report["validation_errors"]) == 0


def test_schema_validator_success_operational():
    """Verify valid operational schema dataset passes schema validation."""
    df = create_synthetic_operational_df(num_rows=20)
    validator = SchemaValidator()
    is_valid, report = validator.validate(df, job_id="job-val-2")

    assert is_valid is True
    assert "Road_width" in report["operational_features_found"]
    assert "Production_ROM" in report["operational_features_found"]


def test_schema_validator_missing_targets_raises_exception():
    """Verify missing target column raises user-friendly SchemaValidationError."""
    df = create_synthetic_legacy_df(num_rows=20)
    df.drop(columns=["PM10"], inplace=True)

    validator = SchemaValidator()
    with pytest.raises(SchemaValidationError) as exc_info:
        validator.validate(df, job_id="job-err-1")

    assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"
    assert "PM10" in exc_info.value.user_message


def test_schema_validator_missing_required_features_raises_exception():
    """Verify missing mandatory environmental feature raises SchemaValidationError."""
    df = create_synthetic_legacy_df(num_rows=20)
    df.drop(columns=["SEASON"], inplace=True)

    validator = SchemaValidator()
    with pytest.raises(SchemaValidationError) as exc_info:
        validator.validate(df, job_id="job-err-2")

    assert exc_info.value.code == "SCHEMA_VALIDATION_ERROR"
    assert "SEASON" in exc_info.value.user_message
