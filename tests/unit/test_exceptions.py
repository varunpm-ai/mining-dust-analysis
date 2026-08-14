"""
Unit Tests for Custom Exceptions Hierarchy.
"""

import pytest
from src.utils.exceptions import (
    MineDustException,
    IngestionError,
    HeaderNormalizationError,
    SchemaValidationError,
    PreprocessingError,
    ModelTrainingError,
    ConfigurationError,
)


def test_base_minedust_exception():
    """Verify base exception attributes and serialization."""
    exc = MineDustException(
        message="Technical database timeout",
        user_message="Service temporarily unavailable.",
        code="DB_TIMEOUT",
        details={"timeout_seconds": 30},
    )

    assert exc.message == "Technical database timeout"
    assert exc.user_message == "Service temporarily unavailable."
    assert exc.code == "DB_TIMEOUT"
    assert exc.details == {"timeout_seconds": 30}

    data = exc.to_dict()
    assert data["code"] == "DB_TIMEOUT"
    assert data["user_message"] == "Service temporarily unavailable."


def test_domain_specific_exceptions():
    """Verify domain specific exceptions set appropriate default user messages and error codes."""
    ingest_err = IngestionError("Corrupted byte stream")
    assert ingest_err.code == "INGESTION_ERROR"
    assert "uploaded dataset file" in ingest_err.user_message

    norm_err = HeaderNormalizationError("Duplicate column names [PM10, PM10]")
    assert norm_err.code == "HEADER_NORMALIZATION_ERROR"
    assert "column headers" in norm_err.user_message

    schema_err = SchemaValidationError("Missing required feature SEASON")
    assert schema_err.code == "SCHEMA_VALIDATION_ERROR"
    assert "validation failed" in schema_err.user_message

    prep_err = PreprocessingError("Non-numeric value 'abc' in PM10 column")
    assert prep_err.code == "PREPROCESSING_ERROR"
    assert "preprocessing" in prep_err.user_message.lower()

    model_err = ModelTrainingError("XGBoost ConvergenceWarning")
    assert model_err.code == "MODEL_TRAINING_ERROR"
    assert "Model training failed" in model_err.user_message
