"""
Unit Tests for Centralized Production Logging System.
"""

import logging
from pathlib import Path
import pytest

from src.utils.logger import setup_logger, log_technical_and_user_error
from src.utils.exceptions import IngestionError, PreprocessingError


def test_setup_logger_creates_file(tmp_path):
    """Verify logger setup creates file handler and outputs log records."""
    log_file = tmp_path / "logs" / "test_pipeline.log"
    logger = setup_logger(name="test_logger", log_file=log_file, log_level="DEBUG")

    assert logger.level == logging.DEBUG
    logger.info("Test log message", extra={"job_id": "job-999"})

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "INFO" in content
    assert "test_logger" in content
    assert "[Job: job-999]" in content
    assert "Test log message" in content


def test_log_technical_and_user_error_formatting(tmp_path):
    """Verify error logger logs full traceback and returns non-technical user message."""
    log_file = tmp_path / "logs" / "error_test.log"
    logger = setup_logger(name="error_logger", log_file=log_file)

    custom_exc = IngestionError(
        message="Failed to parse header row 3 in file test.xlsx due to pandas CParserError",
        user_message="The uploaded Excel file could not be read. Please check column headers.",
        details={"file": "test.xlsx", "row": 3},
    )

    user_msg = log_technical_and_user_error(logger=logger, exc=custom_exc, job_id="job-101")
    assert user_msg == "The uploaded Excel file could not be read. Please check column headers."

    content = log_file.read_text(encoding="utf-8")
    assert "[INGESTION_ERROR]" in content
    assert "pandas CParserError" in content
    assert "[Job: job-101]" in content


def test_unhandled_exception_user_safety(tmp_path):
    """Verify unhandled system exception returns safe generic message to user."""
    log_file = tmp_path / "logs" / "unhandled_test.log"
    logger = setup_logger(name="unhandled_logger", log_file=log_file)

    try:
        _ = 1 / 0
    except ZeroDivisionError as unhandled:
        user_msg = log_technical_and_user_error(logger=logger, exc=unhandled, job_id="job-202")

    assert "division by zero" not in user_msg  # Raw trace omitted from user msg
    assert "unexpected error occurred" in user_msg.lower()

    content = log_file.read_text(encoding="utf-8")
    assert "ZeroDivisionError" in content
    assert "division by zero" in content
