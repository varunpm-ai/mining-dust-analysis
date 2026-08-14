"""
Centralized Production Logging Infrastructure for Mine Dust Analysis Platform.

Provides console and rotating file logging with job ID context filtering,
structured log formatting, and dual technical/user exception handling.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from src.utils.exceptions import MineDustException


class JobContextFilter(logging.Filter):
    """Filter that ensures 'job_id' attribute is present in all log records."""

    def __init__(self, default_job_id: str = "N/A"):
        super().__init__()
        self.default_job_id = default_job_id

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "job_id") or record.job_id is None:
            record.job_id = self.default_job_id
        return True


def setup_logger(
    name: str = "mine_dust_mlops",
    log_file: Optional[Path] = None,
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure and return a structured logger instance.

    Args:
        name (str): Logger module name.
        log_file (Path, optional): Path to log file. Defaults to logs/pipeline.log.
        log_level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_bytes (int): Maximum size of log file before rotation.
        backup_count (int): Number of backup log files to retain.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # If logger already has handlers configured, return existing instance
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Log Formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | [Job: %(job_id)s] | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Job Context Filter
    job_filter = JobContextFilter()

    # Console Handler (Stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(job_filter)
    logger.addHandler(console_handler)

    # Rotating File Handler
    target_log_file = log_file or Path("logs/pipeline.log")
    target_log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(target_log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(job_filter)
    logger.addHandler(file_handler)

    return logger


def log_technical_and_user_error(
    logger: logging.Logger,
    exc: Exception,
    job_id: Optional[str] = None,
) -> str:
    """
    Logs full technical exception trace for developers while returning
    a clean, non-technical error message suitable for client display.

    Args:
        logger (logging.Logger): Logger instance.
        exc (Exception): Caught exception.
        job_id (str, optional): Pipeline execution job ID.

    Returns:
        str: User-safe error message.
    """
    extra_context = {"job_id": job_id or "N/A"}

    if isinstance(exc, MineDustException):
        logger.error(
            f"[{exc.code}] {type(exc).__name__}: {exc.message} | Details: {exc.details}",
            exc_info=True,
            extra=extra_context,
        )
        return exc.user_message
    else:
        logger.error(
            f"[UNHANDLED_EXCEPTION] {type(exc).__name__}: {str(exc)}",
            exc_info=True,
            extra=extra_context,
        )
        return "An unexpected error occurred during processing. Please verify your dataset and try again."
