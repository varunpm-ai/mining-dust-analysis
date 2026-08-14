"""
Centralized Custom Exception Hierarchy for Mine Dust Analysis Platform.

All domain exceptions inherit from MineDustException and separate technical log messages
from non-technical, user-friendly error messages designed for client presentation.
"""

from typing import Any, Dict, Optional


class MineDustException(Exception):
    """
    Base exception class for all Mine Dust Platform errors.
    
    Attributes:
        message (str): Detailed technical exception message for logging/debugging.
        user_message (str): Clean, non-technical error message for client interface.
        code (str): Machine-readable error code.
        details (dict): Optional diagnostic key-value context.
    """

    def __init__(
        self,
        message: str,
        user_message: Optional[str] = None,
        code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.user_message = (
            user_message
            if user_message
            else "An unexpected error occurred during processing. Please try again or contact support."
        )
        self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            "code": self.code,
            "message": self.message,
            "user_message": self.user_message,
            "details": self.details,
        }


class ConfigurationError(MineDustException):
    """Raised when system or schema configuration fails to load or validate."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "System configuration error. Please notify the administrator.",
            code="CONFIG_ERROR",
            details=details,
        )


class IngestionError(MineDustException):
    """Raised when loading CSV or Excel files fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Unable to read the uploaded dataset file. Please ensure it is a valid CSV or Excel file.",
            code="INGESTION_ERROR",
            details=details,
        )


DataIngestionError = IngestionError


class HeaderNormalizationError(IngestionError):
    """Raised when Excel/CSV header normalization fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Dataset column headers could not be processed automatically. Please verify header formatting.",
            details=details,
        )
        self.code = "HEADER_NORMALIZATION_ERROR"


class SchemaValidationError(MineDustException):
    """Raised when a dataset fails required schema validations."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Dataset schema validation failed. Required variables or target columns are missing.",
            code="SCHEMA_VALIDATION_ERROR",
            details=details,
        )


class PreprocessingError(MineDustException):
    """Raised when data cleaning, type conversion, or transformation fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Data preprocessing encountered an error while cleaning numerical or categorical fields.",
            code="PREPROCESSING_ERROR",
            details=details,
        )


class StatisticalAnalysisError(MineDustException):
    """Raised when statistical tests, correlations, or MLR fail."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Statistical analysis could not be completed for the uploaded dataset.",
            code="STATISTICAL_ANALYSIS_ERROR",
            details=details,
        )


class ModelTrainingError(MineDustException):
    """Raised when machine learning model training fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Model training failed for one or more regression algorithms.",
            code="MODEL_TRAINING_ERROR",
            details=details,
        )


class ModelEvaluationError(MineDustException):
    """Raised when model evaluation or selection fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Model evaluation and selection failed.",
            code="MODEL_EVALUATION_ERROR",
            details=details,
        )


class ModelSelectionError(MineDustException):
    """Raised when automatic model ranking or selection fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Automatic model ranking or selection failed.",
            code="MODEL_SELECTION_ERROR",
            details=details,
        )


class ModelArtifactError(MineDustException):
    """Raised when model artifact versioning, saving, or loading fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Model artifact operations failed.",
            code="MODEL_ARTIFACT_ERROR",
            details=details,
        )


class JobNotFoundError(MineDustException):
    """Raised when requested job ID is not found."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Analysis job not found.",
            code="JOB_NOT_FOUND",
            details=details,
        )


class DatasetNotFoundError(MineDustException):
    """Raised when dataset file is not found."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Dataset file not found.",
            code="DATASET_NOT_FOUND",
            details=details,
        )


class ReportGenerationError(MineDustException):
    """Raised when automated report or chart generation fails."""

    def __init__(self, message: str, user_message: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            user_message=user_message or "Report generation failed while creating final visualizations and metrics summary.",
            code="REPORT_GENERATION_ERROR",
            details=details,
        )
