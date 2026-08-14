"""
API Response Schemas.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "Mine Dust MLOps API"
    version: str = "2.0.0"


class DatasetUploadResponse(BaseModel):
    message: str
    filename: str
    filepath: str
    dataset_hash: str
    file_size_bytes: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "UPLOAD", "VALIDATING", "PREPROCESSING", "ANALYZING", "TRAINING", "EVALUATING", "SELECTING", "REPORTING", "COMPLETED", "FAILED"
    progress_percent: int = 100
    target_column: str
    dataset_filename: str
    created_at: str
    completed_at: Optional[str] = None
    error_message: Optional[str] = None


class AnalysisResultsResponse(BaseModel):
    job_id: str
    target_column: str
    quality_report: Dict[str, Any]
    winning_model_name: str
    winning_metrics: Dict[str, float]
    comparison_table: List[Dict[str, Any]]
    report_html_path: str


class ModelDetailsResponse(BaseModel):
    target: str
    version: str
    metadata: Dict[str, Any]


class ModelListResponse(BaseModel):
    target: str
    versions: List[str]


class ErrorResponse(BaseModel):
    code: str
    user_message: str
    details: Optional[Dict[str, Any]] = None
