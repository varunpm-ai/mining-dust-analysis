"""
API Request Schemas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class AnalysisStartRequest(BaseModel):
    """Payload to launch an asynchronous analysis pipeline job."""

    dataset_filename: str = Field(..., description="Filename of previously uploaded CSV or Excel dataset.")
    target_column: str = Field(default="PM10", description="Target column ('PM10' or 'PM2.5').")
    cv_folds: int = Field(default=5, ge=2, le=10, description="Cross-validation fold count.")
    primary_metric: str = Field(default="RMSE", description="Primary metric for model selection ('RMSE', 'MAE', 'R2').")
