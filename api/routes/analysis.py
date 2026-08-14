"""
Pipeline Analysis Routes.
"""

from pathlib import Path
from typing import Dict, Any
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from api.schemas.requests import AnalysisStartRequest
from api.schemas.responses import JobStatusResponse, AnalysisResultsResponse
from api.services.pipeline_service import PipelineService
from src.utils.exceptions import MineDustException, JobNotFoundError, DatasetNotFoundError
from src.utils.logger import setup_logger

logger = setup_logger("api.routes.analysis")
router = APIRouter(prefix="/analysis", tags=["Analysis Pipeline"])

pipeline_service = PipelineService()


@router.post("/start", response_model=JobStatusResponse)
def start_analysis(request: AnalysisStartRequest, background_tasks: BackgroundTasks):
    """
    Start automated data validation, statistical analysis, ML training, and report generation.
    """
    job_id = pipeline_service.create_job(
        dataset_filename=request.dataset_filename,
        target_column=request.target_column,
    )

    # Synchronous execution for API processing / background tasks
    pipeline_service.execute_pipeline(
        job_id=job_id,
        cv_folds=request.cv_folds,
        primary_metric=request.primary_metric,
    )

    job_info = pipeline_service.get_job_status(job_id)
    return JobStatusResponse(**job_info)


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_analysis_status(job_id: str):
    """Get job execution status."""
    job_info = pipeline_service.get_job_status(job_id)
    return JobStatusResponse(**job_info)


@router.get("/{job_id}/results", response_model=AnalysisResultsResponse)
def get_analysis_results(job_id: str):
    """Get structured statistical and ML model selection results."""
    job_info = pipeline_service.get_job_status(job_id)
    if job_info["status"].lower() != "completed" or not job_info["results"]:
        raise HTTPException(status_code=400, detail=f"Job '{job_id}' is not completed yet.")
    return AnalysisResultsResponse(**job_info["results"])


@router.get("/{job_id}/report")
def get_analysis_report(job_id: str):
    """Download or view the generated executive HTML report."""
    job_info = pipeline_service.get_job_status(job_id)
    if job_info["status"].lower() != "completed" or not job_info["report_path"]:
        raise HTTPException(status_code=400, detail=f"Report for job '{job_id}' is not available.")

    report_file = Path(job_info["report_path"])
    if not report_file.exists():
        raise HTTPException(status_code=404, detail="Report file missing on disk.")

    return FileResponse(path=report_file, media_type="text/html", filename=report_file.name)


@router.get("/{job_id}/dataset")
def get_analysis_dataset(job_id: str):
    """Download the dataset associated with an analysis job."""
    job_info = pipeline_service.get_job_status(job_id)
    file_path = Path(job_info["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Dataset file missing on disk.")

    return FileResponse(path=file_path, filename=file_path.name)


@router.get("/{job_id}/model")
def get_analysis_model(job_id: str):
    """Retrieve winning model details for a completed analysis job."""
    job_info = pipeline_service.get_job_status(job_id)
    if job_info["status"].lower() != "completed" or not job_info["artifact_version"]:
        raise HTTPException(status_code=400, detail=f"Model artifact for job '{job_id}' is not available.")

    return {
        "job_id": job_id,
        "target": job_info["target_column"],
        "winning_model_name": job_info["results"]["winning_model_name"],
        "artifact_version": job_info["artifact_version"],
        "winning_metrics": job_info["results"]["winning_metrics"],
    }
