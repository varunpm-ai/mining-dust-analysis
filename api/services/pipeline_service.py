"""
Pipeline Orchestration Service with Fine-Grained Job Lifecycle Management.

Coordinates data ingestion, schema validation, statistical analysis,
ML training, model selection, artifact versioning, and HTML report generation
with explicit lifecycle state transitions:
UPLOAD -> VALIDATING -> PREPROCESSING -> ANALYZING -> TRAINING -> EVALUATING -> SELECTING -> REPORTING -> COMPLETED / FAILED
"""

import uuid
import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from src.ingestion.loader import DatasetLoader
from src.ingestion.schema_validator import SchemaValidator
from src.preprocessing.cleaner import DataCleaner
from src.statistics.correlation import CorrelationAnalyzer
from src.modeling.trainer import ModelTrainer
from src.modeling.registry import ModelArtifactRegistry
from src.reporting.chart_generator import ChartGenerator
from src.reporting.report_generator import ReportGenerator
from src.utils.hashing import calculate_dataframe_hash
from src.utils.exceptions import MineDustException, JobNotFoundError, DatasetNotFoundError
from src.utils.logger import setup_logger

logger = setup_logger("api.services.pipeline_service")


class PipelineService:
    """
    Service layer orchestrating complete end-to-end mine dust analysis workflows.
    """

    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("artifacts/datasets/uploads")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, dataset_filename: str, target_column: str) -> str:
        """Create new job entry in UPLOAD stage and return unique job ID."""
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        file_path = self.upload_dir / dataset_filename

        if not file_path.exists():
            raise DatasetNotFoundError(
                message=f"Uploaded file '{dataset_filename}' not found at {file_path}",
                user_message=f"Dataset file '{dataset_filename}' was not found. Please upload it first.",
            )

        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._jobs[job_id] = {
            "job_id": job_id,
            "status": "UPLOAD",
            "progress_percent": 10,
            "target_column": target_column,
            "dataset_filename": dataset_filename,
            "file_path": str(file_path.resolve()),
            "created_at": timestamp_str,
            "completed_at": None,
            "error_message": None,
            "results": None,
            "report_path": None,
            "artifact_version": None,
        }
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Retrieve job execution status and current lifecycle stage."""
        if job_id not in self._jobs:
            raise JobNotFoundError(
                message=f"Job ID '{job_id}' not found in job registry.",
                user_message=f"Analysis job '{job_id}' does not exist.",
            )
        return self._jobs[job_id]

    def _update_stage(self, job_id: str, status: str, progress: int):
        """Update job stage status and progress percentage."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = status
            self._jobs[job_id]["progress_percent"] = progress
            logger.info(f"Job '{job_id}' stage -> {status} ({progress}%)", extra={"job_id": job_id})

    def execute_pipeline(self, job_id: str, cv_folds: int = 5, primary_metric: str = "RMSE") -> Dict[str, Any]:
        """
        Execute orchestrated pipeline through explicit lifecycle stages:
        VALIDATING -> PREPROCESSING -> ANALYZING -> TRAINING -> EVALUATING -> SELECTING -> REPORTING -> COMPLETED
        """
        job = self.get_job_status(job_id)
        file_path = job["file_path"]
        target = job["target_column"]

        logger.info(f"Starting orchestrated pipeline execution for job '{job_id}'", extra={"job_id": job_id})

        try:
            # Stage 1: Ingestion & Validation
            self._update_stage(job_id, "VALIDATING", 20)
            loader = DatasetLoader()
            df_raw, quality_report = loader.load_dataset(file_path, job_id=job_id)
            dataset_hash = calculate_dataframe_hash(df_raw)

            validator = SchemaValidator()
            validator.validate_schema(df_raw, target_columns=[target], job_id=job_id)

            # Stage 2: Preprocessing
            self._update_stage(job_id, "PREPROCESSING", 35)
            cleaner = DataCleaner()
            df_clean, cleaning_report = cleaner.clean(df_raw, job_id=job_id)

            # Stage 3: Analyzing
            self._update_stage(job_id, "ANALYZING", 50)
            corr_analyzer = CorrelationAnalyzer()
            corr_table = corr_analyzer.analyze_pairwise(df_clean, target=target, job_id=job_id)

            # Stage 4: Training & Evaluating
            self._update_stage(job_id, "TRAINING", 65)
            trainer = ModelTrainer()
            training_report = trainer.train_models_cv(df_clean, target_column=target, n_splits=cv_folds, job_id=job_id)

            # Stage 5: Selecting
            self._update_stage(job_id, "SELECTING", 80)
            best_model_result = training_report["best_model"]
            comparison_df = training_report["comparison_df"]

            # Save Model Artifact
            registry = ModelArtifactRegistry()
            version_str = registry.save_artifact(
                target=target,
                best_model_result=best_model_result,
                feature_names=training_report["feature_names"],
                dataset_hash=dataset_hash,
                encoder=training_report.get("encoder"),
                transformer=training_report.get("transformer"),
                job_id=job_id,
            )

            # Stage 6: Reporting
            self._update_stage(job_id, "REPORTING", 90)
            chart_gen = ChartGenerator(output_dir=Path(f"artifacts/reports/charts/{job_id}"))
            dist_chart = chart_gen.plot_numerical_distributions(df_clean)
            seasonal_chart = chart_gen.plot_seasonal_analysis(df_clean, target=target)
            model_chart = chart_gen.plot_model_comparison(comparison_df, primary_metric=primary_metric)

            chart_paths = {
                "Numerical Distributions": dist_chart,
                "Seasonal Analysis": seasonal_chart,
                "Model Comparison": model_chart,
            }

            report_gen = ReportGenerator(output_dir=Path("artifacts/reports"))
            report_path = report_gen.generate_html_report(
                quality_report=quality_report,
                correlation_df=corr_table,
                model_comparison_df=comparison_df,
                chart_paths=chart_paths,
                winning_model_name=best_model_result.best_model_name,
                winning_metric_val=best_model_result.best_model_metrics.get(primary_metric, 0.0),
                primary_metric=primary_metric,
                job_id=job_id,
            )

            # Final Stage: COMPLETED
            self._update_stage(job_id, "COMPLETED", 100)
            job["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job["report_path"] = report_path
            job["artifact_version"] = version_str
            job["results"] = {
                "job_id": job_id,
                "target_column": target,
                "quality_report": quality_report,
                "winning_model_name": best_model_result.best_model_name,
                "winning_metrics": best_model_result.best_model_metrics,
                "comparison_table": comparison_df.to_dict(orient="records"),
                "report_html_path": report_path,
                "artifact_version": version_str,
            }

            logger.info(f"Pipeline job '{job_id}' completed successfully. Artifact: '{version_str}'", extra={"job_id": job_id})
            return job["results"]

        except Exception as e:
            self._update_stage(job_id, "FAILED", job.get("progress_percent", 0))
            job["completed_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_msg = getattr(e, "user_message", "An error occurred during pipeline execution.")
            job["error_message"] = user_msg
            logger.error(f"Pipeline execution FAILED for job '{job_id}': {e}", exc_info=True, extra={"job_id": job_id})
            raise
