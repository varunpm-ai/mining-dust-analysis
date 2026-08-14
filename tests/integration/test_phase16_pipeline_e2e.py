"""
End-to-End ML Pipeline Lifecycle & Model Management Integration Tests (Phase 16).
Tests full job lifecycle, deterministic model selection, artifact versioning, and report generation.
"""

import pytest
from pathlib import Path

from api.services.pipeline_service import PipelineService
from src.modeling.registry import ModelArtifactRegistry
from tests.fixtures.synthetic_datasets import create_synthetic_csv, create_invalid_schema_csv


def test_full_pipeline_lifecycle_successful_execution(tmp_path):
    """Test complete successful 9-stage pipeline lifecycle from UPLOAD to COMPLETED."""
    csv_file = create_synthetic_csv(tmp_path / "train_dataset.csv", n_samples=80)

    service = PipelineService(upload_dir=tmp_path)
    job_id = service.create_job(dataset_filename=csv_file.name, target_column="PM10")

    results = service.execute_pipeline(job_id=job_id, cv_folds=3, primary_metric="RMSE")

    assert results["job_id"] == job_id
    assert results["winning_model_name"] is not None
    assert results["artifact_version"].startswith("v")
    assert Path(results["report_html_path"]).exists()

    job_status = service.get_job_status(job_id)
    assert job_status["status"] == "COMPLETED"
    assert job_status["progress_percent"] == 100


def test_full_pipeline_lifecycle_failure_handling(tmp_path):
    """Test job failure handling when processing invalid dataset, confirming FAILED status state."""
    invalid_csv = create_invalid_schema_csv(tmp_path / "invalid_schema.csv")

    service = PipelineService(upload_dir=tmp_path)
    job_id = service.create_job(dataset_filename=invalid_csv.name, target_column="PM10")

    with pytest.raises(Exception):
        service.execute_pipeline(job_id=job_id)

    job_status = service.get_job_status(job_id)
    assert job_status["status"] == "FAILED"
    assert job_status["error_message"] is not None


def test_deterministic_model_selection_and_versioning(tmp_path):
    """Test deterministic selection of best model and immutable artifact versioning."""
    csv_file = create_synthetic_csv(tmp_path / "selection_dataset.csv", n_samples=60)
    service = PipelineService(upload_dir=tmp_path)

    # Run 1
    job_id1 = service.create_job(dataset_filename=csv_file.name, target_column="PM10")
    res1 = service.execute_pipeline(job_id=job_id1, cv_folds=3, primary_metric="RMSE")

    # Run 2
    job_id2 = service.create_job(dataset_filename=csv_file.name, target_column="PM10")
    res2 = service.execute_pipeline(job_id=job_id2, cv_folds=3, primary_metric="RMSE")

    # Deterministic winner selection
    assert res1["winning_model_name"] == res2["winning_model_name"]

    # Immutable versioning increment
    registry = ModelArtifactRegistry()
    assert res1["artifact_version"] != res2["artifact_version"]

    # Verify both versions exist simultaneously
    loaded1 = registry.load_artifact(target="PM10", version=res1["artifact_version"])
    loaded2 = registry.load_artifact(target="PM10", version=res2["artifact_version"])
    assert loaded1["metadata"]["version"] == res1["artifact_version"]
    assert loaded2["metadata"]["version"] == res2["artifact_version"]
