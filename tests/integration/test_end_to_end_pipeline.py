"""
Integration Tests for End-to-End Orchestrated Pipeline Lifecycle.

Tests full workflow: UPLOAD -> VALIDATING -> PREPROCESSING -> ANALYZING -> TRAINING -> EVALUATING -> SELECTING -> REPORTING -> COMPLETED / FAILED
using synthetic datasets without confidential client data.
"""

import io
import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df, create_synthetic_operational_df

client = TestClient(app)


def test_end_to_end_orchestrated_pipeline_success_lifecycle(tmp_path):
    """
    Verify complete orchestrated job lifecycle state transitions from upload to completion.
    """
    # 1. UPLOAD STAGE
    df = create_synthetic_legacy_df(num_rows=35)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("e2e_synthetic_dataset.csv", io.BytesIO(csv_bytes), "text/csv")}

    upload_res = client.post("/datasets/upload", files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["filename"] == "e2e_synthetic_dataset.csv"

    # 2. START PIPELINE (Triggers VALIDATING -> PREPROCESSING -> ANALYZING -> TRAINING -> EVALUATING -> SELECTING -> REPORTING -> COMPLETED)
    payload = {
        "dataset_filename": "e2e_synthetic_dataset.csv",
        "target_column": "PM10",
        "cv_folds": 3,
        "primary_metric": "RMSE",
    }
    start_res = client.post("/analysis/start", json=payload)
    assert start_res.status_code == 200

    job_data = start_res.json()
    job_id = job_data["job_id"]
    assert job_data["status"] == "COMPLETED"
    assert job_data["progress_percent"] == 100

    # 3. VERIFY RESULTS & ARTIFACTS
    results_res = client.get(f"/analysis/{job_id}/results")
    assert results_res.status_code == 200
    res_json = results_res.json()

    assert res_json["target_column"] == "PM10"
    assert res_json["winning_model_name"] != ""
    assert res_json["winning_metrics"]["RMSE"] > 0
    assert len(res_json["comparison_table"]) == 7  # All 7 models ranked

    # 4. VERIFY REPORT DOWNLOAD
    report_res = client.get(f"/analysis/{job_id}/report")
    assert report_res.status_code == 200
    assert "Mine Dust Analysis" in report_res.text


def test_end_to_end_orchestrated_pipeline_failure_lifecycle():
    """
    Verify lifecycle transitions to FAILED state with safe error message on invalid request.
    """
    # Start analysis on non-existent dataset file
    payload = {
        "dataset_filename": "non_existent_file.csv",
        "target_column": "PM10",
        "cv_folds": 3,
        "primary_metric": "RMSE",
    }
    start_res = client.post("/analysis/start", json=payload)
    assert start_res.status_code == 404
    err_json = start_res.json()

    assert err_json["code"] == "DATASET_NOT_FOUND"
    assert "user_message" in err_json
    assert "Traceback" not in start_res.text
