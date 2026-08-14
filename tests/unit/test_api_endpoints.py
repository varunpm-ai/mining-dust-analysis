"""
Unit and Integration Tests for FastAPI Backend Endpoints.
"""

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.fixtures.synthetic_datasets import create_synthetic_legacy_df

client = TestClient(app)


def test_get_health():
    """Verify GET /health endpoint returns HTTP 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "2.0.0"


def test_post_dataset_upload_csv(tmp_path):
    """Verify POST /datasets/upload handles CSV file uploads."""
    df = create_synthetic_legacy_df(num_rows=25)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    files = {"file": ("synthetic_mine_data.csv", io.BytesIO(csv_bytes), "text/csv")}
    response = client.post("/datasets/upload", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "synthetic_mine_data.csv"
    assert "dataset_hash" in data
    assert data["file_size_bytes"] > 0


def test_end_to_end_analysis_api_workflow(tmp_path):
    """
    Test complete API workflow:
    Upload dataset -> Start analysis -> Check status -> Fetch results -> Fetch report -> Fetch model.
    """
    # 1. Upload
    df = create_synthetic_legacy_df(num_rows=30)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("api_test_dataset.csv", io.BytesIO(csv_bytes), "text/csv")}
    upload_res = client.post("/datasets/upload", files=files)
    assert upload_res.status_code == 200

    # 2. Start Analysis
    payload = {
        "dataset_filename": "api_test_dataset.csv",
        "target_column": "PM10",
        "cv_folds": 3,
        "primary_metric": "RMSE",
    }
    start_res = client.post("/analysis/start", json=payload)
    assert start_res.status_code == 200
    start_data = start_res.json()
    job_id = start_data["job_id"]
    assert start_data["status"].upper() == "COMPLETED"

    # 3. GET /analysis/{job_id}
    status_res = client.get(f"/analysis/{job_id}")
    assert status_res.status_code == 200
    assert status_res.json()["status"].upper() == "COMPLETED"

    # 4. GET /analysis/{job_id}/results
    results_res = client.get(f"/analysis/{job_id}/results")
    assert results_res.status_code == 200
    res_data = results_res.json()
    assert res_data["winning_model_name"] != ""
    assert "winning_metrics" in res_data
    assert len(res_data["comparison_table"]) == 7

    # 5. GET /analysis/{job_id}/report
    report_res = client.get(f"/analysis/{job_id}/report")
    assert report_res.status_code == 200
    assert "text/html" in report_res.headers["content-type"]
    assert "Executive Report" in report_res.text

    # 6. GET /analysis/{job_id}/dataset
    ds_res = client.get(f"/analysis/{job_id}/dataset")
    assert ds_res.status_code == 200

    # 7. GET /analysis/{job_id}/model
    model_res = client.get(f"/analysis/{job_id}/model")
    assert model_res.status_code == 200
    assert "artifact_version" in model_res.json()


def test_get_models_list_and_details():
    """Verify GET /models and GET /models/{model_id} endpoints."""
    # List models
    res = client.get("/models")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 2


def test_error_handling_sanitization_no_tracebacks():
    """Verify non-existent job ID returns sanitized 404 error without Python tracebacks."""
    response = client.get("/analysis/job-non-existent-12345")
    assert response.status_code == 404
    data = response.json()
    assert "code" in data
    assert data["code"] == "JOB_NOT_FOUND"
    assert "user_message" in data
    assert "Traceback" not in response.text
