"""
End-to-End Streamlit REST Communication & Jenkins CLI Runner Tests (Phase 16).
Tests UI client communication with FastAPI backend endpoints and headless CLI pipeline runner.
"""

import json
import pytest
from pathlib import Path

from fastapi.testclient import TestClient
from api.main import app
from src.pipeline_runner import main as run_pipeline_main, generate_synthetic_dataset
from tests.fixtures.synthetic_datasets import create_synthetic_csv

client = TestClient(app)


def test_streamlit_to_fastapi_rest_communication_flow(tmp_path):
    """Simulate complete Streamlit UI interaction sequence via FastAPI REST endpoints."""
    csv_file = create_synthetic_csv(tmp_path / "ui_test.csv")

    # 1. Upload Dataset via REST
    with open(csv_file, "rb") as f:
        res_upload = client.post("/datasets/upload", files={"file": ("ui_test.csv", f, "text/csv")})
    assert res_upload.status_code == 200
    upload_data = res_upload.json()
    assert "filename" in upload_data

    # 2. Start Analysis Job via REST
    res_start = client.post(
        "/analysis/start",
        json={"dataset_filename": upload_data["filename"], "target_column": "PM10"},
    )
    assert res_start.status_code == 200
    job_id = res_start.json()["job_id"]

    # 3. Check Job Status via REST
    res_status = client.get(f"/analysis/{job_id}")
    assert res_status.status_code == 200
    assert res_status.json()["status"] == "COMPLETED"

    # 4. Fetch Job Results via REST
    res_results = client.get(f"/analysis/{job_id}/results")
    assert res_results.status_code == 200
    results_data = res_results.json()["results"]
    assert "winning_model_name" in results_data

    # 5. Fetch Models List via REST
    res_models = client.get("/models")
    assert res_models.status_code == 200
    assert isinstance(res_models.json()["models"], list)


def test_jenkins_cli_pipeline_runner_execution(tmp_path, monkeypatch):
    """Test CLI pipeline runner for headless Jenkins CI/CD build execution."""
    dataset_csv = tmp_path / "jenkins_ci_input.csv"
    summary_json = tmp_path / "build_summary.json"

    generate_synthetic_dataset(dataset_csv)
    monkeypatch.setenv("BUILD_NUMBER", "101")
    monkeypatch.setenv("GIT_COMMIT", "fedcba9876543210")

    cli_args = [
        "pipeline_runner.py",
        "--dataset-path", str(dataset_csv),
        "--target", "PM2.5",
        "--selection-metric", "MAE",
        "--training-mode", "fast",
        "--output-summary", str(summary_json),
    ]

    monkeypatch.setattr("sys.argv", cli_args)
    run_pipeline_main()

    assert summary_json.exists()
    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert summary["jenkins_build_number"] == "101"
    assert summary["git_commit"] == "fedcba9876543210"
    assert summary["target"] == "PM2.5"
    assert summary["selection_metric"] == "MAE"
    assert "MAE" in summary
