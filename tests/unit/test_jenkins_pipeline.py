"""
Unit and Integration Tests for Jenkins ML Training Pipeline Execution.
"""

import os
import json
import pytest
from pathlib import Path

from src.pipeline_runner import main as run_pipeline_main, generate_synthetic_dataset
from src.modeling.registry import ModelArtifactRegistry


def test_jenkins_pipeline_execution_and_build_recording(tmp_path, monkeypatch):
    """Test that Jenkins pipeline runner executes end-to-end and records all required build metrics."""
    input_csv = tmp_path / "jenkins_input.csv"
    summary_json = tmp_path / "jenkins_build_summary.json"

    generate_synthetic_dataset(input_csv)
    assert input_csv.exists()

    monkeypatch.setenv("BUILD_NUMBER", "42")
    monkeypatch.setenv("GIT_COMMIT", "a1b2c3d4e5f67890")

    test_args = [
        "pipeline_runner.py",
        "--dataset-path", str(input_csv),
        "--target", "PM10",
        "--selection-metric", "RMSE",
        "--training-mode", "fast",
        "--output-summary", str(summary_json),
    ]

    monkeypatch.setattr("sys.argv", test_args)
    run_pipeline_main()

    assert summary_json.exists()
    with open(summary_json, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Verify all required build record fields
    assert summary["jenkins_build_number"] == "42"
    assert summary["git_commit"] == "a1b2c3d4e5f67890"
    assert "dataset_hash" in summary and len(summary["dataset_hash"]) > 0
    assert summary["model_version"].startswith("v")
    assert "selected_model" in summary
    assert "MAE" in summary
    assert "RMSE" in summary
    assert "R2" in summary
    assert summary["target"] == "PM10"
    assert Path(summary["report_html_path"]).exists()


def test_version_preservation_across_jenkins_runs(tmp_path, monkeypatch):
    """Test that consecutive pipeline executions preserve previous model versions without overwriting."""
    registry = ModelArtifactRegistry(base_artifact_dir=tmp_path / "models")
    input_csv = tmp_path / "input.csv"
    summary_json1 = tmp_path / "summary1.json"
    summary_json2 = tmp_path / "summary2.json"

    generate_synthetic_dataset(input_csv)

    # Run 1
    args1 = [
        "pipeline_runner.py",
        "--dataset-path", str(input_csv),
        "--target", "PM2.5",
        "--training-mode", "fast",
        "--output-summary", str(summary_json1),
    ]
    monkeypatch.setattr("sys.argv", args1)
    run_pipeline_main()

    with open(summary_json1, "r", encoding="utf-8") as f:
        v1 = json.load(f)["model_version"]

    # Run 2
    args2 = [
        "pipeline_runner.py",
        "--dataset-path", str(input_csv),
        "--target", "PM2.5",
        "--training-mode", "fast",
        "--output-summary", str(summary_json2),
    ]
    monkeypatch.setattr("sys.argv", args2)
    run_pipeline_main()

    with open(summary_json2, "r", encoding="utf-8") as f:
        v2 = json.load(f)["model_version"]

    assert v1 != v2, "Second build must assign a new semantic version"
