"""
CLI Pipeline Runner for Production ML Training Pipeline.

Provides command-line execution for Jenkins CI/CD pipelines with configurable targets,
metrics, training modes, and detailed build metric recording.
"""

import sys
import os
from pathlib import Path

# Ensure project root is at index 0 of sys.path before any package imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import pandas as pd

from api.services.pipeline_service import PipelineService
from src.utils.logger import setup_logger

logger = setup_logger("src.pipeline_runner")


def generate_synthetic_dataset(output_path: Path) -> Path:
    """Generate a clean synthetic dataset for automated CI/CD pipeline execution."""
    logger.info(f"Generating synthetic dataset at {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 150
    dates = pd.date_range(start="2024-01-01", periods=n_samples, freq="D")

    locations = ["Mining Zone A", "Mining Zone B", "Processing Area"]
    seasons = ["Monsoon", "Summer", "Winter"]

    df = pd.DataFrame({
        "YEAR": dates.year,
        "SEASON": [np.random.choice(seasons) for _ in range(n_samples)],
        "MONTH": dates.month_name(),
        "LOCATION": [np.random.choice(locations) for _ in range(n_samples)],
        "WEEK": dates.isocalendar().week.astype(int),
        "Temperature": np.random.uniform(15.0, 42.0, n_samples).round(1),
        "Relative_humidity": np.random.uniform(30.0, 90.0, n_samples).round(1),
        "RF-CUMULATIVE": np.random.exponential(15.0, n_samples).round(1),
        "Wind_speed": np.random.uniform(1.0, 15.0, n_samples).round(1),
        "PM10": np.random.uniform(40.0, 220.0, n_samples).round(1),
        "PM2.5": np.random.uniform(20.0, 110.0, n_samples).round(1),
    })

    df.to_csv(output_path, index=False)
    logger.info(f"Synthetic dataset saved with {len(df)} rows.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Mine Dust Analysis ML Training Pipeline CLI Runner")
    parser.add_argument("--dataset-path", type=str, default="artifacts/datasets/uploads/jenkins_input.csv", help="Path to input dataset CSV/XLSX")
    parser.add_argument("--target", type=str, default="PM10", choices=["PM10", "PM2.5"], help="Target column for training")
    parser.add_argument("--selection-metric", type=str, default="RMSE", choices=["RMSE", "MAE", "R2"], help="Primary model selection metric")
    parser.add_argument("--training-mode", type=str, default="full", choices=["fast", "full", "cross_validation"], help="Model training mode")
    parser.add_argument("--output-summary", type=str, default="artifacts/jenkins_build_summary.json", help="Path to write build summary JSON")

    args = parser.parse_args()

    logger.info("=== Starting Jenkins ML Training Pipeline Execution ===")
    logger.info(f"Target: {args.target}")
    logger.info(f"Selection Metric: {args.selection_metric}")
    logger.info(f"Training Mode: {args.training_mode}")

    dataset_path = Path(args.dataset_path)

    # If dataset path does not exist, generate synthetic dataset for CI/CD
    if not dataset_path.exists():
        dataset_path = generate_synthetic_dataset(dataset_path)

    # Initialize Pipeline Service
    upload_dir = dataset_path.parent
    service = PipelineService(upload_dir=upload_dir)

    # Create & execute job
    job_id = service.create_job(dataset_filename=dataset_path.name, target_column=args.target)
    cv_folds = 3 if args.training_mode == "fast" else 5
    results = service.execute_pipeline(job_id=job_id, cv_folds=cv_folds, primary_metric=args.selection_metric)

    # Load registry metadata for build recording
    from src.modeling.registry import ModelArtifactRegistry
    registry = ModelArtifactRegistry()
    loaded_artifact = registry.load_artifact(target=args.target, version=results["artifact_version"])
    metadata = loaded_artifact["metadata"]

    build_summary = {
        "git_commit": metadata.get("git_commit", os.getenv("GIT_COMMIT", "N/A")),
        "jenkins_build_number": metadata.get("jenkins_build_number", os.getenv("BUILD_NUMBER", "N/A")),
        "dataset_hash": metadata.get("dataset_hash", "N/A"),
        "model_version": metadata.get("version", results["artifact_version"]),
        "selected_model": metadata.get("model_name", results["winning_model_name"]),
        "MAE": metadata["metrics"].get("MAE", 0.0),
        "RMSE": metadata["metrics"].get("RMSE", 0.0),
        "R2": metadata["metrics"].get("R2", 0.0),
        "target": args.target,
        "selection_metric": args.selection_metric,
        "training_mode": args.training_mode,
        "report_html_path": results["report_html_path"],
        "job_id": job_id,
    }

    summary_path = Path(args.output_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(build_summary, f, indent=4)

    logger.info("=== Build Recording Summary ===")
    logger.info(json.dumps(build_summary, indent=2))
    logger.info(f"Build summary exported to {summary_path}")
    print("\n" + json.dumps(build_summary, indent=2))


if __name__ == "__main__":
    main()
