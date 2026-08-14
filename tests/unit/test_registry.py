"""
Unit Tests for ModelArtifactRegistry.
"""

from pathlib import Path
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.modeling.registry import ModelArtifactRegistry
from src.modeling.selector import BestModelResult
from src.utils.exceptions import ModelArtifactError


def test_save_and_load_model_artifacts_immutability(tmp_path):
    """Verify saving v1.0.0 and v1.0.1 and proving v1.0.0 remains accessible."""
    registry = ModelArtifactRegistry(base_artifact_dir=tmp_path)

    # 1. First Model Artifact v1.0.0 (Linear Regression)
    lr_model = LinearRegression()
    res_v1 = BestModelResult(
        best_model_key="linear_regression",
        best_model_name="Linear Regression",
        best_model_instance=lr_model,
        best_model_metrics={"MAE": 8.0, "RMSE": 12.0, "R2": 0.75, "R": 0.86},
        comparison_table=None,
        selection_reason="Initial baseline model",
    )

    v1_version = registry.save_artifact(
        target="PM10",
        best_model_result=res_v1,
        feature_names=["Temperature", "Wind_speed"],
        dataset_hash="hash_v1_abc123",
        job_id="job-v1",
    )

    assert v1_version == "v1.0.0"
    assert (tmp_path / "PM10" / "v1.0.0" / "metadata.json").exists()
    assert (tmp_path / "PM10" / "v1.0.0" / "model.joblib").exists()

    # 2. Second Model Artifact v1.0.1 (Random Forest)
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    res_v2 = BestModelResult(
        best_model_key="random_forest",
        best_model_name="Random Forest Regressor",
        best_model_instance=rf_model,
        best_model_metrics={"MAE": 5.0, "RMSE": 8.0, "R2": 0.88, "R": 0.94},
        comparison_table=None,
        selection_reason="Improved non-linear model",
    )

    v2_version = registry.save_artifact(
        target="PM10",
        best_model_result=res_v2,
        feature_names=["Temperature", "Wind_speed"],
        dataset_hash="hash_v2_xyz456",
        job_id="job-v2",
    )

    assert v2_version == "v1.0.1"
    assert (tmp_path / "PM10" / "v1.0.1" / "metadata.json").exists()

    # 3. PROOF: Verify old version v1.0.0 remains intact and loadable!
    loaded_v1 = registry.load_artifact(target="PM10", version="v1.0.0")
    assert loaded_v1["version"] == "v1.0.0"
    assert loaded_v1["metadata"]["model_key"] == "linear_regression"
    assert loaded_v1["metadata"]["dataset_hash"] == "hash_v1_abc123"
    assert loaded_v1["metadata"]["metrics"]["RMSE"] == 12.0

    # 4. Verify latest version resolves to v1.0.1
    loaded_latest = registry.load_artifact(target="PM10", version="latest")
    assert loaded_latest["version"] == "v1.0.1"
    assert loaded_latest["metadata"]["model_key"] == "random_forest"
    assert loaded_latest["metadata"]["dataset_hash"] == "hash_v2_xyz456"

    # 5. Verify Metadata Content
    meta = loaded_latest["metadata"]
    assert "git_commit" in meta
    assert "jenkins_build_number" in meta
    assert "timestamp" in meta
    assert meta["preprocessing_version"] == "2.0.0"


def test_overwriting_version_raises_exception(tmp_path):
    """Verify attempting to overwrite an existing version directory raises exception."""
    registry = ModelArtifactRegistry(base_artifact_dir=tmp_path)
    res = BestModelResult(
        best_model_key="linear_regression",
        best_model_name="Linear Regression",
        best_model_instance=LinearRegression(),
        best_model_metrics={"MAE": 8.0, "RMSE": 12.0, "R2": 0.75, "R": 0.86},
        comparison_table=None,
        selection_reason="Test",
    )

    registry.save_artifact(target="PM2.5", best_model_result=res, feature_names=["f1"], dataset_hash="h1", version="v1.0.0")

    with pytest.raises(ModelArtifactError):
        registry.save_artifact(target="PM2.5", best_model_result=res, feature_names=["f1"], dataset_hash="h1", version="v1.0.0")
