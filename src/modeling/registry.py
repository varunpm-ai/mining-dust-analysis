"""
Production Model Artifact Registry and Persistence Manager.

Saves and loads immutable, semantic-versioned model artifacts with full lineage metadata
(dataset hash, features, metrics, preprocessing version, git commit, Jenkins build number).
"""

import os
import json
import joblib
import subprocess
import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

from src.modeling.selector import BestModelResult
from src.utils.exceptions import ModelArtifactError
from src.utils.logger import setup_logger

logger = setup_logger("modeling.registry")


class ModelArtifactRegistry:
    """
    Manages immutable model artifact versioning, serialization, and retrieval.
    """

    PREPROCESSING_VERSION = "2.0.0"

    def __init__(self, base_artifact_dir: Optional[Union[str, Path]] = None):
        self.base_dir = Path(base_artifact_dir) if base_artifact_dir else Path("artifacts/models")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_git_commit(self) -> str:
        """Retrieve current Git commit hash if running in a Git repository."""
        try:
            cmd = ["git", "rev-parse", "HEAD"]
            commit = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("utf-8").strip()
            return commit
        except Exception:
            return "N/A (Not a Git repository)"

    def _get_jenkins_build(self) -> str:
        """Retrieve Jenkins build number from environment variables if present."""
        return os.getenv("BUILD_NUMBER", os.getenv("JENKINS_BUILD_NUMBER", "N/A"))

    def _get_next_version(self, target: str) -> str:
        """Determine next immutable semantic version string (e.g. v1.0.0, v1.0.1)."""
        target_dir = self.base_dir / target
        if not target_dir.exists():
            return "v1.0.0"

        existing_versions = []
        for p in target_dir.iterdir():
            if p.is_dir() and p.name.startswith("v"):
                existing_versions.append(p.name)

        if not existing_versions:
            return "v1.0.0"

        # Extract patch numbers for sorting (v1.0.X)
        latest_patch = 0
        for v_str in existing_versions:
            parts = v_str.replace("v", "").split(".")
            if len(parts) == 3:
                try:
                    patch = int(parts[2])
                    if patch > latest_patch:
                        latest_patch = patch
                except ValueError:
                    continue

        return f"v1.0.{latest_patch + 1}"

    def save_artifact(
        self,
        target: str,
        best_model_result: BestModelResult,
        feature_names: List[str],
        dataset_hash: str,
        encoder: Optional[Any] = None,
        transformer: Optional[Any] = None,
        version: Optional[str] = None,
        job_id: str = "N/A",
    ) -> str:
        """
        Save immutable model artifact directory with model file, metadata, and preprocessors.

        Returns:
            str: Assigned version string (e.g., 'v1.0.0').
        """
        extra_ctx = {"job_id": job_id}
        version_str = version or self._get_next_version(target)
        version_dir = self.base_dir / target / version_str

        if version_dir.exists():
            raise ModelArtifactError(
                message=f"Version directory '{version_dir}' already exists. Overwriting immutable versions is forbidden.",
                user_message="Model version conflict detected. Immutable artifact storage prevents overwriting.",
            )

        version_dir.mkdir(parents=True, exist_ok=False)

        # 1. Serialize Model File
        model_filepath = version_dir / "model.joblib"
        joblib.dump(best_model_result.best_model_instance, model_filepath)

        # 2. Serialize Preprocessors if provided
        if encoder:
            joblib.dump(encoder, version_dir / "encoder.joblib")
        if transformer:
            joblib.dump(transformer, version_dir / "transformer.joblib")

        # 3. Construct Complete Lineage Metadata JSON
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        metadata = {
            "version": version_str,
            "target": target,
            "model_key": best_model_result.best_model_key,
            "model_name": best_model_result.best_model_name,
            "metrics": best_model_result.best_model_metrics,
            "selection_reason": best_model_result.selection_reason,
            "features": feature_names,
            "dataset_hash": dataset_hash,
            "preprocessing_version": self.PREPROCESSING_VERSION,
            "timestamp": timestamp_str,
            "git_commit": self._get_git_commit(),
            "jenkins_build_number": self._get_jenkins_build(),
            "job_id": job_id,
        }

        metadata_filepath = version_dir / "metadata.json"
        with open(metadata_filepath, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Successfully saved immutable model artifact target '{target}' version '{version_str}' at {version_dir}", extra=extra_ctx)
        return version_str

    def list_versions(self, target: str) -> List[str]:
        """List all saved version strings for a target in ascending order."""
        target_dir = self.base_dir / target
        if not target_dir.exists():
            return []

        versions = [p.name for p in target_dir.iterdir() if p.is_dir() and p.name.startswith("v")]
        return sorted(versions)

    def load_artifact(self, target: str, version: str = "latest") -> Dict[str, Any]:
        """
        Load specified model version artifact and metadata.

        Args:
            target (str): Target name ("PM10" or "PM2.5").
            version (str): Exact version string (e.g. "v1.0.0") or "latest".

        Returns:
            Dict[str, Any]: Loaded model instance, metadata, and preprocessors.
        """
        versions = self.list_versions(target)
        if not versions:
            raise ModelArtifactError(
                message=f"No artifacts found for target '{target}'.",
                user_message=f"No trained model versions exist for target '{target}'.",
            )

        target_version = versions[-1] if version.lower() == "latest" else version
        version_dir = self.base_dir / target / target_version

        if not version_dir.exists():
            raise ModelArtifactError(
                message=f"Requested model artifact version '{target_version}' for target '{target}' does not exist.",
                user_message=f"Model version '{target_version}' not found.",
            )

        model_path = version_dir / "model.joblib"
        metadata_path = version_dir / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            raise ModelArtifactError(
                message=f"Corrupted model artifact directory at '{version_dir}'.",
                user_message="Model artifact files are missing or corrupted.",
            )

        model_instance = joblib.load(model_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        encoder_path = version_dir / "encoder.joblib"
        transformer_path = version_dir / "transformer.joblib"

        encoder = joblib.load(encoder_path) if encoder_path.exists() else None
        transformer = joblib.load(transformer_path) if transformer_path.exists() else None

        return {
            "target": target,
            "version": target_version,
            "model_instance": model_instance,
            "metadata": metadata,
            "encoder": encoder,
            "transformer": transformer,
        }
