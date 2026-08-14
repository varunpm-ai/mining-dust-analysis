"""
Centralized Configuration Loader for Mine Dust Analysis Platform.

Supports environment variables (.env), YAML configuration files (config.yaml, schema.yaml),
and default fallbacks with strict error checking.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from dotenv import load_dotenv

from src.utils.exceptions import ConfigurationError

# Auto-load .env file from working directory if present
load_dotenv()


class AppConfig:
    """
    Central application configuration manager.
    """

    def __init__(self, config_dict: Dict[str, Any], schema_dict: Dict[str, Any]):
        self.raw_config = config_dict
        self.raw_schema = schema_dict

        # System settings
        system_cfg = config_dict.get("system", {})
        self.app_name: str = os.getenv("APP_NAME", system_cfg.get("name", "Mine Dust Analysis Production MLOps Platform"))
        self.env: str = os.getenv("ENV", system_cfg.get("environment", "development"))
        self.version: str = system_cfg.get("version", "2.0.0")

        # Pipeline settings
        pipeline_cfg = config_dict.get("pipeline", {})
        self.random_state: int = int(pipeline_cfg.get("random_state", 42))
        self.test_size: float = float(pipeline_cfg.get("test_size", 0.2))
        self.cv_folds: int = int(pipeline_cfg.get("cv_folds", 5))

        # Modeling settings
        model_cfg = config_dict.get("modeling", {})
        self.target_columns: List[str] = model_cfg.get("targets", ["PM10", "PM2.5"])
        self.primary_metric: str = model_cfg.get("primary_metric", "RMSE")
        self.secondary_metric: str = model_cfg.get("secondary_metric", "MAE")
        self.supporting_metric: str = model_cfg.get("supporting_metric", "R2")
        self.models_to_train: List[str] = model_cfg.get(
            "models_to_train",
            ["linear_regression", "decision_tree", "random_forest", "bagging", "svr", "xgboost", "voting"],
        )

        # Artifact directories
        art_cfg = config_dict.get("artifacts", {})
        self.artifacts_dir: Path = Path(os.getenv("ARTIFACTS_DIR", art_cfg.get("base_dir", "artifacts")))
        self.models_dir: Path = Path(art_cfg.get("models_dir", str(self.artifacts_dir / "models")))
        self.reports_dir: Path = Path(art_cfg.get("reports_dir", str(self.artifacts_dir / "reports")))
        self.metrics_dir: Path = Path(art_cfg.get("metrics_dir", str(self.artifacts_dir / "metrics")))
        self.datasets_dir: Path = Path(art_cfg.get("datasets_dir", str(self.artifacts_dir / "datasets")))

        # Logging settings
        log_cfg = config_dict.get("logging", {})
        self.log_level: str = os.getenv("LOG_LEVEL", log_cfg.get("level", "INFO")).upper()
        self.log_file: Path = Path(os.getenv("LOG_FILE", log_cfg.get("log_file", "logs/pipeline.log")))

        # Schema rules
        self.schema_targets: List[Dict[str, Any]] = schema_dict.get("targets", [])
        self.legacy_features: List[Dict[str, Any]] = schema_dict.get("legacy_features", [])
        self.operational_features: List[Dict[str, Any]] = schema_dict.get("operational_features", [])

    def ensure_directories_exist(self) -> None:
        """Create configured artifact and log directories if they do not exist."""
        for path in [
            self.artifacts_dir,
            self.models_dir,
            self.reports_dir,
            self.metrics_dir,
            self.datasets_dir,
            self.log_file.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)


_CONFIG_INSTANCE: Optional[AppConfig] = None


def load_config(
    config_path: Optional[str] = None,
    schema_path: Optional[str] = None,
) -> AppConfig:
    """
    Load YAML configuration files and environment overrides.
    
    Args:
        config_path (str, optional): Path to config.yaml. Defaults to CONFIG_PATH env or configs/config.yaml.
        schema_path (str, optional): Path to schema.yaml. Defaults to SCHEMA_PATH env or configs/schema.yaml.
        
    Returns:
        AppConfig: Parsed configuration object.
    """
    global _CONFIG_INSTANCE

    resolved_config_path = Path(
        config_path or os.getenv("CONFIG_PATH", "configs/config.yaml")
    )
    resolved_schema_path = Path(
        schema_path or os.getenv("SCHEMA_PATH", "configs/schema.yaml")
    )

    config_dict: Dict[str, Any] = {}
    if resolved_config_path.exists():
        try:
            with open(resolved_config_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(
                message=f"Failed to parse config YAML at {resolved_config_path}: {e}",
                user_message="System configuration file is corrupted or unreadable.",
            )
    elif config_path is not None:
        raise ConfigurationError(
            message=f"Specified config file does not exist: {resolved_config_path}",
            user_message="Configuration file missing.",
        )

    schema_dict: Dict[str, Any] = {}
    if resolved_schema_path.exists():
        try:
            with open(resolved_schema_path, "r", encoding="utf-8") as f:
                schema_dict = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(
                message=f"Failed to parse schema YAML at {resolved_schema_path}: {e}",
                user_message="Dataset schema configuration file is corrupted or unreadable.",
            )
    elif schema_path is not None:
        raise ConfigurationError(
            message=f"Specified schema file does not exist: {resolved_schema_path}",
            user_message="Schema configuration file missing.",
        )

    _CONFIG_INSTANCE = AppConfig(config_dict=config_dict, schema_dict=schema_dict)
    _CONFIG_INSTANCE.ensure_directories_exist()
    return _CONFIG_INSTANCE


def get_config() -> AppConfig:
    """Get singleton configuration instance or load default if uninitialized."""
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        return load_config()
    return _CONFIG_INSTANCE
