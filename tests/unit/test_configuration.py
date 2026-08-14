"""
Unit Tests for Configuration Management System.
"""

import os
from pathlib import Path
import pytest
import yaml

from src.utils.configuration import AppConfig, load_config, get_config
from src.utils.exceptions import ConfigurationError


def test_load_default_configuration():
    """Verify loading default config.yaml and schema.yaml."""
    config = load_config()
    assert isinstance(config, AppConfig)
    assert config.app_name is not None
    assert config.version == "2.0.0"
    assert config.random_state == 42
    assert "PM10" in config.target_columns
    assert "PM2.5" in config.target_columns
    assert config.primary_metric == "RMSE"


def test_environment_variable_overrides(tmp_path, monkeypatch):
    """Verify environment variables override configuration defaults."""
    monkeypatch.setenv("APP_NAME", "Custom Mine Dust Test App")
    monkeypatch.setenv("ENV", "staging")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    config = load_config()
    assert config.app_name == "Custom Mine Dust Test App"
    assert config.env == "staging"
    assert config.log_level == "DEBUG"


def test_custom_yaml_paths(tmp_path):
    """Verify loading from custom config and schema YAML files."""
    custom_config = tmp_path / "custom_config.yaml"
    custom_schema = tmp_path / "custom_schema.yaml"

    config_data = {
        "system": {"name": "Test Platform", "environment": "test", "version": "2.1.0"},
        "pipeline": {"random_state": 123, "test_size": 0.3},
        "modeling": {"targets": ["PM10"], "primary_metric": "MAE"},
    }
    schema_data = {
        "targets": [{"name": "PM10", "type": "numeric"}],
        "legacy_features": [],
    }

    with open(custom_config, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f)
    with open(custom_schema, "w", encoding="utf-8") as f:
        yaml.dump(schema_data, f)

    config = load_config(config_path=str(custom_config), schema_path=str(custom_schema))
    assert config.version == "2.1.0"
    assert config.random_state == 123
    assert config.test_size == 0.3
    assert config.target_columns == ["PM10"]
    assert config.primary_metric == "MAE"

    # Restore default singleton config for subsequent tests
    load_config()


def test_invalid_yaml_raises_configuration_error(tmp_path):
    """Verify invalid YAML raises ConfigurationError."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("system: [unclosed_list", encoding="utf-8")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(config_path=str(invalid_yaml))
    assert exc_info.value.code == "CONFIG_ERROR"
