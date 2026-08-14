"""
Package Structure & Import Validation Test.
"""

import importlib
import pytest


@pytest.mark.parametrize(
    "package_name",
    [
        "src",
        "src.ingestion",
        "src.preprocessing",
        "src.analysis",
        "src.statistics",
        "src.modeling",
        "src.reporting",
        "src.utils",
        "api",
        "api.routes",
        "api.schemas",
        "api.services",
    ],
)
def test_package_import(package_name: str):
    """Verify that all core modules and subpackages can be imported cleanly."""
    module = importlib.import_module(package_name)
    assert module is not None


def test_fastapi_app_import():
    """Verify that FastAPI app instance initializes correctly."""
    from api.main import app
    assert app.title == "Mine Dust Analysis Production MLOps API"
