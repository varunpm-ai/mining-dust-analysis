"""
Unit Tests for ModelFactory.
"""

import numpy as np
import pytest

from src.modeling.models import ModelFactory
from src.utils.exceptions import ModelTrainingError


@pytest.mark.parametrize(
    "model_key",
    [
        "linear_regression",
        "decision_tree",
        "random_forest",
        "bagging",
        "svr",
        "xgboost",
        "voting",
    ],
)
def test_create_all_supported_models(model_key: str):
    """Verify instantiating all 7 required regression models."""
    model = ModelFactory.create_model(model_key, random_state=42)
    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_invalid_model_key_raises_exception():
    """Verify invalid model key raises ModelTrainingError."""
    with pytest.raises(ModelTrainingError) as exc_info:
        ModelFactory.create_model("unsupported_algorithm")
    assert exc_info.value.code == "MODEL_TRAINING_ERROR"
