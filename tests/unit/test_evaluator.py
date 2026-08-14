"""
Unit Tests for ModelEvaluator.
"""

import numpy as np
import pytest

from src.modeling.evaluator import ModelEvaluator
from src.utils.exceptions import ModelEvaluationError


def test_model_evaluator_metrics_calculation():
    """Verify MAE, RMSE, R2, and R evaluation metrics calculations."""
    y_true = np.array([50.0, 60.0, 70.0, 80.0, 90.0])
    y_pred = np.array([52.0, 58.0, 71.0, 79.0, 92.0])

    metrics = ModelEvaluator.evaluate(y_true, y_pred)

    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "R" in metrics

    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert metrics["R2"] > 0.9  # High agreement
    assert metrics["R"] > 0.9


def test_model_evaluator_empty_arrays_raises_exception():
    """Verify evaluating empty arrays raises ModelEvaluationError."""
    with pytest.raises(ModelEvaluationError):
        ModelEvaluator.evaluate(np.array([]), np.array([]))
