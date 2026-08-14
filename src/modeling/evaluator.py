"""
Model Evaluation and Metrics Calculation Module.

Calculates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE),
R-squared (R2), and Pearson correlation (R).
"""

from typing import Dict, Any, Tuple
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.exceptions import ModelEvaluationError
from src.utils.logger import setup_logger

logger = setup_logger("modeling.evaluator")


class ModelEvaluator:
    """
    Evaluates regression model predictions using MAE, RMSE, R2, and R.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute standard regression evaluation metrics.

        Args:
            y_true (np.ndarray): Actual target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            Dict[str, float]: Metrics dictionary containing MAE, RMSE, R2, R.
        """
        y_true_arr = np.asarray(y_true, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
            raise ModelEvaluationError(
                message="Cannot evaluate model metrics on empty prediction arrays.",
                user_message="Evaluation array is empty.",
            )

        if len(y_true_arr) != len(y_pred_arr):
            raise ModelEvaluationError(
                message=f"Shape mismatch in evaluation: y_true ({len(y_true_arr)}) vs y_pred ({len(y_pred_arr)})",
                user_message="Prediction size mismatch during model evaluation.",
            )

        # 1. MAE
        mae = float(mean_absolute_error(y_true_arr, y_pred_arr))

        # 2. RMSE
        rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

        # 3. R2 Score
        r2 = float(r2_score(y_true_arr, y_pred_arr))

        # 4. Pearson R Correlation Coefficient
        try:
            if np.std(y_true_arr) > 0 and np.std(y_pred_arr) > 0:
                r_val, _ = pearsonr(y_true_arr, y_pred_arr)
                r = float(r_val)
            else:
                r = 0.0
        except Exception:
            r = 0.0

        metrics = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
            "R": round(r, 4),
        }

        return metrics
