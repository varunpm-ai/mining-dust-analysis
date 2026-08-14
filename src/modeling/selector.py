"""
Automatic Model Evaluation, Ranking, and Selection Module.

Provides deterministic model selection using out-of-fold validation metrics.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass

from src.utils.exceptions import ModelSelectionError
from src.utils.logger import setup_logger

logger = setup_logger("modeling.selector")


@dataclass
class BestModelResult:
    """Dataclass holding winning model artifact and metrics."""
    best_model_key: str
    best_model_name: str
    best_model_instance: Any
    best_model_metrics: Dict[str, float]
    comparison_table: pd.DataFrame
    selection_reason: str


class ModelSelector:
    """
    Deterministic model selector based on validation performance metrics.
    """

    def __init__(
        self,
        primary_metric: str = "RMSE",
        secondary_metric: str = "MAE",
        tertiary_metric: str = "R",
    ):
        self.primary_metric = primary_metric.upper()
        self.secondary_metric = secondary_metric.upper()
        self.tertiary_metric = tertiary_metric.upper()

        # Define metric optimization directions (True for ascending/lower is better)
        self.metric_ascending = {
            "RMSE": True,
            "MAE": True,
            "R2": False,
            "R": False,
        }

    def select_best_model(
        self,
        model_results: Dict[str, Dict[str, Any]],
        job_id: str = "N/A",
    ) -> BestModelResult:
        """
        Rank and select the optimal regression model deterministically.

        Args:
            model_results: Dict mapping model_key -> { 'model_instance': ..., 'metrics': {...}, 'model_name': ... }
            job_id: Pipeline execution job ID.

        Returns:
            BestModelResult: Winning model artifact, metrics, and comparison table.
        """
        extra_ctx = {"job_id": job_id}

        if not model_results:
            raise ModelSelectionError(
                message="Cannot perform model selection on empty model results dictionary.",
                user_message="Model selection failed because no trained models were provided.",
            )

        records = []
        for key, res in model_results.items():
            metrics = res.get("metrics", {})
            
            # Safeguard: Verify metrics are from out-of-fold validation, not training
            if "RMSE" not in metrics or "MAE" not in metrics:
                raise ModelSelectionError(
                    message=f"Model '{key}' lacks required out-of-fold validation metrics.",
                    user_message="Model evaluation metrics are incomplete.",
                )

            records.append({
                "model_key": key,
                "model_name": res.get("model_name", key),
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "R": metrics["R"],
                "_instance": res.get("model_instance"),
            })

        df = pd.DataFrame(records)

        # Build sorting criteria
        sort_cols = [self.primary_metric, self.secondary_metric, self.tertiary_metric, "model_name"]
        sort_ascending = [
            self.metric_ascending.get(self.primary_metric, True),
            self.metric_ascending.get(self.secondary_metric, True),
            self.metric_ascending.get(self.tertiary_metric, False),
            True,  # Deterministic alphabetical tie breaker
        ]

        df_sorted = df.sort_values(by=sort_cols, ascending=sort_ascending).reset_index(drop=True)
        df_sorted.index = df_sorted.index + 1
        df_sorted.index.name = "Rank"

        winner_row = df_sorted.iloc[0]
        winning_key = str(winner_row["model_key"])
        winning_name = str(winner_row["model_name"])
        winning_instance = winner_row["_instance"]

        winning_metrics = {
            "RMSE": float(winner_row["RMSE"]),
            "MAE": float(winner_row["MAE"]),
            "R2": float(winner_row["R2"]),
            "R": float(winner_row["R"]),
        }

        # Clean display table (drop internal object column)
        comparison_table = df_sorted.drop(columns=["_instance"])

        reason = (
            f"Selected '{winning_name}' as the winning model because it achieved the optimal "
            f"{self.primary_metric} ({winning_metrics[self.primary_metric]:.4f}) "
            f"and secondary {self.secondary_metric} ({winning_metrics[self.secondary_metric]:.4f}) "
            f"on out-of-fold cross-validation."
        )

        logger.info(f"Model Selection Winner: {reason}", extra=extra_ctx)

        return BestModelResult(
            best_model_key=winning_key,
            best_model_name=winning_name,
            best_model_instance=winning_instance,
            best_model_metrics=winning_metrics,
            comparison_table=comparison_table,
            selection_reason=reason,
        )
