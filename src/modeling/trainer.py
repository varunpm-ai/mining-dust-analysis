"""
Production Model Trainer Engine for Mine Dust Regression Pipeline.

Executes K-Fold Cross-Validation or Holdout Validation across all 7 regression models
without data leakage, returning structured metrics and trained estimators.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.modeling.models import ModelFactory
from src.modeling.evaluator import ModelEvaluator
from src.modeling.selector import ModelSelector, BestModelResult
from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.encoder import CategoricalEncoder
from src.preprocessing.transformer import FeatureTransformer
from src.utils.configuration import get_config, AppConfig
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import setup_logger

logger = setup_logger("modeling.trainer")


class ModelTrainer:
    """
    Production ML training engine for PM10 and PM2.5 regression models.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        models_to_train: Optional[List[str]] = None,
    ):
        self.config = config or get_config()
        self.models_to_train = models_to_train or self.config.models_to_train
        self.random_state = self.config.random_state

    def train_models_cv(
        self,
        df: pd.DataFrame,
        target_column: str,
        n_splits: int = 5,
        job_id: str = "N/A",
    ) -> Dict[str, Any]:
        """
        Train and evaluate all models using K-Fold Cross-Validation.

        Args:
            df (pd.DataFrame): Cleaned dataset.
            target_column (str): Target column ("PM10" or "PM2.5").
            n_splits (int): K-Fold splits count (default 5).
            job_id (str): Pipeline execution job ID.

        Returns:
            Dict[str, Any]: Comprehensive training report.
        """
        extra_ctx = {"job_id": job_id}

        if target_column not in df.columns:
            raise ModelTrainingError(
                message=f"Target column '{target_column}' missing for model training.",
                user_message=f"Target column '{target_column}' is missing from dataset.",
            )

        # Preprocessing: Categorical encoding and matrix preparation
        encoder = CategoricalEncoder()
        encoded_df, _ = encoder.fit_transform(df, job_id=job_id)

        transformer = FeatureTransformer(scaling_method="standard")
        X, y = transformer.prepare_x_y(encoded_df, target_column=target_column, job_id=job_id)

        if len(X) < 10:
            raise ModelTrainingError(
                message=f"Insufficient dataset rows ({len(X)}) for model training.",
                user_message="Dataset contains too few valid records for machine learning training.",
            )

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        training_results: Dict[str, Any] = {}
        comparison_records: List[Dict[str, Any]] = []

        logger.info(
            f"Starting ML Training Engine for target '{target_column}' with {len(self.models_to_train)} models across {n_splits}-fold CV.",
            extra=extra_ctx,
        )

        for model_key in self.models_to_train:
            try:
                oof_preds = np.zeros(len(X))
                fold_metrics: List[Dict[str, float]] = []

                for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    # Scale feature matrices inside fold to prevent leakage
                    X_train_scaled = transformer.fit_scale(X_train, job_id=job_id)
                    X_val_scaled = transformer.transform_scale(X_val, job_id=job_id)

                    # Instantiate fresh model for fold
                    fold_model = ModelFactory.create_model(model_key, random_state=self.random_state)
                    fold_model.fit(X_train_scaled, y_train)

                    val_pred = fold_model.predict(X_val_scaled)
                    oof_preds[val_idx] = val_pred

                    metrics = ModelEvaluator.evaluate(y_val.values, val_pred)
                    fold_metrics.append(metrics)

                # Fit final model on full dataset for persistence
                X_full_scaled = transformer.fit_scale(X, job_id=job_id)
                final_model = ModelFactory.create_model(model_key, random_state=self.random_state)
                final_model.fit(X_full_scaled, y)

                # Overall Out-Of-Fold Evaluation Metrics
                overall_metrics = ModelEvaluator.evaluate(y.values, oof_preds)

                # Feature Importances / Coefficients
                feature_imp_df = self._extract_feature_importance(final_model, X.columns.tolist())

                formatted_name = self._format_model_name(model_key)
                training_results[model_key] = {
                    "model_key": model_key,
                    "model_name": formatted_name,
                    "model_instance": final_model,
                    "metrics": overall_metrics,
                    "fold_metrics": fold_metrics,
                    "oof_predictions": oof_preds,
                    "feature_importance": feature_imp_df,
                }

                comparison_records.append({
                    "model_key": model_key,
                    "model_name": formatted_name,
                    "MAE": overall_metrics["MAE"],
                    "RMSE": overall_metrics["RMSE"],
                    "R2": overall_metrics["R2"],
                    "R": overall_metrics["R"],
                })

                logger.info(
                    f"Model '{formatted_name}' trained. CV Out-Of-Fold RMSE: {overall_metrics['RMSE']:.4f}, MAE: {overall_metrics['MAE']:.4f}, R²: {overall_metrics['R2']:.4f}",
                    extra=extra_ctx,
                )

            except Exception as e:
                logger.error(f"Error training model '{model_key}': {e}", exc_info=True, extra=extra_ctx)
                continue

        if not comparison_records:
            raise ModelTrainingError(
                message="All regression models failed during cross-validation training.",
                user_message="Model training failed for all regression algorithms.",
            )

        # Automatic model evaluation and ranking using ModelSelector
        selector = ModelSelector(primary_metric="RMSE", secondary_metric="MAE", tertiary_metric="R")
        best_model_result = selector.select_best_model(training_results, job_id=job_id)

        report = {
            "target": target_column,
            "feature_names": X.columns.tolist(),
            "models": training_results,
            "comparison_df": best_model_result.comparison_table,
            "best_model": best_model_result,
            "encoder": encoder,
            "transformer": transformer,
            "y_true": y.values,
        }

        return report

    def _extract_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importances or linear model coefficients."""
        records = []
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feat, imp in zip(feature_names, importances):
                records.append({"feature": feat, "importance": float(round(imp, 4))})
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_)
            for feat, imp in zip(feature_names, coefs):
                records.append({"feature": feat, "importance": float(round(imp, 4))})
        else:
            # Equal dummy importances if unprovided
            for feat in feature_names:
                records.append({"feature": feat, "importance": 0.0})

        df_imp = pd.DataFrame(records).sort_values(by="importance", ascending=False)
        return df_imp

    def _format_model_name(self, model_key: str) -> str:
        """Convert model key to human-readable title."""
        names = {
            "linear_regression": "Linear Regression",
            "decision_tree": "Decision Tree Regressor",
            "random_forest": "Random Forest Regressor",
            "bagging": "Bagging Regressor",
            "svr": "Support Vector Regressor (SVR)",
            "xgboost": "XGBoost Regressor",
            "voting": "Voting Regressor (Ensemble)",
        }
        return names.get(model_key.lower(), model_key.title())
