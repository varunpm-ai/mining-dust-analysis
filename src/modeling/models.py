"""
Regression Model Registry and Factory Module for Mine Dust Engine.

Provides model instantiation for:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Bagging Regressor
- SVR (Support Vector Regressor)
- XGBoost Regressor
- Voting Regressor (Ensemble)
"""

from typing import Dict, Any, Optional, List
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from src.utils.exceptions import ModelTrainingError
from src.utils.logger import setup_logger

logger = setup_logger("modeling.models")


class ModelFactory:
    """
    Factory for instantiating regression models with standardized random state and hyperparameters.
    """

    @staticmethod
    def get_supported_models() -> List[str]:
        """Return list of supported model keys."""
        return [
            "linear_regression",
            "decision_tree",
            "random_forest",
            "bagging",
            "svr",
            "xgboost",
            "voting",
        ]

    @staticmethod
    def create_model(model_key: str, random_state: int = 42, **kwargs) -> Any:
        """
        Instantiate regression model by key name.

        Args:
            model_key (str): Model algorithm identifier.
            random_state (int): Seed for reproducibility.
            **kwargs: Hyperparameter overrides.

        Returns:
            Any: Scikit-learn or XGBoost regressor instance.
        """
        key = model_key.lower().strip()

        if key == "linear_regression":
            return LinearRegression(**kwargs)

        elif key == "decision_tree":
            params = {"random_state": random_state}
            params.update(kwargs)
            return DecisionTreeRegressor(**params)

        elif key == "random_forest":
            params = {"n_estimators": 100, "random_state": random_state, "n_jobs": -1}
            params.update(kwargs)
            return RandomForestRegressor(**params)

        elif key == "bagging":
            params = {"n_estimators": 50, "random_state": random_state, "n_jobs": -1}
            params.update(kwargs)
            return BaggingRegressor(**params)

        elif key == "svr":
            params = {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"}
            params.update(kwargs)
            return SVR(**params)

        elif key == "xgboost":
            params = {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": random_state,
                "n_jobs": -1,
            }
            params.update(kwargs)
            return XGBRegressor(**params)

        elif key == "voting":
            # Voting Ensemble combining Linear Regression, Random Forest, and XGBoost
            lr = LinearRegression()
            rf = RandomForestRegressor(n_estimators=50, random_state=random_state, n_jobs=-1)
            xgb = XGBRegressor(n_estimators=50, learning_rate=0.05, random_state=random_state, n_jobs=-1)
            
            estimators = [
                ("linear", lr),
                ("rf", rf),
                ("xgb", xgb),
            ]
            return VotingRegressor(estimators=estimators, n_jobs=-1)

        else:
            raise ModelTrainingError(
                message=f"Unsupported model algorithm key: {model_key}",
                user_message=f"Model type '{model_key}' is not supported by the platform.",
            )
