import numpy as np
import pandas as pd
from typing import Dict


class ForecastEvaluator:
    @staticmethod
    def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE)"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # Avoid division by zero
        mask = y_true != 0

        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Mean Absolute Error (MAE)"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs(y_true - y_pred))

    @staticmethod
    def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
        """Calculate Root Mean Squared Error (RMSE)"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    @classmethod
    def evaluate(cls, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Return all evaluation metrics"""
        return {
            "MAPE": round(cls.mean_absolute_percentage_error(y_true, y_pred), 4),
            "MAE": round(cls.mean_absolute_error(y_true, y_pred), 4),
            "RMSE": round(cls.root_mean_squared_error(y_true, y_pred), 4),
        }