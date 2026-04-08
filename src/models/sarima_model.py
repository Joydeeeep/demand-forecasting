import pandas as pd
from typing import Tuple, Dict, Any
from src.evaluation.metrics import ForecastEvaluator
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.config.config import DATE_COLUMN, TARGET_COLUMN, TEST_SIZE


class SARIMAForecaster:
    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 52)
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None

    def prepare_series(self, df: pd.DataFrame) -> pd.Series:
        """Extract target series with datetime index"""
        series = df[[DATE_COLUMN, TARGET_COLUMN]].copy()
        series = series.set_index(DATE_COLUMN)[TARGET_COLUMN]
        series = series.asfreq("W-FRI")
        return series

    def train_test_split(self, series: pd.Series):
        """Chronological train-test split"""
        train = series[:-TEST_SIZE]
        test = series[-TEST_SIZE:]
        return train, test

    def check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Run Augmented Dickey-Fuller test"""
        result = adfuller(series.dropna())

        return {
            "adf_statistic": result[0],
            "p_value": result[1],
            "used_lag": result[2],
            "n_obs": result[3],
            "critical_values": result[4],
            "is_stationary": result[1] < 0.05
        }

    def fit(self, train: pd.Series):
        """Fit SARIMA model"""
        self.model = SARIMAX(
            train,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False)

    def forecast(self, steps: int) -> pd.Series:
        """Forecast future values"""
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted yet.")

        forecast = self.fitted_model.forecast(steps=steps)
        return forecast

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Full SARIMA workflow"""
        series = self.prepare_series(df)

        stationarity = self.check_stationarity(series)

        train, test = self.train_test_split(series)

        self.fit(train)

        predictions = self.forecast(len(test))

        metrics = ForecastEvaluator.evaluate(test.values, predictions.values)

        return {
            "train": train,
            "test": test,
            "predictions": predictions,
            "metrics": metrics,
            "stationarity": stationarity,
            "model_summary": self.fitted_model.summary().as_text()
        }