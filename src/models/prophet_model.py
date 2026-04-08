import pandas as pd
from typing import Dict, Any

from prophet import Prophet
from src.evaluation.metrics import ForecastEvaluator
from src.config.config import DATE_COLUMN, TARGET_COLUMN, TEST_SIZE


class ProphetForecaster:
    def __init__(self):
        self.model = None
        self.fitted_model = None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for Prophet"""
        prophet_df = df[[DATE_COLUMN, TARGET_COLUMN]].copy()
        prophet_df.columns = ["ds", "y"]
        return prophet_df

    def train_test_split(self, df: pd.DataFrame):
        """Chronological train-test split"""
        train = df[:-TEST_SIZE]
        test = df[-TEST_SIZE:]
        return train, test

    def fit(self, train: pd.DataFrame):
        """Fit Prophet model"""
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        self.fitted_model = self.model.fit(train)

    def forecast(self, periods: int) -> pd.DataFrame:
        """Forecast future periods"""
        if self.fitted_model is None:
            raise ValueError("Model has not been fitted yet.")

        future = self.model.make_future_dataframe(periods=periods, freq="W-FRI")
        forecast = self.model.predict(future)
        return forecast

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Full Prophet workflow"""
        prophet_df = self.prepare_data(df)

        train, test = self.train_test_split(prophet_df)

        self.fit(train)

        forecast = self.forecast(periods=len(test))

        predictions = forecast.tail(len(test))["yhat"]
        predictions.index = test["ds"]

        metrics = ForecastEvaluator.evaluate(test["y"].values, predictions.values)

        return {
            "train": train.set_index("ds")["y"],
            "test": test.set_index("ds")["y"],
            "predictions": predictions,
            "metrics": metrics,
            "forecast_full": forecast,
            "model_summary": "Prophet model fitted successfully."
        }