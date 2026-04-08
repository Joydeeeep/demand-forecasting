import pandas as pd
from typing import Dict, Any

from xgboost import XGBRegressor
from src.evaluation.metrics import ForecastEvaluator
from src.config.config import DATE_COLUMN, TARGET_COLUMN, TEST_SIZE


class XGBoostForecaster:
    def __init__(self):
        self.model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=1,
        random_state=42,
        objective="reg:squarederror")

    def train_test_split(self, df: pd.DataFrame):
        """Chronological train-test split"""
        train = df[:-TEST_SIZE]
        test = df[-TEST_SIZE:]
        return train, test

    def prepare_features(self, df: pd.DataFrame):
        """Separate features and target"""
        X = df.drop(columns=[TARGET_COLUMN, DATE_COLUMN])
        y = df[TARGET_COLUMN]
        return X, y

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train XGBoost model"""
        self.model.fit(X_train, y_train)

    def predict(self, X_test: pd.DataFrame):
        """Generate predictions"""
        return self.model.predict(X_test)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Full XGBoost workflow"""
        train, test = self.train_test_split(df)

        X_train, y_train = self.prepare_features(train)
        X_test, y_test = self.prepare_features(test)

        self.fit(X_train, y_train)
        predictions = self.predict(X_test)

        predictions_series = pd.Series(predictions, index=test[DATE_COLUMN])

        metrics = ForecastEvaluator.evaluate(y_test.values, predictions)

        return {
            "train": pd.Series(y_train.values, index=train[DATE_COLUMN]),
            "test": pd.Series(y_test.values, index=test[DATE_COLUMN]),
            "predictions": predictions_series,
            "metrics": metrics,
            "feature_columns": list(X_train.columns),
            "model_summary": "XGBoost model trained successfully."
        }