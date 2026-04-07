import pandas as pd
from typing import Callable, Dict, List

from src.evaluation.metrics import ForecastEvaluator


class TimeSeriesCrossValidator:
    def __init__(self, initial_train_size: int, test_size: int, step_size: int):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, series: pd.Series):
        """
        Generate expanding window train-test splits
        """
        n = len(series)
        splits = []

        train_end = self.initial_train_size

        while train_end + self.test_size <= n:
            train = series[:train_end]
            test = series[train_end:train_end + self.test_size]

            splits.append((train, test))
            train_end += self.step_size

        return splits

    def evaluate_model(
        self,
        series: pd.Series,
        model_builder: Callable
    ) -> pd.DataFrame:
        """
        Evaluate model across all folds
        model_builder must return a fresh model instance each time
        """
        splits = self.split(series)
        results = []

        for fold, (train, test) in enumerate(splits, start=1):
            model = model_builder()
            model.fit(train)
            predictions = model.forecast(len(test))

            metrics = ForecastEvaluator.evaluate(test, predictions)
            metrics["fold"] = fold

            results.append(metrics)

        return pd.DataFrame(results)

    @staticmethod
    def summarize_results(results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Return average metrics across folds
        """
        return {
            "MAPE_mean": round(results_df["MAPE"].mean(), 4),
            "MAE_mean": round(results_df["MAE"].mean(), 4),
            "RMSE_mean": round(results_df["RMSE"].mean(), 4),
        }