import itertools
import pandas as pd

from xgboost import XGBRegressor

from src.evaluation.cross_validation import TimeSeriesCrossValidator
from src.evaluation.metrics import ForecastEvaluator


class XGBoostTuner:
    def __init__(self):
        self.param_grid = {
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.05, 0.1],
            "n_estimators": [200, 300, 500],
            "subsample": [0.7, 0.8],
            "colsample_bytree": [0.7, 0.8],
            "min_child_weight": [1, 3]
        }

    def generate_param_combinations(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def evaluate_params(self, df: pd.DataFrame, params: dict):
        from src.models.xgboost_model import XGBoostForecaster

        cv = TimeSeriesCrossValidator(
            initial_train_size=80,
            test_size=12,
            step_size=12
        )

        series = df.set_index("Date")["Weekly_Sales"].asfreq("W-FRI")
        splits = cv.split(series)

        results = []

        for train_series, test_series in splits:
            train_end = train_series.index[-1]
            test_end = test_series.index[-1]

            fold_df = df[df["Date"] <= test_end].copy()

            train_df = fold_df[fold_df["Date"] <= train_end]
            test_df = fold_df[
                (fold_df["Date"] > train_end) & (fold_df["Date"] <= test_end)
            ]

            model = XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                **params
            )

            X_train = train_df.drop(columns=["Weekly_Sales", "Date"])
            y_train = train_df["Weekly_Sales"]

            X_test = test_df.drop(columns=["Weekly_Sales", "Date"])
            y_test = test_df["Weekly_Sales"]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics = ForecastEvaluator.evaluate(y_test.values, preds)
            results.append(metrics["MAPE"])

        return sum(results) / len(results)

    def tune(self, df: pd.DataFrame, max_evals=20):
        results = []

        for i, params in enumerate(self.generate_param_combinations()):
            if i >= max_evals:
                break

            score = self.evaluate_params(df, params)

            print(f"Test {i+1}: MAPE={score:.4f} | Params={params}")

            results.append((score, params))

        results.sort(key=lambda x: x[0])

        best_score, best_params = results[0]

        return {
            "best_mape": round(best_score, 4),
            "best_params": best_params,
            "all_results": results
        }