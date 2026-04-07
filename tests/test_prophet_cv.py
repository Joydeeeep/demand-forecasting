import pandas as pd

from src.data.ingestion import DataIngestion
from src.models.prophet_model import ProphetForecaster
from src.evaluation.cross_validation import TimeSeriesCrossValidator
from src.evaluation.metrics import ForecastEvaluator


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    prophet_df = df[["Date", "Weekly_Sales"]].copy()
    prophet_df.columns = ["ds", "y"]

    series = prophet_df.set_index("ds")["y"].asfreq("W-FRI")

    cv = TimeSeriesCrossValidator(
        initial_train_size=80,
        test_size=12,
        step_size=12
    )

    splits = cv.split(series)
    results = []

    for fold, (train_series, test_series) in enumerate(splits, start=1):
        train_df = train_series.reset_index()
        train_df.columns = ["ds", "y"]

        model = ProphetForecaster()
        model.fit(train_df)

        forecast = model.forecast(periods=len(test_series))
        predictions = forecast.tail(len(test_series))["yhat"]
        predictions.index = test_series.index

        metrics = ForecastEvaluator.evaluate(test_series, predictions)
        metrics["fold"] = fold
        results.append(metrics)

    results_df = pd.DataFrame(results)
    summary = cv.summarize_results(results_df)

    print("\n✅ Prophet Cross-Validation Complete!\n")
    print("🔹 Fold Results:")
    print(results_df)

    print("\n🔹 Average Metrics:")
    print(summary)


if __name__ == "__main__":
    main()