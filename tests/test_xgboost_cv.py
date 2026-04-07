import pandas as pd

from src.data.ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostForecaster
from src.evaluation.cross_validation import TimeSeriesCrossValidator
from src.evaluation.metrics import ForecastEvaluator


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    engineer = FeatureEngineer()
    feature_df = engineer.run(df).reset_index(drop=True)

    series = feature_df.set_index("Date")["Weekly_Sales"].asfreq("W-FRI")

    cv = TimeSeriesCrossValidator(
        initial_train_size=80,
        test_size=12,
        step_size=12
    )

    splits = cv.split(series)
    results = []

    for fold, (train_series, test_series) in enumerate(splits, start=1):
        train_end_date = train_series.index[-1]
        test_end_date = test_series.index[-1]

        fold_df = feature_df[
            (feature_df["Date"] <= test_end_date)
        ].copy()

        train_df = fold_df[fold_df["Date"] <= train_end_date].copy()
        test_df = fold_df[
            (fold_df["Date"] > train_end_date) & (fold_df["Date"] <= test_end_date)
        ].copy()

        model = XGBoostForecaster()

        X_train, y_train = model.prepare_features(train_df)
        X_test, y_test = model.prepare_features(test_df)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        predictions_series = pd.Series(predictions, index=test_df["Date"])

        metrics = ForecastEvaluator.evaluate(y_test.values, predictions_series.values)
        metrics["fold"] = fold
        results.append(metrics)

    results_df = pd.DataFrame(results)
    summary = cv.summarize_results(results_df)

    print("\n✅ XGBoost Cross-Validation Complete!\n")
    print("🔹 Fold Results:")
    print(results_df)

    print("\n🔹 Average Metrics:")
    print(summary)


if __name__ == "__main__":
    main()