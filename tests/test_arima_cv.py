from src.data.ingestion import DataIngestion
from src.models.arima_model import ARIMAForecaster
from src.evaluation.cross_validation import TimeSeriesCrossValidator


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    series = df.set_index("Date")["Weekly_Sales"].asfreq("W-FRI")

    cv = TimeSeriesCrossValidator(
        initial_train_size=80,
        test_size=12,
        step_size=12
    )

    results_df = cv.evaluate_model(
        series=series,
        model_builder=lambda: ARIMAForecaster(order=(1, 1, 1))
    )

    summary = cv.summarize_results(results_df)

    print("\n✅ ARIMA Cross-Validation Complete!\n")
    print("🔹 Fold Results:")
    print(results_df)

    print("\n🔹 Average Metrics:")
    print(summary)


if __name__ == "__main__":
    main()