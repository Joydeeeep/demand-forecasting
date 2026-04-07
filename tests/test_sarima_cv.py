from src.data.ingestion import DataIngestion
from src.models.sarima_model import SARIMAForecaster
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
        model_builder=lambda: SARIMAForecaster(
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 52)
        )
    )

    summary = cv.summarize_results(results_df)

    print("\n✅ SARIMA Cross-Validation Complete!\n")
    print("🔹 Fold Results:")
    print(results_df)

    print("\n🔹 Average Metrics:")
    print(summary)


if __name__ == "__main__":
    main()