from src.data.ingestion import DataIngestion
from src.models.arima_model import ARIMAForecaster
from src.evaluation.metrics import ForecastEvaluator
from src.utils.visualization import ForecastVisualizer


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    arima = ARIMAForecaster(order=(1, 1, 1))
    results = arima.run(df)

    metrics = ForecastEvaluator.evaluate(
        results["test"],
        results["predictions"]
    )

    print("\n✅ ARIMA Pipeline Complete!\n")

    print("🔹 Stationarity Check:")
    print(results["stationarity"])

    print("\n🔹 Evaluation Metrics:")
    print(metrics)

    print("\n🔹 Predictions:")
    print(results["predictions"])

    print("\n🔹 Model Summary:")
    print(results["model_summary"])

    ForecastVisualizer.plot_forecast(
        train=results["train"],
        test=results["test"],
        predictions=results["predictions"],
        title="ARIMA Forecast vs Actual"
    )


if __name__ == "__main__":
    main()