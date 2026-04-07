from src.data.ingestion import DataIngestion
from src.models.sarima_model import SARIMAForecaster
from src.evaluation.metrics import ForecastEvaluator
from src.utils.visualization import ForecastVisualizer


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    sarima = SARIMAForecaster(
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 52)
    )

    results = sarima.run(df)

    metrics = ForecastEvaluator.evaluate(
        results["test"],
        results["predictions"]
    )

    print("\n✅ SARIMA Pipeline Complete!\n")

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
        title="SARIMA Forecast vs Actual"
    )


if __name__ == "__main__":
    main()