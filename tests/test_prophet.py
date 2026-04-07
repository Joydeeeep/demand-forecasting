from src.data.ingestion import DataIngestion
from src.models.prophet_model import ProphetForecaster
from src.evaluation.metrics import ForecastEvaluator
from src.utils.visualization import ForecastVisualizer


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    prophet = ProphetForecaster()
    results = prophet.run(df)

    metrics = ForecastEvaluator.evaluate(
        results["test"],
        results["predictions"]
    )

    print("\n✅ Prophet Pipeline Complete!\n")

    print("🔹 Evaluation Metrics:")
    print(metrics)

    print("\n🔹 Predictions:")
    print(results["predictions"])

    ForecastVisualizer.plot_forecast(
        train=results["train"],
        test=results["test"],
        predictions=results["predictions"],
        title="Prophet Forecast vs Actual"
    )


if __name__ == "__main__":
    main()