from src.data.ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostForecaster
from src.evaluation.metrics import ForecastEvaluator
from src.utils.visualization import ForecastVisualizer


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    engineer = FeatureEngineer()
    feature_df = engineer.run(df)

    xgb = XGBoostForecaster()
    results = xgb.run(feature_df)

    metrics = ForecastEvaluator.evaluate(
        results["test"],
        results["predictions"]
    )

    print("\n✅ XGBoost Pipeline Complete!\n")

    print("🔹 Evaluation Metrics:")
    print(metrics)

    print("\n🔹 Feature Columns Used:")
    print(results["feature_columns"])

    print("\n🔹 Predictions:")
    print(results["predictions"])

    ForecastVisualizer.plot_forecast(
        train=results["train"],
        test=results["test"],
        predictions=results["predictions"],
        title="XGBoost Forecast vs Actual"
    )


if __name__ == "__main__":
    main()