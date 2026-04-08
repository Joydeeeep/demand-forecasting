from src.data.ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostForecaster
from src.utils.artifact_manager import ArtifactManager


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    engineer = FeatureEngineer()
    feature_df = engineer.run(df).reset_index(drop=True)

    model = XGBoostForecaster()
    results = model.run(feature_df)

    artifact_manager = ArtifactManager()

    model_path = artifact_manager.save_model(model.model, "xgboost_forecaster")
    metrics_path = artifact_manager.save_metrics(results["metrics"], "xgboost_forecaster")
    preds_path = artifact_manager.save_predictions(results["predictions"], "xgboost_forecaster")

    print("\n✅ XGBoost artifacts saved successfully!\n")
    print("Model saved to:", model_path)
    print("Metrics saved to:", metrics_path)
    print("Predictions saved to:", preds_path)


if __name__ == "__main__":
    main()