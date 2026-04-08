from src.data.ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer

from src.models.arima_model import ARIMAForecaster
from src.models.sarima_model import SARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster

from src.utils.artifact_manager import ArtifactManager


def main():
    ingestor = DataIngestion()
    raw_df = ingestor.run()

    engineer = FeatureEngineer()
    feature_df = engineer.run(raw_df).reset_index(drop=True)

    artifact_manager = ArtifactManager()

    models = {
        "arima": ARIMAForecaster(),
        "sarima": SARIMAForecaster(),
        "prophet": ProphetForecaster(),
        "xgboost": XGBoostForecaster()
    }

    model_inputs = {
        "arima": raw_df,
        "sarima": raw_df,
        "prophet": raw_df,
        "xgboost": feature_df
    }

    for model_name, model in models.items():
        print(f"\n🔄 Running {model_name.upper()}...")

        results = model.run(model_inputs[model_name])

        # Save trained model object
        if model_name == "xgboost":
            model_obj = model.model
        elif model_name == "prophet":
            model_obj = model.model
        else:
            model_obj = model.fitted_model

        model_path = artifact_manager.save_model(model_obj, model_name)
        metrics_path = artifact_manager.save_metrics(results["metrics"], model_name)
        preds_path = artifact_manager.save_predictions(results["predictions"], model_name)

        print(f"✅ Saved {model_name.upper()} artifacts")
        print("   Model:", model_path)
        print("   Metrics:", metrics_path)
        print("   Predictions:", preds_path)

    print("\n🎉 All model artifacts saved successfully!")


if __name__ == "__main__":
    main()