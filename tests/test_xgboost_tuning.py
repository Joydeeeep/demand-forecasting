from src.data.ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer
from src.models.xgboost_tuner import XGBoostTuner


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    engineer = FeatureEngineer()
    feature_df = engineer.run(df).reset_index(drop=True)

    tuner = XGBoostTuner()

    results = tuner.tune(feature_df, max_evals=20)

    print("\n🏆 Tuning Complete!\n")
    print("Best MAPE:", results["best_mape"])
    print("Best Params:", results["best_params"])


if __name__ == "__main__":
    main()