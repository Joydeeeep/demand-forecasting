import pandas as pd

from src.data.ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer

from src.models.arima_model import ARIMAForecaster
from src.models.sarima_model import SARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.xgboost_model import XGBoostForecaster
from src.utils.artifact_manager import ArtifactManager


class ModelComparison:
    def __init__(self):
        self.ingestor = DataIngestion()
        self.engineer = FeatureEngineer()
        self.artifact_manager = ArtifactManager()
        self._comparison_cache = None

    def load_data(self):
        raw_df = self.ingestor.run()
        feature_df = self.engineer.run(raw_df).reset_index(drop=True)
        return raw_df, feature_df

    def run_models(self):
        raw_df, feature_df = self.load_data()

        results = {}

        arima = ARIMAForecaster()
        results["ARIMA"] = arima.run(raw_df)["metrics"]

        sarima = SARIMAForecaster()
        results["SARIMA"] = sarima.run(raw_df)["metrics"]

        prophet = ProphetForecaster()
        results["Prophet"] = prophet.run(raw_df)["metrics"]

        xgb = XGBoostForecaster()
        results["XGBoost"] = xgb.run(feature_df)["metrics"]

        return results

    def get_comparison(self):
        if self._comparison_cache is None:
            results = self.run_models()
            df = pd.DataFrame(results).T
            df = df.sort_values(by="MAPE")
            self._comparison_cache = df

        return self._comparison_cache

    def save_comparison_results(self):
        holdout_df = self.get_comparison()
        self.artifact_manager.save_dataframe(holdout_df, "holdout_model_comparison.csv")