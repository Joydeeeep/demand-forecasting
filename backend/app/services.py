import json
#import pickle
import pandas as pd

from backend.app.config import METRICS_DIR, PREDICTIONS_DIR, RESULTS_DIR

class ArtifactService:
    @staticmethod
    def list_models():
        return ["arima", "sarima", "prophet", "xgboost"]

    @staticmethod
    def load_metrics(model_name: str):
        path = METRICS_DIR / f"{model_name}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Metrics for '{model_name}' not found")

        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def load_predictions(model_name: str):
        path = PREDICTIONS_DIR / f"{model_name}_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Predictions for '{model_name}' not found")

        df = pd.read_csv(path, index_col=0)
        return df.squeeze()

    @staticmethod
    def load_comparison():
        path = RESULTS_DIR / "holdout_model_comparison.csv"
        if not path.exists():
            raise FileNotFoundError("Comparison results not found")

        return pd.read_csv(path, index_col=0)
   

    @staticmethod
    def load_actual_data():
        path = "data/raw/Walmart_Sales.csv"
        df = pd.read_csv(path)

        # Convert properly
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

        # Same store
        df = df[df["Store"] == 1].copy()

        # Sort
        df = df.sort_values("Date")

        # VERY IMPORTANT: filter ONLY prediction period
        df = df[df["Date"] >= "2012-08-01"]

        return df[["Date", "Weekly_Sales"]]