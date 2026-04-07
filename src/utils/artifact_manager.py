import os
import json
import pickle
import pandas as pd


class ArtifactManager:
    def __init__(self, base_dir="artifacts"):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, "models")
        self.metrics_dir = os.path.join(base_dir, "metrics")
        self.predictions_dir = os.path.join(base_dir, "predictions")
        self.results_dir = os.path.join(base_dir, "results")

        self._create_directories()

    def _create_directories(self):
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def save_model(self, model, model_name: str):
        path = os.path.join(self.models_dir, f"{model_name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path

    def load_model(self, model_name: str):
        path = os.path.join(self.models_dir, f"{model_name}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_metrics(self, metrics: dict, model_name: str):
        path = os.path.join(self.metrics_dir, f"{model_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)
        return path

    def save_predictions(self, predictions: pd.Series, model_name: str):
        path = os.path.join(self.predictions_dir, f"{model_name}_predictions.csv")
        predictions.to_csv(path, header=True)
        return path

    def save_dataframe(self, df: pd.DataFrame, filename: str):
        path = os.path.join(self.results_dir, filename)
        df.to_csv(path, index=True)
        return path