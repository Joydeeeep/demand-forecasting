import matplotlib.pyplot as plt
import pandas as pd


class ForecastVisualizer:
    @staticmethod
    def plot_forecast(
        train: pd.Series,
        test: pd.Series,
        predictions: pd.Series,
        title: str = "Forecast vs Actual"
    ):
        """Plot train, test, and forecasted values"""

        plt.figure(figsize=(14, 6))

        plt.plot(train.index, train.values, label="Train")
        plt.plot(test.index, test.values, label="Test")
        plt.plot(predictions.index, predictions.values, label="Forecast")

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()