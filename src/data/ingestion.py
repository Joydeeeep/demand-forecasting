import pandas as pd

from src.config.config import (
    SALES_FILE,
    DEFAULT_STORE,
    DATE_COLUMN
)


class DataIngestion:
    def __init__(self):
        self.sales_path = SALES_FILE

    def load_data(self) -> pd.DataFrame:
        """Load dataset"""
        df = pd.read_csv(self.sales_path)
        return df

    def validate_data(self, df: pd.DataFrame):
        """Basic validation"""
        required_columns = [
            "Store",
            "Date",
            "Weekly_Sales"
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing"""

        # Convert date
        df[DATE_COLUMN] = pd.to_datetime(
           df[DATE_COLUMN],
           dayfirst=True
         )

        # Sort by time
        df = df.sort_values(by=DATE_COLUMN)

        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter single store (V1 simplification)"""
        df = df[df["Store"] == DEFAULT_STORE]
        return df

    def run(self) -> pd.DataFrame:
        """Full pipeline"""
        df = self.load_data()

        self.validate_data(df)

        df = self.preprocess(df)

        df = self.filter_data(df)

        return df.reset_index(drop=True)