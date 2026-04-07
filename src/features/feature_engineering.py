import pandas as pd
import numpy as np
from typing import List

from src.config.config import DATE_COLUMN, TARGET_COLUMN


class FeatureEngineer:
    def __init__(self, lags: List[int] = [1, 2, 4, 8], rolling_windows: List[int] = [4, 8]):
        self.lags = lags
        self.rolling_windows = rolling_windows

    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features (past values)"""
        for lag in self.lags:
            df[f"lag_{lag}"] = df[TARGET_COLUMN].shift(lag)
        return df

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling mean and std"""
        for window in self.rolling_windows:
            df[f"rolling_mean_{window}"] = df[TARGET_COLUMN].shift(1).rolling(window).mean()
            df[f"rolling_std_{window}"] = df[TARGET_COLUMN].shift(1).rolling(window).std()
        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features"""
        df["year"] = df[DATE_COLUMN].dt.year
        df["month"] = df[DATE_COLUMN].dt.month
        df["weekofyear"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)

        return df

    def drop_na(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows with NaNs created by lagging"""
        return df.dropna().reset_index(drop=True)
    
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full feature engineering pipeline"""
        df = df.copy()

        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_time_features(df)

        df = self.drop_na(df)
        
        return df