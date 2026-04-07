from src.data.ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer


def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    fe = FeatureEngineer()
    df_features = fe.run(df)

    print("\n✅ Feature Engineering Complete!\n")

    print("🔹 Shape:", df_features.shape)
    print("\n🔹 Columns:")
    print(df_features.columns.tolist())

    print("\n🔹 Sample:")
    print(df_features.head())


if __name__ == "__main__":
    main()