from src.data.ingestion import DataIngestion

def main():
    ingestor = DataIngestion()
    df = ingestor.run()

    print("\n✅ Data Loaded Successfully!\n")

    print("🔹 Shape:", df.shape)
    print("\n🔹 Columns:", df.columns.tolist())

    print("\n🔹 First 5 rows:")
    print(df.head())

    print("\n🔹 Date Range:")
    print(df["Date"].min(), "→", df["Date"].max())

    print("\n🔹 Any missing values?")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()