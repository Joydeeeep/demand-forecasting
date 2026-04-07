from pathlib import Path

# ======================
# PATHS
# ======================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# ======================
# DATA FILE
# ======================

SALES_FILE = RAW_DATA_DIR / "Walmart_Sales.csv"

# ======================
# COLUMNS
# ======================

DATE_COLUMN = "Date"
TARGET_COLUMN = "Weekly_Sales"

# ======================
# FILTERING (V1)
# ======================

DEFAULT_STORE = 1  # integer now

# ======================
# TRAINING
# ======================

FORECAST_HORIZON = 12  # weeks
TEST_SIZE = 12

# ======================
# LOGGING
# ======================

LOG_LEVEL = "INFO"