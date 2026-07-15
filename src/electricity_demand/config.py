from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODEL_OBJECTS_DIR = OUTPUTS_DIR / "model_objects"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORT_FIGURES_DIR = REPORTS_DIR / "figures"

OPSD_URL = "https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv"

RAW_LOAD_FILENAME = "time_series_60min_singleindex.csv"
PROCESSED_WEEKLY_FILENAME = "weekly_load_temperature_features.csv"

TARGET_COLUMN = "load_gw"
DATE_COLUMN = "date"

START_DATE = "2015-01-01"
TEST_HORIZON_WEEKS = 104
SEASONAL_PERIOD_WEEKS = 52
RANDOM_SEED = 42

BERLIN_LATITUDE = 52.52
BERLIN_LONGITUDE = 13.405

for path in [
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    FORECASTS_DIR,
    METRICS_DIR,
    MODEL_OBJECTS_DIR,
    REPORT_FIGURES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)