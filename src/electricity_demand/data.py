import pandas as pd

from electricity_demand.config import (
    DATE_COLUMN,
    PROCESSED_DATA_DIR,
    PROCESSED_WEEKLY_FILENAME,
    RAW_DATA_DIR,
    RAW_LOAD_FILENAME,
)


def load_raw_load_data(filename: str = RAW_LOAD_FILENAME) -> pd.DataFrame:
    path = RAW_DATA_DIR / filename
    return pd.read_csv(path)


def save_processed_data(
    df: pd.DataFrame,
    filename: str = PROCESSED_WEEKLY_FILENAME,
) -> None:
    path = PROCESSED_DATA_DIR / filename
    df.to_csv(path, index=False)


def load_processed_data(
    filename: str = PROCESSED_WEEKLY_FILENAME,
) -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / filename
    df = pd.read_csv(path, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).set_index(DATE_COLUMN)
    return df