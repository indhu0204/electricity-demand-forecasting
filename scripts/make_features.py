import pandas as pd

from electricity_demand.config import DATE_COLUMN, PROCESSED_DATA_DIR
from electricity_demand.data import load_raw_load_data, save_processed_data
from electricity_demand.features import (
    build_hourly_feature_table,
    build_weekly_feature_table,
)


def main() -> None:
    raw_df = load_raw_load_data()

    temp_path = PROCESSED_DATA_DIR / "weekly_temperature_features.csv"
    if temp_path.exists():
        temp_df = pd.read_csv(temp_path)
        temp_df[DATE_COLUMN] = pd.to_datetime(temp_df[DATE_COLUMN], utc=True)
    else:
        temp_df = None

    weekly_df = build_weekly_feature_table(raw_load_df=raw_df, temp_df=temp_df)
    save_processed_data(weekly_df)
    print("Saved processed dataset to data/processed/weekly_load_temperature_features.csv")

    hourly_df = build_hourly_feature_table(
        raw_load_df=raw_df,
        temp_df=None,
    )
    hourly_path = PROCESSED_DATA_DIR / "hourly_load_temperature_features.csv"
    hourly_df.to_csv(hourly_path, index=False)
    print("Saved processed dataset to data/processed/hourly_load_temperature_features.csv")


if __name__ == "__main__":
    main()