import numpy as np
import pandas as pd

from electricity_demand.config import DATE_COLUMN, START_DATE, TARGET_COLUMN


def prepare_hourly_load(df: pd.DataFrame) -> pd.DataFrame:
    load_col = "DE_load_actual_entsoe_transparency"

    data = df.copy()
    data["utc_timestamp"] = pd.to_datetime(data["utc_timestamp"], utc=True, errors="coerce")
    data = data.dropna(subset=["utc_timestamp"])
    data = data.rename(columns={"utc_timestamp": "timestamp"})

    if load_col not in data.columns:
        raise ValueError(f"Expected column '{load_col}' not found in raw data.")

    data = data[["timestamp", load_col]].rename(columns={load_col: "load_mw"})
    data = data.dropna(subset=["load_mw"])
    data = data[data["timestamp"] >= pd.Timestamp(START_DATE, tz="UTC")]
    data = data.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    data["load_gw"] = data["load_mw"] / 1000.0
    return data


def aggregate_to_weekly(hourly_df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        hourly_df.set_index("timestamp")[["load_gw"]]
        .resample("W-SUN")
        .mean()
        .rename(columns={"load_gw": TARGET_COLUMN})
        .reset_index()
        .rename(columns={"timestamp": DATE_COLUMN})
    )
    return weekly


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], utc=True)

    iso = data[DATE_COLUMN].dt.isocalendar()
    data["year"] = data[DATE_COLUMN].dt.year
    data["month"] = data[DATE_COLUMN].dt.month
    data["quarter"] = data[DATE_COLUMN].dt.quarter
    data["weekofyear"] = iso.week.astype(int)

    data["sin_week"] = np.sin(2 * np.pi * data["weekofyear"] / 52)
    data["cos_week"] = np.cos(2 * np.pi * data["weekofyear"] / 52)

    return data


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    dates = pd.to_datetime(data[DATE_COLUMN], utc=True)

    christmas_weeks = dates.dt.month.eq(12) & dates.dt.day.ge(20)
    newyear_weeks = dates.dt.month.eq(1) & dates.dt.day.le(7)

    data["holiday_days"] = 0
    data.loc[christmas_weeks, "holiday_days"] = 2
    data.loc[newyear_weeks, "holiday_days"] = data.loc[newyear_weeks, "holiday_days"] + 1
    data["has_holiday"] = (data["holiday_days"] > 0).astype(int)

    return data


def add_temperature_features(
    df: pd.DataFrame,
    temp_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    data = df.copy()

    if temp_df is None:
        data["temp_mean"] = np.nan
        data["temp_min"] = np.nan
        data["temp_max"] = np.nan
        data["heating_degree_days"] = np.nan
        data["cooling_degree_days"] = np.nan
        return data

    temp = temp_df.copy()

    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], utc=True)
    temp[DATE_COLUMN] = pd.to_datetime(temp[DATE_COLUMN], utc=True)

    data = data.merge(temp, on=DATE_COLUMN, how="left")
    return data


def add_lag_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["lag_1"] = data[TARGET_COLUMN].shift(1)
    data["lag_2"] = data[TARGET_COLUMN].shift(2)
    data["lag_4"] = data[TARGET_COLUMN].shift(4)
    data["lag_8"] = data[TARGET_COLUMN].shift(8)
    data["lag_52"] = data[TARGET_COLUMN].shift(52)

    data["rolling_mean_4"] = data[TARGET_COLUMN].shift(1).rolling(4).mean()
    data["rolling_mean_8"] = data[TARGET_COLUMN].shift(1).rolling(8).mean()
    data["rolling_std_4"] = data[TARGET_COLUMN].shift(1).rolling(4).std()
    data["rolling_std_8"] = data[TARGET_COLUMN].shift(1).rolling(8).std()

    return data


def build_weekly_feature_table(
    raw_load_df: pd.DataFrame,
    temp_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    hourly = prepare_hourly_load(raw_load_df)
    weekly = aggregate_to_weekly(hourly)
    weekly = add_calendar_features(weekly)
    weekly = add_holiday_features(weekly)
    weekly = add_temperature_features(weekly, temp_df=temp_df)
    weekly = add_lag_rolling_features(weekly)

    weekly = weekly.sort_values(DATE_COLUMN).reset_index(drop=True)
    return weekly

def build_hourly_feature_table(
    raw_load_df: pd.DataFrame,
    temp_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    hourly = prepare_hourly_load(raw_load_df).copy()
    hourly = hourly.sort_values("timestamp").reset_index(drop=True)

    hourly["date"] = hourly["timestamp"]

    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["day_of_week"] = hourly["timestamp"].dt.dayofweek
    hourly["month"] = hourly["timestamp"].dt.month
    hourly["is_weekend"] = (hourly["day_of_week"] >= 5).astype(int)

    hourly["lag_1"] = hourly["load_gw"].shift(1)
    hourly["lag_24"] = hourly["load_gw"].shift(24)
    hourly["lag_168"] = hourly["load_gw"].shift(168)

    hourly["rolling_mean_24"] = hourly["load_gw"].shift(1).rolling(24).mean()
    hourly["rolling_mean_168"] = hourly["load_gw"].shift(1).rolling(168).mean()

    if temp_df is not None:
        temp = temp_df.copy()
        temp[DATE_COLUMN] = pd.to_datetime(temp[DATE_COLUMN], utc=True)
        hourly = hourly.merge(temp, left_on="date", right_on=DATE_COLUMN, how="left")

    hourly = hourly.dropna().reset_index(drop=True)
    return hourly