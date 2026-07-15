import pandas as pd
import requests

from electricity_demand.config import (
    BERLIN_LATITUDE,
    BERLIN_LONGITUDE,
    DATE_COLUMN,
    PROCESSED_DATA_DIR,
)


def main() -> None:
    start_date = "2015-01-01"
    end_date = "2020-10-04"

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={BERLIN_LATITUDE}"
        f"&longitude={BERLIN_LONGITUDE}"
        f"&start_date={start_date}"
        f"&end_date={end_date}"
        "&hourly=temperature_2m"
        "&timezone=UTC"
    )

    print(f"Requesting temperature data from: {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    times = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]

    temp_df = pd.DataFrame({"timestamp": pd.to_datetime(times), "temp_c": temps})
    temp_df["date"] = temp_df["timestamp"].dt.floor("D")

    weekly_temp = (
        temp_df.set_index("timestamp")
        .resample("W-SUN")
        .agg({"temp_c": ["mean", "min", "max"]})
    )
    weekly_temp.columns = ["temp_mean", "temp_min", "temp_max"]
    weekly_temp = weekly_temp.reset_index().rename(columns={"timestamp": DATE_COLUMN})

    base_for_hdd = 18.0
    base_for_cdd = 22.0
    weekly_temp["heating_degree_days"] = (
        (base_for_hdd - weekly_temp["temp_mean"]).clip(lower=0)
    )
    weekly_temp["cooling_degree_days"] = (
        (weekly_temp["temp_mean"] - base_for_cdd).clip(lower=0)
    )

    out_path = PROCESSED_DATA_DIR / "weekly_temperature_features.csv"
    weekly_temp.to_csv(out_path, index=False)
    print(f"Saved weekly temperature features to {out_path}")


if __name__ == "__main__":
    main()