from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

TARGET_COLUMN = "load_gw"

RESULTS_DIR = Path("outputs/results/part1")
FIGURES_DIR = Path("outputs/figures/part1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_weekly_series() -> pd.Series:
    df = pd.read_csv(
        "data/processed/weekly_load_temperature_features.csv",
        parse_dates=["date"],
    )
    df = df.set_index("date").sort_index()
    return df[TARGET_COLUMN]


def adf_test(series: pd.Series) -> pd.DataFrame:
    stat, p_value, used_lags, nobs, critical_values, icbest = adfuller(series.dropna())
    out = pd.DataFrame(
        {
            "statistic": [stat],
            "p_value": [p_value],
            "used_lags": [used_lags],
            "nobs": [nobs],
            "icbest": [icbest],
        }
    )
    for level, value in critical_values.items():
        out[f"critical_value_{level}"] = value
    return out


def kpss_test(series: pd.Series) -> pd.DataFrame:
    stat, p_value, used_lags, critical_values = kpss(
        series.dropna(), regression="c", nlags="auto"
    )
    out = pd.DataFrame(
        {
            "statistic": [stat],
            "p_value": [p_value],
            "used_lags": [used_lags],
        }
    )
    for level, value in critical_values.items():
        out[f"critical_value_{level}"] = value
    return out


def save_acf_pacf(series: pd.Series, suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(series.dropna(), lags=60, ax=ax)
    ax.set_title(f"ACF of weekly load ({suffix})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"weekly_load_acf_{suffix}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_pacf(series.dropna(), lags=60, ax=ax)
    ax.set_title(f"PACF of weekly load ({suffix})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"weekly_load_pacf_{suffix}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    series = load_weekly_series()

    adf_raw = adf_test(series)
    kpss_raw = kpss_test(series)
    adf_raw.to_csv(RESULTS_DIR / "weekly_load_adf_raw.csv", index=False)
    kpss_raw.to_csv(RESULTS_DIR / "weekly_load_kpss_raw.csv", index=False)
    save_acf_pacf(series, "raw")

    diff_1 = series.diff().dropna()
    adf_diff = adf_test(diff_1)
    kpss_diff = kpss_test(diff_1)
    adf_diff.to_csv(RESULTS_DIR / "weekly_load_adf_diff.csv", index=False)
    kpss_diff.to_csv(RESULTS_DIR / "weekly_load_kpss_diff.csv", index=False)
    save_acf_pacf(diff_1, "diff")

    print("Saved Part 1 stationarity results to outputs/results/part1/")
    print("Saved Part 1 figures to outputs/figures/part1/")


if __name__ == "__main__":
    main()