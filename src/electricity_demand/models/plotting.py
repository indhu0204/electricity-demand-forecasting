from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


FIGURES_DIR = Path("outputs/figures")
PART1_FIGURES_DIR = FIGURES_DIR / "part1"
PART2_FIGURES_DIR = FIGURES_DIR / "part2"
PART3_FIGURES_DIR = FIGURES_DIR / "part3"
PART3_RESULTS_DIR = Path("outputs/results/part3")

PART1_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PART2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PART3_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PART3_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET_COLUMN = "load_gw"
TEST_HORIZON_WEEKS = 104


def load_weekly_data() -> pd.DataFrame:
    df = pd.read_csv(
        "data/processed/weekly_load_temperature_features.csv",
        parse_dates=["date"],
    )
    df = df.set_index("date").sort_index()
    return df


def load_forecasts() -> pd.DataFrame:
    df = pd.read_csv(
        "outputs/forecasts/part2/all_forecasts.csv",
        parse_dates=["date"],
    )
    df = df.set_index("date").sort_index()
    return df


def plot_weekly_load(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df[TARGET_COLUMN], color="tab:blue")
    ax.set_title("Weekly German electricity load (2015–2020)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (GW)")
    fig.tight_layout()
    fig.savefig(PART1_FIGURES_DIR / "weekly_load_timeseries.png", dpi=150)
    plt.close(fig)


def plot_load_temperature_eda(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))

    ax1.plot(df.index, df[TARGET_COLUMN], color="tab:blue", label="Weekly load")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Load (GW)", color="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(df.index, df["temp_mean"], color="tab:red", alpha=0.6, label="Temp mean")
    ax2.set_ylabel("Temperature (°C)", color="tab:red")

    fig.suptitle("Weekly load and Berlin mean temperature")
    fig.tight_layout()
    fig.savefig(PART1_FIGURES_DIR / "weekly_load_temperature_eda.png", dpi=150)
    plt.close(fig)


def plot_benchmark_forecasts(forecasts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecasts.index, forecasts["actual"], label="Actual", color="black")
    ax.plot(forecasts.index, forecasts["mean"], label="Mean", linestyle="--")
    ax.plot(forecasts.index, forecasts["naive"], label="Naive", linestyle="--")
    ax.plot(
        forecasts.index,
        forecasts["seasonal_naive"],
        label="Seasonal naive",
        linestyle="--",
    )
    ax.plot(forecasts.index, forecasts["drift"], label="Drift", linestyle="--")

    ax.set_title("Two-year benchmark forecasts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (GW)")
    ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(PART2_FIGURES_DIR / "benchmark_forecasts_2y.png", dpi=150)
    plt.close(fig)


def plot_model_forecasts(forecasts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(forecasts.index, forecasts["actual"], label="Actual", color="black")
    ax.plot(
        forecasts.index,
        forecasts["seasonal_naive"],
        label="Seasonal naive",
        color="gray",
        linestyle="--",
    )
    ax.plot(forecasts.index, forecasts["sarima"], label="SARIMA", color="tab:blue")
    ax.plot(
        forecasts.index,
        forecasts["sarimax_temp"],
        label="SARIMAX (temp)",
        color="tab:orange",
    )

    ax.set_title("Two-year forecasts: seasonal naive vs SARIMA vs SARIMAX")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (GW)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PART2_FIGURES_DIR / "model_forecasts_2y.png", dpi=150)
    plt.close(fig)


def plot_sarima_confidence_intervals(df: pd.DataFrame) -> None:
    train = df.iloc[:-TEST_HORIZON_WEEKS]
    test = df.iloc[-TEST_HORIZON_WEEKS:]

    y_train = train[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    model = SARIMAX(
        y_train,
        order=(1, 1, 2),
        seasonal_order=(0, 1, 1, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    forecast_res = result.get_forecast(steps=TEST_HORIZON_WEEKS)
    mean_forecast = forecast_res.predicted_mean
    ci = forecast_res.conf_int(alpha=0.05)

    lower_col = ci.columns[0]
    upper_col = ci.columns[1]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test.index, y_test, label="Actual", color="black")
    ax.plot(test.index, mean_forecast, label="SARIMA forecast", color="tab:blue")
    ax.fill_between(
        test.index,
        ci[lower_col].values,
        ci[upper_col].values,
        color="tab:blue",
        alpha=0.2,
        label="95% CI",
    )

    ax.set_title("SARIMA forecast with 95% confidence intervals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Load (GW)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PART3_FIGURES_DIR / "sarima_forecast_ci_2y.png", dpi=150)
    plt.close(fig)

    ci_out = pd.DataFrame(
        {
            "actual": y_test.values,
            "sarima_forecast": mean_forecast.values,
            "ci_lower": ci[lower_col].values,
            "ci_upper": ci[upper_col].values,
        },
        index=test.index,
    )
    ci_out.to_csv(PART3_RESULTS_DIR / "sarima_forecast_ci_values.csv")


def main() -> None:
    weekly = load_weekly_data()
    forecasts = load_forecasts()

    plot_weekly_load(weekly)
    plot_load_temperature_eda(weekly)
    plot_benchmark_forecasts(forecasts)
    plot_model_forecasts(forecasts)
    plot_sarima_confidence_intervals(weekly)

    print("Saved figures to outputs/figures/part1, part2, and part3/")


def plot_sarima_forecast_with_ci(
    train: pd.Series,
    test: pd.Series,
    sarima_mean_forecast: pd.Series,
    sarima_ci: pd.DataFrame,
    save_path,
    title: str = "Weekly electricity demand: SARIMA forecast vs actual",
):
    plt.figure(figsize=(14, 6))

    plt.plot(train.index, train, label="Train", color="C0", linewidth=1.5)
    plt.plot(test.index, test, label="Test", color="C1", linewidth=2)
    plt.plot(
        sarima_mean_forecast.index,
        sarima_mean_forecast,
        label="SARIMA forecast",
        color="C3",
        linewidth=2,
    )

    plt.fill_between(
        sarima_ci.index,
        sarima_ci.iloc[:, 0],
        sarima_ci.iloc[:, 1],
        color="C3",
        alpha=0.2,
        label="95% prediction interval",
    )

    plt.title(title)
    plt.ylabel("Load (GW)")
    plt.xlabel("Date")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
if __name__ == "__main__":
    main()