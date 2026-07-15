# scripts/run_sarima_ci_plot.py
from electricity_demand.models.plotting import plot_sarima_forecast_with_ci
from pathlib import Path

import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from electricity_demand.config import (
    PROCESSED_WEEKLY_FILENAME,
    SEASONAL_PERIOD_WEEKS,
    TARGET_COLUMN,
    TEST_HORIZON_WEEKS,
)
from electricity_demand.data import load_processed_data
from electricity_demand.models.sarimax import (
    forecast_sarima,
    grid_search_sarima_aic,
)


def main():
    df = load_processed_data(PROCESSED_WEEKLY_FILENAME)

    y_train_sarima, y_test_sarima, _ = forecast_sarima(
        df=df,
        target_col=TARGET_COLUMN,
        test_horizon=TEST_HORIZON_WEEKS,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )

    best_order, best_seasonal_order, best_aic = grid_search_sarima_aic(
        y_train=y_train_sarima,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )

    print(
        f"Best SARIMA order={best_order}, "
        f"seasonal_order={best_seasonal_order}, "
        f"AIC={best_aic}"
    )

    model = SARIMAX(
        y_train_sarima,
        order=best_order,
        seasonal_order=best_seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    forecast_res = result.get_forecast(steps=len(y_test_sarima))
    forecast_mean = forecast_res.predicted_mean
    forecast_ci = forecast_res.conf_int(alpha=0.05)

    figures_dir = Path("outputs/figures/part3")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "sarima_weekly_forecast.png"

    plt.figure(figsize=(14, 6))
    plt.plot(y_train_sarima.index, y_train_sarima, label="Train", color="C0", linewidth=1.5)
    plt.plot(y_test_sarima.index, y_test_sarima, label="Test", color="C1", linewidth=2)
    plt.plot(forecast_mean.index, forecast_mean, label="SARIMA forecast", color="C3", linewidth=2)

    plt.fill_between(
        forecast_ci.index.to_pydatetime(),
        forecast_ci.iloc[:, 0].astype(float).to_numpy(),
        forecast_ci.iloc[:, 1].astype(float).to_numpy(),
        color="C3",
        alpha=0.2,
        label="95% prediction interval",
    )

    plt.title("Weekly electricity demand: SARIMA forecast vs actual")
    plt.xlabel("Date")
    plt.ylabel("Load (GW)")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()