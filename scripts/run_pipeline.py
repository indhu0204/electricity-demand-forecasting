from pathlib import Path

import pandas as pd

from electricity_demand.config import (
    METRICS_DIR,
    PROCESSED_WEEKLY_FILENAME,
    SEASONAL_PERIOD_WEEKS,
    TARGET_COLUMN,
    TEST_HORIZON_WEEKS,
)
from electricity_demand.data import load_processed_data
from electricity_demand.evaluation import evaluate_many_forecasts
from electricity_demand.models.benchmarks import (
    make_benchmark_forecasts,
    train_test_split_last_n,
)
from electricity_demand.models.sarimax import (
    forecast_sarima,
    forecast_sarimax,
    grid_search_sarima_aic,
    save_sarima_diagnostics,
    sarima_aic_table,
)
from electricity_demand.pipeline import run_part5_workflow, run_part6_workflow


PART2_FORECASTS_DIR = Path("outputs/forecasts/part2")
PART2_RESULTS_DIR = Path("outputs/results/part2")
PART2_METRICS_DIR = METRICS_DIR / "part2"

PART3_FIGURES_DIR = Path("outputs/figures/part3")
PART3_RESULTS_DIR = Path("outputs/results/part3")
PART3_METRICS_DIR = METRICS_DIR / "part3"


for path in [
    PART2_FORECASTS_DIR,
    PART2_RESULTS_DIR,
    PART2_METRICS_DIR,
    PART3_FIGURES_DIR,
    PART3_RESULTS_DIR,
    PART3_METRICS_DIR,
    METRICS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_processed_data(PROCESSED_WEEKLY_FILENAME)

    train_df, test_df, y_train, y_test = train_test_split_last_n(
        df,
        target_col=TARGET_COLUMN,
        test_horizon=TEST_HORIZON_WEEKS,
    )

    benchmark_forecasts = make_benchmark_forecasts(
        y_train=y_train,
        y_test=y_test,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )

    y_train_sarima, y_test_sarima, sarima_forecast = forecast_sarima(
        df=df,
        target_col=TARGET_COLUMN,
        test_horizon=TEST_HORIZON_WEEKS,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )

    y_train_sarimax, y_test_sarimax, sarimax_forecast = forecast_sarimax(
        df=df,
        target_col=TARGET_COLUMN,
        test_horizon=TEST_HORIZON_WEEKS,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )

    all_forecasts = benchmark_forecasts.copy()
    all_forecasts["sarima"] = sarima_forecast.values
    all_forecasts["sarimax_temp"] = sarimax_forecast.values
    all_forecasts.insert(0, "actual", y_test.values)
    all_forecasts.to_csv(PART2_FORECASTS_DIR / "all_forecasts.csv", index=True)

    split_sizes = pd.DataFrame(
        {
            "set": ["train", "test"],
            "n_rows": [len(train_df), len(test_df)],
        }
    )
    split_sizes.to_csv(PART2_RESULTS_DIR / "train_test_sizes.csv", index=False)

    train_df.reset_index()[["date"]].to_csv(
        PART2_RESULTS_DIR / "train_index.csv",
        index=False,
    )
    test_df.reset_index()[["date"]].to_csv(
        PART2_RESULTS_DIR / "test_index.csv",
        index=False,
    )

    metrics_benchmarks = evaluate_many_forecasts(
        actual=y_test,
        forecasts=benchmark_forecasts,
        train_series=y_train,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )
    metrics_benchmarks.to_csv(PART2_RESULTS_DIR / "benchmark_metrics.csv", index=False)

    metrics_sarima = evaluate_many_forecasts(
        actual=y_test_sarima,
        forecasts=sarima_forecast.to_frame(name="sarima"),
        train_series=y_train_sarima,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )
    metrics_sarima.to_csv(PART3_METRICS_DIR / "sarima_metrics.csv", index=False)

    sarima_aic_results = sarima_aic_table(
        y_train=y_train_sarima,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )
    sarima_aic_results.to_csv(PART3_RESULTS_DIR / "sarima_aic_grid.csv", index=False)

    best_order, best_seasonal_order, best_aic = grid_search_sarima_aic(
        y_train=y_train_sarima,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )

    pd.DataFrame(
        {
            "model": ["sarima"],
            "order": [str(best_order)],
            "seasonal_order": [str(best_seasonal_order)],
            "aic": [best_aic],
        }
    ).to_csv(PART3_RESULTS_DIR / "sarima_best_model.csv", index=False)

    save_sarima_diagnostics(
        y_train=y_train_sarima,
        order=best_order,
        seasonal_order=best_seasonal_order,
        figures_dir=PART3_FIGURES_DIR,
        results_dir=PART3_RESULTS_DIR,
        model_name="sarima",
    )

    metrics_sarimax = evaluate_many_forecasts(
        actual=y_test_sarimax,
        forecasts=sarimax_forecast.to_frame(name="sarimax_temp"),
        train_series=y_train_sarimax,
        seasonal_period=SEASONAL_PERIOD_WEEKS,
    )

    metrics = pd.concat(
        [metrics_benchmarks, metrics_sarima, metrics_sarimax],
        ignore_index=True,
    )
    metrics.to_csv(PART2_METRICS_DIR / "model_comparison.csv", index=False)

    part5_results = run_part5_workflow(test_size=TEST_HORIZON_WEEKS)
    print(part5_results["metrics"])

    print("Saved:")
    print(PART2_FORECASTS_DIR / "all_forecasts.csv")
    print(PART2_RESULTS_DIR / "train_test_sizes.csv")
    print(PART2_RESULTS_DIR / "train_index.csv")
    print(PART2_RESULTS_DIR / "test_index.csv")
    print(PART2_RESULTS_DIR / "benchmark_metrics.csv")
    print(PART2_METRICS_DIR / "model_comparison.csv")
    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")

    print(PART3_METRICS_DIR / "sarima_metrics.csv")
    print(PART3_RESULTS_DIR / "sarima_aic_grid.csv")
    print(PART3_RESULTS_DIR / "sarima_best_model.csv")
    print(PART3_RESULTS_DIR / "sarima_residuals.csv")
    print(PART3_RESULTS_DIR / "sarima_ljung_box.csv")
    print(PART3_RESULTS_DIR / "sarima_diagnostics_summary.csv")
    print(PART3_FIGURES_DIR / "sarima_residual_acf.png")
    print(PART3_FIGURES_DIR / "sarima_residual_histogram.png")
    print(PART3_FIGURES_DIR / "sarima_residual_qq.png")

    print("\nPart 6: LSTM")
    part6_results = run_part6_workflow()
    print(part6_results["metrics"])
    print("Saved:")
    print("data/processed/part6_lstm_predictions.csv")
    print("data/processed/part6_lstm_metrics.csv")
    print("data/processed/part6_lstm_tuning_results.csv")
    print("reports/figures/part6_lstm_actual_vs_predicted.png")


if __name__ == "__main__":
    main()