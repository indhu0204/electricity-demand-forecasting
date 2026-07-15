from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUTS_DIR = ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FORECASTS_DIR = OUTPUTS_DIR / "forecasts"
FIGURES_DIR = OUTPUTS_DIR / "figures"

PART7_METRICS_DIR = METRICS_DIR / "part7"
PART7_FIGURES_DIR = FIGURES_DIR / "part7"

FINAL_METRICS_PATH = PART7_METRICS_DIR / "all_model_metrics_comparison.csv"
FINAL_PLOT_PATH = PART7_FIGURES_DIR / "final_model_comparison.png"
ALL_FORECASTS_PATH = FORECASTS_DIR / "all_model_forecasts.csv"


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def standardize_metrics_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    df = clean_columns(df)

    rename_map = {
        "model_name": "model",
        "method": "model",
        "rmse_gw": "rmse",
        "mae_gw": "mae",
        "mean_absolute_error": "mae",
        "root_mean_squared_error": "rmse",
        "forecast_bias": "bias",
        "mean_bias": "bias",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "model" not in df.columns:
        df["model"] = source_name

    required_cols = ["model", "mae", "rmse", "mase", "bias"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[required_cols].copy()
    df["source_file"] = source_name
    return df


def load_metrics_files() -> list:
    metric_files = []

    explicit_files = [
        PROCESSED_DIR / "part4_feature_model_metrics.csv",
        PROCESSED_DIR / "part5_feature_model_metrics.csv",
        PROCESSED_DIR / "part6_lstm_metrics.csv",
    ]

    for file in explicit_files:
        if file.exists():
            metric_files.append(file)

    for folder in [METRICS_DIR / "part2", METRICS_DIR / "part3"]:
        if folder.exists():
            metric_files.extend(sorted(folder.glob("*.csv")))

    return metric_files


def build_final_metrics_csv():
    metric_files = load_metrics_files()
    all_frames = []

    if not metric_files:
        print("No metrics files found.")
        return None

    for file in metric_files:
        try:
            df = pd.read_csv(file)
            std = standardize_metrics_df(df, file.name.replace(".csv", ""))
            all_frames.append(std)
        except Exception as e:
            print(f"Skipping metrics file {file}: {e}")

    if not all_frames:
        print("No metrics tables could be loaded.")
        return None

    final_metrics = pd.concat(all_frames, ignore_index=True)

    for col in ["mae", "rmse", "mase", "bias"]:
        final_metrics[col] = pd.to_numeric(final_metrics[col], errors="coerce")

    final_metrics = final_metrics.sort_values(by=["rmse", "mae"], na_position="last")

    PART7_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    final_metrics.to_csv(FINAL_METRICS_PATH, index=False)

    print(f"Saved final metrics CSV to: {FINAL_METRICS_PATH}")
    print(final_metrics)
    return final_metrics


def guess_time_column(df: pd.DataFrame):
    candidates = ["datetime", "timestamp", "date", "week", "ds"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def guess_actual_column(df: pd.DataFrame):
    candidates = ["actual", "y_true", "observed", "load", "target"]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_final_comparison_plot():
    if not ALL_FORECASTS_PATH.exists():
        print(f"Forecast file not found: {ALL_FORECASTS_PATH}")
        return

    df = pd.read_csv(ALL_FORECASTS_PATH)
    df = clean_columns(df)

    time_col = guess_time_column(df)
    actual_col = guess_actual_column(df)

    if time_col is not None:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
            x = df[time_col]
        except Exception:
            x = df.index
    else:
        x = df.index

    plt.figure(figsize=(14, 7))

    if actual_col is not None:
        plt.plot(x, df[actual_col], label="Observed", color="black", linewidth=2.2)

    excluded = set(filter(None, [time_col, actual_col]))

    forecast_cols = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]

    for col in forecast_cols:
        plt.plot(x, df[col], label=col.replace("_", " ").title(), linewidth=1.4, alpha=0.9)

    plt.title("Model forecasts vs observed demand")
    plt.xlabel("Time")
    plt.ylabel("Demand")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    PART7_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FINAL_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved final comparison plot to: {FINAL_PLOT_PATH}")


def main():
    build_final_metrics_csv()
    build_final_comparison_plot()


if __name__ == "__main__":
    main()