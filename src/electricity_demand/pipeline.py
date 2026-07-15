from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from electricity_demand.evaluation import evaluate_many_forecasts
from electricity_demand.models.feature_models import (
    get_feature_importance,
    make_feature_matrix,
    predict_feature_model,
    prepare_xy,
    train_gradient_boosting,
    train_linear_regression,
    train_random_forest,
    train_test_split_time,
)
from electricity_demand.models.neural import (
    inverse_scale_actuals,
    predict_lstm_model,
    prepare_lstm_datasets,
    train_lstm_model,
    tune_lstm_hyperparameters,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

INPUT_FILE = PROCESSED_DIR / "weekly_load_temperature_features.csv"
PREDICTIONS_FILE = PROCESSED_DIR / "part5_feature_model_predictions.csv"
METRICS_FILE = PROCESSED_DIR / "part5_feature_model_metrics.csv"
RF_IMPORTANCE_FILE = PROCESSED_DIR / "part5_random_forest_feature_importance.csv"
PLOT_FILE = FIGURES_DIR / "part5_actual_vs_predicted.png"

HOURLY_INPUT_FILE = PROCESSED_DIR / "hourly_load_temperature_features.csv"
PART6_PREDICTIONS_FILE = PROCESSED_DIR / "part6_lstm_predictions.csv"
PART6_METRICS_FILE = PROCESSED_DIR / "part6_lstm_metrics.csv"
PART6_TUNING_FILE = PROCESSED_DIR / "part6_lstm_tuning_results.csv"
PART6_PLOT_FILE = FIGURES_DIR / "part6_lstm_actual_vs_predicted.png"


def load_modeling_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE, parse_dates=True, index_col=0)
    df = df.sort_index()
    return df


def load_hourly_modeling_data() -> pd.DataFrame:
    df = pd.read_csv(HOURLY_INPUT_FILE, parse_dates=True, index_col=0)
    df = df.sort_index()
    return df


def save_forecast_plot(
    actual: pd.Series,
    forecasts: pd.DataFrame,
    output_path: Path,
    title: str = "Part 5: Actual vs Predicted Electricity Demand",
) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual.values, label="Actual", linewidth=2)

    for col in forecasts.columns:
        plt.plot(forecasts.index, forecasts[col].values, label=col, linestyle="--")

    plt.title(title)
    plt.xlabel("Week")
    plt.ylabel("Load (GW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_lstm_forecast_plot(
    actual: pd.Series,
    forecast: pd.Series,
    output_path: Path,
    title: str = "Part 6: LSTM Actual vs Predicted Electricity Demand",
) -> None:
    plt.figure(figsize=(14, 6))
    plt.plot(actual.index, actual.values, label="Actual", linewidth=2)
    plt.plot(forecast.index, forecast.values, label=forecast.name, linestyle="--")
    plt.title(title)
    plt.xlabel("Hour")
    plt.ylabel("Load (GW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_part5_workflow(test_size: int = 104) -> dict[str, pd.DataFrame | pd.Series]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_modeling_data()
    features = make_feature_matrix(df, target_col="load_gw")
    X, y = prepare_xy(features, target_col="load_gw")

    X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_size=test_size)

    linear_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)

    linear_pred = predict_feature_model(
        linear_model,
        X_test,
        index=y_test.index,
        name="LinearRegression",
    )
    rf_pred = predict_feature_model(
        rf_model,
        X_test,
        index=y_test.index,
        name="RandomForest",
    )
    gb_pred = predict_feature_model(
        gb_model,
        X_test,
        index=y_test.index,
        name="GradientBoosting",
    )

    forecasts = pd.concat([linear_pred, rf_pred, gb_pred], axis=1)
    forecasts["Actual"] = y_test

    metrics = evaluate_many_forecasts(
        actual=y_test,
        forecasts=forecasts[["LinearRegression", "RandomForest", "GradientBoosting"]],
        train_series=y_train,
        seasonal_period=52,
        sort_by="RMSE",
    )

    rf_importance = get_feature_importance(rf_model, X_train)

    forecasts.to_csv(PREDICTIONS_FILE)
    metrics.to_csv(METRICS_FILE, index=False)
    rf_importance.to_csv(RF_IMPORTANCE_FILE, index=False)

    save_forecast_plot(
        actual=y_test,
        forecasts=forecasts[["LinearRegression", "RandomForest", "GradientBoosting"]],
        output_path=PLOT_FILE,
    )

    return {
        "features": features,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "forecasts": forecasts,
        "metrics": metrics,
        "rf_importance": rf_importance,
    }


def run_part6_workflow(
    test_size: int = 24 * 365 * 2,
    lookback: int = 24 * 7,
) -> dict[str, pd.DataFrame | pd.Series]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_hourly_modeling_data()

    lstm_data = prepare_lstm_datasets(
        df=df,
        target_col="load_gw",
        test_size=test_size,
        lookback=lookback,
    )

    best_config, tuning_results = tune_lstm_hyperparameters(
        X_train=lstm_data["X_train"],
        y_train=lstm_data["y_train"],
        verbose=0,
    )

    best_params = {
        "lstm_units_1": int(best_config["lstm_units_1"]),
        "lstm_units_2": int(best_config["lstm_units_2"]),
        "dropout": float(best_config["dropout"]),
        "learning_rate": float(best_config["learning_rate"]),
        "batch_size": int(best_config["batch_size"]),
        "epochs": int(best_config["epochs"]),
    }

    model, history = train_lstm_model(
        X_train=lstm_data["X_train"],
        y_train=lstm_data["y_train"],
        lstm_units_1=best_params["lstm_units_1"],
        lstm_units_2=best_params["lstm_units_2"],
        dropout=best_params["dropout"],
        learning_rate=best_params["learning_rate"],
        batch_size=best_params["batch_size"],
        epochs=best_params["epochs"],
        verbose=1,
    )

    actual = inverse_scale_actuals(
        y_scaled=lstm_data["y_test"],
        scaler=lstm_data["scaler"],
        feature_cols=lstm_data["feature_cols"],
        target_col_idx=lstm_data["target_col_idx"],
        index=lstm_data["test_index"],
        name="Actual",
    )

    forecast = predict_lstm_model(
        model=model,
        X_test=lstm_data["X_test"],
        scaler=lstm_data["scaler"],
        feature_cols=lstm_data["feature_cols"],
        target_col_idx=lstm_data["target_col_idx"],
        index=lstm_data["test_index"],
        name="LSTM",
    )

    forecasts = pd.concat([forecast, actual], axis=1)

    metrics = evaluate_many_forecasts(
        actual=actual,
        forecasts=forecasts[["LSTM"]],
        train_series=lstm_data["train_df"]["load_gw"],
        seasonal_period=24,
        sort_by="RMSE",
    )

    forecasts.to_csv(PART6_PREDICTIONS_FILE)
    metrics.to_csv(PART6_METRICS_FILE, index=False)
    tuning_results.to_csv(PART6_TUNING_FILE, index=False)

    save_lstm_forecast_plot(
        actual=actual,
        forecast=forecast,
        output_path=PART6_PLOT_FILE,
    )

    return {
        "hourly_frame": lstm_data["frame"],
        "train_df": lstm_data["train_df"],
        "test_df": lstm_data["test_df"],
        "X_train": lstm_data["X_train"],
        "X_test": lstm_data["X_test"],
        "y_train": lstm_data["y_train"],
        "y_test": lstm_data["y_test"],
        "actual": actual,
        "forecast": forecast,
        "forecasts": forecasts,
        "metrics": metrics,
        "tuning_results": tuning_results,
        "best_config": pd.DataFrame([best_params]),
    }