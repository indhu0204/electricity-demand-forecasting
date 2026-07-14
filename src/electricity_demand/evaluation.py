from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0
    if not np.any(mask):
        return np.nan

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    if not np.any(mask):
        return np.nan

    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def mase(y_true, y_pred, y_train, seasonal_period: int = 52) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    if len(y_train) <= seasonal_period:
        return np.nan

    scale = np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period]))
    if scale == 0:
        return np.nan

    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def evaluate_forecast(
    y_true,
    y_pred,
    y_train,
    model_name: str,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "MAE": mae(y_true, y_pred),
                "RMSE": rmse(y_true, y_pred),
                "MAPE": mape(y_true, y_pred),
                "sMAPE": smape(y_true, y_pred),
                "MASE": mase(y_true, y_pred, y_train, seasonal_period=seasonal_period),
                "Bias": bias(y_true, y_pred),
            }
        ]
    )


def evaluate_many_forecasts(
    actual: pd.Series,
    forecasts: pd.DataFrame,
    train_series: pd.Series,
    seasonal_period: int = 52,
    sort_by: str | None = "RMSE",
) -> pd.DataFrame:
    rows = []

    for col in forecasts.columns:
        metrics = evaluate_forecast(
            y_true=actual.values,
            y_pred=forecasts[col].values,
            y_train=train_series.values,
            model_name=col,
            seasonal_period=seasonal_period,
        )
        rows.append(metrics)

    results = pd.concat(rows, ignore_index=True)

    if sort_by is not None and sort_by in results.columns:
        results = results.sort_values(sort_by, ascending=True).reset_index(drop=True)

    return results