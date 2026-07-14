import numpy as np
import pandas as pd


def train_test_split_last_n(df: pd.DataFrame, target_col: str, test_horizon: int = 104):
    data = df.copy().sort_index()
    train = data.iloc[:-test_horizon]
    test = data.iloc[-test_horizon:]
    y_train = train[target_col]
    y_test = test[target_col]
    return train, test, y_train, y_test


def mean_forecast(y_train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(y_train.mean(), horizon)


def naive_forecast(y_train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(y_train.iloc[-1], horizon)


def seasonal_naive_forecast(y_train: pd.Series, horizon: int, seasonal_period: int = 52) -> np.ndarray:
    history = y_train.values
    if len(history) < seasonal_period:
        raise ValueError("Training series shorter than seasonal period.")

    forecast = []
    for i in range(horizon):
        forecast.append(history[-seasonal_period + (i % seasonal_period)])
    return np.array(forecast)


def drift_forecast(y_train: pd.Series, horizon: int) -> np.ndarray:
    y0 = y_train.iloc[0]
    yt = y_train.iloc[-1]
    n = len(y_train) - 1

    if n <= 0:
        return np.repeat(yt, horizon)

    slope = (yt - y0) / n
    return np.array([yt + slope * h for h in range(1, horizon + 1)])


def make_benchmark_forecasts(
    y_train: pd.Series,
    y_test: pd.Series,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    horizon = len(y_test)
    forecasts = pd.DataFrame(index=y_test.index)
    forecasts["mean"] = mean_forecast(y_train, horizon)
    forecasts["naive"] = naive_forecast(y_train, horizon)
    forecasts["seasonal_naive"] = seasonal_naive_forecast(y_train, horizon, seasonal_period=seasonal_period)
    forecasts["drift"] = drift_forecast(y_train, horizon)
    return forecasts