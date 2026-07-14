from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
warnings.filterwarnings("ignore")


def train_test_split_last_n(
    df: pd.DataFrame,
    target_col: str,
    test_horizon: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = df.copy().sort_index()
    train = data.iloc[:-test_horizon]
    test = data.iloc[-test_horizon:]
    y_train = train[target_col]
    y_test = test[target_col]
    return train, test, y_train, y_test


def _clean_series(y: pd.Series) -> pd.Series:
    y = pd.Series(y).astype(float)
    y = y.replace([np.inf, -np.inf], np.nan).dropna()
    return y


def grid_search_sarima_aic(
    y_train: pd.Series,
    seasonal_period: int = 52,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int], float]:
    y_train = _clean_series(y_train)

    candidate_orders = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
    ]
    candidate_seasonal_orders = [
        (0, 1, 1, seasonal_period),
        (1, 1, 1, seasonal_period),
    ]

    best_aic = np.inf
    best_order = None
    best_seasonal_order = None

    for order in candidate_orders:
        for seasonal_order in candidate_seasonal_orders:
            try:
                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)

                if np.isfinite(result.aic) and result.aic < best_aic:
                    best_aic = result.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
            except Exception:
                continue

    if best_order is None or best_seasonal_order is None or not np.isfinite(best_aic):
        fallback_order = (1, 1, 1)
        fallback_seasonal_order = (0, 1, 1, seasonal_period)
        return fallback_order, fallback_seasonal_order, np.nan

    return best_order, best_seasonal_order, best_aic


def sarima_aic_table(
    y_train: pd.Series,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    y_train = _clean_series(y_train)

    candidate_orders = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
    ]
    candidate_seasonal_orders = [
        (0, 1, 1, seasonal_period),
        (1, 1, 1, seasonal_period),
    ]

    rows = []

    for order in candidate_orders:
        for seasonal_order in candidate_seasonal_orders:
            try:
                model = SARIMAX(
                    y_train,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False)

                if np.isfinite(result.aic):
                    rows.append(
                        {
                            "p": order[0],
                            "d": order[1],
                            "q": order[2],
                            "P": seasonal_order[0],
                            "D": seasonal_order[1],
                            "Q": seasonal_order[2],
                            "s": seasonal_order[3],
                            "aic": result.aic,
                            "bic": result.bic,
                        }
                    )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame(
            columns=["p", "d", "q", "P", "D", "Q", "s", "aic", "bic"]
        )

    return pd.DataFrame(rows).sort_values("aic").reset_index(drop=True)


def fit_sarima(
    y_train: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> SARIMAX:
    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model


def fit_sarimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> SARIMAX:
    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model


def save_sarima_diagnostics(
    y_train: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    figures_dir: Path,
    results_dir: Path,
    model_name: str = "sarima",
) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    y_train = _clean_series(y_train)

    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    result = model.fit(disp=False)

    residuals = pd.Series(result.resid, index=y_train.index).dropna()

    residuals.to_frame(name="residual").to_csv(
        results_dir / f"{model_name}_residuals.csv",
        index=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(residuals, lags=min(40, max(1, len(residuals) // 2 - 1)), ax=ax)
    ax.set_title("Residual ACF")
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"{model_name}_residual_acf.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(residuals, bins=20, edgecolor="black")
    ax.set_title("Residual Histogram")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"{model_name}_residual_histogram.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    sm.qqplot(residuals, line="s", ax=ax)
    ax.set_title("Residual QQ Plot")
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"{model_name}_residual_qq.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    ljung_box = acorr_ljungbox(residuals, lags=[10, 20], return_df=True)
    ljung_box.to_csv(
        results_dir / f"{model_name}_ljung_box.csv",
        index=True,
    )

    diagnostics_summary = pd.DataFrame(
        {
            "model": [model_name],
            "order": [str(order)],
            "seasonal_order": [str(seasonal_order)],
            "aic": [result.aic],
            "bic": [result.bic],
        }
    )
    diagnostics_summary.to_csv(
        results_dir / f"{model_name}_diagnostics_summary.csv",
        index=False,
    )

    return diagnostics_summary


def forecast_sarima(
    df: pd.DataFrame,
    target_col: str,
    test_horizon: int,
    seasonal_period: int = 52,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    train_df, test_df, y_train, y_test = train_test_split_last_n(
        df,
        target_col=target_col,
        test_horizon=test_horizon,
    )

    y_train = _clean_series(y_train)

    order, seasonal_order, best_aic = grid_search_sarima_aic(
        y_train=y_train,
        seasonal_period=seasonal_period,
    )
    print(f"Best SARIMA order={order}, seasonal_order={seasonal_order}, AIC={best_aic}")

    model = fit_sarima(
        y_train=y_train,
        order=order,
        seasonal_order=seasonal_order,
    )
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=test_horizon)
    mean_forecast = pd.Series(forecast.predicted_mean, index=test_df.index)

    return y_train, y_test, mean_forecast


def forecast_sarimax(
    df: pd.DataFrame,
    target_col: str,
    test_horizon: int,
    seasonal_period: int = 52,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    train_df, test_df, y_train, y_test = train_test_split_last_n(
        df,
        target_col=target_col,
        test_horizon=test_horizon,
    )

    exog_cols = [
        "temp_mean",
        "heating_degree_days",
        "cooling_degree_days",
        "has_holiday",
    ]

    train_subset = train_df[[target_col] + exog_cols].dropna()
    test_subset = test_df[[target_col] + exog_cols].dropna()

    y_train_clean = _clean_series(train_subset[target_col])
    y_test_clean = test_subset[target_col]
    train_exog = train_subset.loc[y_train_clean.index, exog_cols]
    test_exog = test_subset[exog_cols]

    forecast_steps = len(test_subset)

    order, seasonal_order, best_aic = grid_search_sarima_aic(
        y_train=y_train_clean,
        seasonal_period=seasonal_period,
    )
    print(
        f"Best SARIMAX order={order}, seasonal_order={seasonal_order}, AIC={best_aic}"
    )

    model = fit_sarimax(
        y_train=y_train_clean,
        exog_train=train_exog,
        order=order,
        seasonal_order=seasonal_order,
    )
    result = model.fit(disp=False)

    forecast = result.get_forecast(steps=forecast_steps, exog=test_exog)
    mean_forecast = pd.Series(forecast.predicted_mean, index=test_subset.index)

    return y_train_clean, y_test_clean, mean_forecast