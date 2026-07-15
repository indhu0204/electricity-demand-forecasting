from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


DEFAULT_LAGS = [1, 2, 3, 4, 12, 24, 52]
DEFAULT_WINDOWS = [4, 12, 24, 52]


def make_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "load_gw",
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    features = df.copy()

    if lags is None:
        lags = DEFAULT_LAGS
    if windows is None:
        windows = DEFAULT_WINDOWS

    features["year"] = features.index.year
    features["month"] = features.index.month
    features["quarter"] = features.index.quarter
    features["week_of_year"] = features.index.isocalendar().week.astype(int)

    for lag in lags:
        features[f"{target_col}_lag_{lag}"] = features[target_col].shift(lag)

    for window in windows:
        features[f"{target_col}_roll_mean_{window}"] = (
            features[target_col].shift(1).rolling(window).mean()
        )
        features[f"{target_col}_roll_std_{window}"] = (
            features[target_col].shift(1).rolling(window).std()
        )

    features = features.dropna()
    return features


def prepare_xy(
    df: pd.DataFrame,
    target_col: str = "load_gw",
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def train_test_split_time(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: int = 104,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = X.iloc[:-test_size].copy()
    X_test = X.iloc[-test_size:].copy()
    y_train = y.iloc[:-test_size].copy()
    y_test = y.iloc[-test_size:].copy()
    return X_train, X_test, y_train, y_test


def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict_feature_model(
    model,
    X_test: pd.DataFrame,
    index: pd.Index | None = None,
    name: str = "prediction",
) -> pd.Series:
    preds = model.predict(X_test)
    if index is None:
        index = X_test.index
    return pd.Series(preds, index=index, name=name)


def get_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    return importance.reset_index(drop=True)

from sklearn.ensemble import GradientBoostingRegressor


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model

