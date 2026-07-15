from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler


def train_mlp_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
):
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def predict_neural_model(
    model,
    X_test: pd.DataFrame,
    index: pd.Index | None = None,
    name: str = "MLP",
) -> pd.Series:
    preds = model.predict(X_test)
    if index is None:
        index = X_test.index
    return pd.Series(preds, index=index, name=name)


def make_hourly_lstm_frame(
    df: pd.DataFrame,
    target_col: str = "load_gw",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    data = df.copy().sort_index()

    if feature_cols is None:
        feature_cols = [target_col]
        for col in [
            "temp_mean",
            "temp_min",
            "temp_max",
            "heating_degree_days",
            "cooling_degree_days",
            "holiday_days",
            "has_holiday",
        ]:
            if col in data.columns and col not in feature_cols:
                feature_cols.append(col)

    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    return data[feature_cols].dropna().copy()


def train_test_split_time_array(
    data: pd.DataFrame,
    test_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if test_size <= 0 or test_size >= len(data):
        raise ValueError("test_size must be between 1 and len(data)-1")
    train = data.iloc[:-test_size].copy()
    test = data.iloc[-test_size:].copy()
    return train, test


def scale_train_test_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)
    return train_scaled, test_scaled, scaler


def create_lstm_sequences(
    data: np.ndarray,
    lookback: int,
    target_col_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []

    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])
        y.append(data[i, target_col_idx])

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    return X_arr, y_arr


def prepare_lstm_datasets(
    df: pd.DataFrame,
    target_col: str = "load_gw",
    feature_cols: list[str] | None = None,
    test_size: int = 24 * 365 * 2,
    lookback: int = 24 * 7,
) -> dict:
    frame = make_hourly_lstm_frame(df, target_col=target_col, feature_cols=feature_cols)
    train_df, test_df = train_test_split_time_array(frame, test_size=test_size)

    combined_for_test = pd.concat([train_df.tail(lookback), test_df], axis=0)

    train_scaled, test_scaled_only, scaler = scale_train_test_data(train_df, test_df)
    combined_test_scaled = scaler.transform(combined_for_test)

    target_col_idx = frame.columns.get_loc(target_col)

    X_train, y_train = create_lstm_sequences(
        train_scaled,
        lookback=lookback,
        target_col_idx=target_col_idx,
    )
    X_test, y_test = create_lstm_sequences(
        combined_test_scaled,
        lookback=lookback,
        target_col_idx=target_col_idx,
    )

    test_index = test_df.index

    return {
        "frame": frame,
        "train_df": train_df,
        "test_df": test_df,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled_only,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "test_index": test_index,
        "scaler": scaler,
        "feature_cols": list(frame.columns),
        "target_col": target_col,
        "target_col_idx": target_col_idx,
        "lookback": lookback,
    }


def build_lstm_model(
    input_shape: tuple[int, int],
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam

    model = Sequential(
        [
            Input(shape=input_shape),
            LSTM(lstm_units_1, return_sequences=True),
            Dropout(dropout),
            LSTM(lstm_units_2),
            Dropout(dropout),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def train_lstm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 30,
    batch_size: int = 32,
    patience: int = 5,
    verbose: int = 1,
):
    from tensorflow.keras.callbacks import EarlyStopping

    if X_val is None or y_val is None:
        split_idx = max(int(len(X_train) * 0.9), 1)
        if split_idx >= len(X_train):
            split_idx = len(X_train) - 1

        X_fit, X_val = X_train[:split_idx], X_train[split_idx:]
        y_fit, y_val = y_train[:split_idx], y_train[split_idx:]
    else:
        X_fit, y_fit = X_train, y_train

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        lstm_units_1=lstm_units_1,
        lstm_units_2=lstm_units_2,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_fit,
        y_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[early_stopping],
        shuffle=False,
    )
    return model, history


def inverse_scale_target(
    scaled_values: np.ndarray,
    scaler: MinMaxScaler,
    n_features: int,
    target_col_idx: int = 0,
) -> np.ndarray:
    scaled_values = np.asarray(scaled_values).reshape(-1, 1)
    full = np.zeros((len(scaled_values), n_features), dtype=np.float32)
    full[:, target_col_idx] = scaled_values[:, 0]
    inv = scaler.inverse_transform(full)
    return inv[:, target_col_idx]


def predict_lstm_model(
    model,
    X_test: np.ndarray,
    scaler: MinMaxScaler,
    feature_cols: list[str],
    target_col_idx: int = 0,
    index: pd.Index | None = None,
    name: str = "LSTM",
) -> pd.Series:
    preds_scaled = model.predict(X_test, verbose=0).reshape(-1)
    preds = inverse_scale_target(
        preds_scaled,
        scaler=scaler,
        n_features=len(feature_cols),
        target_col_idx=target_col_idx,
    )
    if index is None:
        index = pd.RangeIndex(start=0, stop=len(preds), step=1)
    return pd.Series(preds, index=index, name=name)


def inverse_scale_actuals(
    y_scaled: np.ndarray,
    scaler: MinMaxScaler,
    feature_cols: list[str],
    target_col_idx: int = 0,
    index: pd.Index | None = None,
    name: str = "Actual",
) -> pd.Series:
    actual = inverse_scale_target(
        y_scaled,
        scaler=scaler,
        n_features=len(feature_cols),
        target_col_idx=target_col_idx,
    )
    if index is None:
        index = pd.RangeIndex(start=0, stop=len(actual), step=1)
    return pd.Series(actual, index=index, name=name)


def tune_lstm_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    configs: list[dict] | None = None,
    verbose: int = 0,
) -> tuple[dict, pd.DataFrame]:
    if configs is None:
        configs = [
            {
                "lstm_units_1": 32,
                "lstm_units_2": 16,
                "dropout": 0.1,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 25,
            },
            {
                "lstm_units_1": 64,
                "lstm_units_2": 32,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 30,
            },
            {
                "lstm_units_1": 64,
                "lstm_units_2": 32,
                "dropout": 0.3,
                "learning_rate": 0.0005,
                "batch_size": 64,
                "epochs": 35,
            },
        ]

    split_idx = max(int(len(X_train) * 0.9), 1)
    if split_idx >= len(X_train):
        split_idx = len(X_train) - 1

    X_fit, X_val = X_train[:split_idx], X_train[split_idx:]
    y_fit, y_val = y_train[:split_idx], y_train[split_idx:]

    results = []

    for i, cfg in enumerate(configs, start=1):
        model, history = train_lstm_model(
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_val,
            y_val=y_val,
            lstm_units_1=cfg["lstm_units_1"],
            lstm_units_2=cfg["lstm_units_2"],
            dropout=cfg["dropout"],
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            verbose=verbose,
        )

        best_val_loss = float(np.min(history.history["val_loss"]))
        epochs_run = len(history.history["loss"])

        row = {
            "config_id": i,
            **cfg,
            "best_val_loss": best_val_loss,
            "epochs_run": epochs_run,
        }
        results.append(row)

    results_df = pd.DataFrame(results).sort_values("best_val_loss").reset_index(drop=True)
    best_config = results_df.iloc[0].to_dict()
    return best_config, results_df