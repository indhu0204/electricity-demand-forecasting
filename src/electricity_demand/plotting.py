import matplotlib.pyplot as plt
import pandas as pd


def plot_forecasts(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict,
    title: str,
    save_path,
):
    plt.figure(figsize=(14, 6))
    plt.plot(train.index, train, label="Training data", linewidth=1.5)
    plt.plot(test.index, test, label="Test data", color="black", linewidth=2)

    for name, forecast in forecasts.items():
        plt.plot(forecast.index, forecast, label=name, linestyle="--")

    plt.title(title)
    plt.ylabel("Average load (GW)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_test_forecasts_only(
    test: pd.Series,
    forecasts: dict,
    title: str,
    save_path,
):
    plt.figure(figsize=(14, 6))
    plt.plot(test.index, test, label="Actual", color="black", linewidth=2)

    for name, forecast in forecasts.items():
        plt.plot(forecast.index, forecast, label=name, linestyle="--")

    plt.title(title)
    plt.ylabel("Average load (GW)")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()