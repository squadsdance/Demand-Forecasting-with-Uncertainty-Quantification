from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


DATA_PATH = Path("data/train.csv")
RESULTS_DIR = Path("results")

STORE_ID = 1
ITEM_ID = 1

FEATURES = [
    "day_of_week",
    "month",
    "day_of_year",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_30",
]


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and prepare the dataset

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["store"] == STORE_ID)
        & (df["item"] == ITEM_ID)
    ].copy()

    df = df.sort_values("date")

    # Calendar features

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Historical sales features

    for lag in [1, 7, 14, 28]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    df["rolling_7"] = (
        df["sales"].shift(1).rolling(7).mean()
    )

    df["rolling_30"] = (
        df["sales"].shift(1).rolling(30).mean()
    )

    df = df.dropna()

    # Chronological train-test split

    train = df[df["date"] < "2016-01-01"]
    test = df[df["date"] >= "2017-01-01"]

    # Train the baseline forecasting model

    model = LGBMRegressor(
        objective="regression",
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1,
    )

    model.fit(
        train[FEATURES],
        train["sales"]
    )

    predictions = model.predict(test[FEATURES])

    # Evaluate prediction accuracy

    mae = mean_absolute_error(
        test["sales"],
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            test["sales"],
            predictions
        )
    )

    print("\n--- Baseline Forecasting Results ---")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Save actual vs predicted demand graph

    plt.figure(figsize=(12, 5))

    plt.plot(
        test["date"],
        test["sales"],
        label="Actual Demand",
        alpha=0.4
    )

    plt.plot(
        test["date"],
        predictions,
        label="Predicted Demand"
    )

    plt.title("Baseline Forecast: Actual vs Predicted Demand")

    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "baseline_forecast.png",
        dpi=300
    )

    plt.close()

    print("\nBaseline forecast saved to results/baseline_forecast.png")


if __name__ == "__main__":
    main()