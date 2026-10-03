from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/train.csv")
RESULTS_DIR = Path("results")

STORE_ID = 1
ITEM_ID = 1

TRAIN_END = "2016-01-01"
CALIBRATION_END = "2017-01-01"

QUANTILES = [0.1, 0.5, 0.9]

TARGET_COVERAGE = 0.80

OVERSTOCK_COST = 2
STOCKOUT_COST = 8

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


# ============================================================
# DATA LOADING
# ============================================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df[
        (df["store"] == STORE_ID)
        & (df["item"] == ITEM_ID)
    ].copy()

    df = df.sort_values("date")

    if df.empty:
        raise ValueError(
            "No observations found for the selected store and item."
        )

    return df


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    for lag in [1, 7, 14, 28]:

        df[f"lag_{lag}"] = df["sales"].shift(lag)

    df["rolling_7"] = (
        df["sales"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["rolling_30"] = (
        df["sales"]
        .shift(1)
        .rolling(30)
        .mean()
    )

    return df.dropna()


# ============================================================
# TRAIN / CALIBRATION / TEST SPLIT
# ============================================================

def split_data(df):

    train = df[
        df["date"] < TRAIN_END
    ].copy()

    calibration = df[
        (df["date"] >= TRAIN_END)
        & (df["date"] < CALIBRATION_END)
    ].copy()

    test = df[
        df["date"] >= CALIBRATION_END
    ].copy()

    if train.empty or calibration.empty or test.empty:
        raise ValueError(
            "One or more dataset splits are empty."
        )

    return train, calibration, test


# ============================================================
# MODEL TRAINING
# ============================================================

def train_models(train):

    models = {}

    X_train = train[FEATURES]
    y_train = train["sales"]

    for q in QUANTILES:

        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )

        model.fit(X_train, y_train)

        models[q] = model

    return models


# ============================================================
# PREDICTION
# ============================================================

def predict_quantiles(models, dataset):
    X = dataset[FEATURES]

    raw_predictions = np.column_stack([
        models[q].predict(X) for q in QUANTILES
    ])

    ordered_predictions = np.sort(raw_predictions, axis=1)

    return {
        q: ordered_predictions[:, i]
        for i, q in enumerate(QUANTILES)
    }


# ============================================================
# CONFORMAL CALIBRATION
# ============================================================

def calibrate_intervals(calibration, predictions):

    actual = calibration["sales"].to_numpy()

    lower = predictions[0.1]
    upper = predictions[0.9]

    scores = np.maximum(
        lower - actual,
        actual - upper
    )

    scores = np.maximum(scores, 0)

    n = len(scores)

    alpha = 1 - TARGET_COVERAGE

    adjusted_quantile = (
        np.ceil((n + 1) * (1 - alpha)) / n
    )

    adjusted_quantile = min(
        adjusted_quantile,
        1.0
    )

    correction = np.quantile(
        scores,
        adjusted_quantile,
        method="higher"
    )

    return correction


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(test, predictions, correction):

    actual = test["sales"].to_numpy()

    median = predictions[0.5]

    lower = predictions[0.1]
    upper = predictions[0.9]

    calibrated_lower = np.maximum(
        lower - correction,
        0
    )

    calibrated_upper = upper + correction

    mae = mean_absolute_error(actual, median)

    rmse = np.sqrt(
        mean_squared_error(actual, median)
    )

    original_inside = (
        (actual >= lower)
        & (actual <= upper)
    )

    calibrated_inside = (
        (actual >= calibrated_lower)
        & (actual <= calibrated_upper)
    )

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "original_coverage": original_inside.mean(),
        "calibrated_coverage": calibrated_inside.mean(),
        "original_width": np.mean(upper - lower),
        "calibrated_width": np.mean(
            calibrated_upper - calibrated_lower
        ),
        "calibration_adjustment": correction,
    }

    return metrics, calibrated_lower, calibrated_upper


# ============================================================
# INVENTORY SIMULATION
# ============================================================

def inventory_cost(actual, stocked):

    overstock = np.maximum(
        stocked - actual,
        0
    )

    stockout = np.maximum(
        actual - stocked,
        0
    )

    return (
        overstock * OVERSTOCK_COST
        + stockout * STOCKOUT_COST
    )


def evaluate_inventory(test, predictions, calibrated_upper):

    actual = test["sales"].to_numpy()

    median_stock = np.maximum(
        np.ceil(predictions[0.5]),
        0
    )

    upper_stock = np.maximum(
        np.ceil(calibrated_upper),
        0
    )

    median_cost = inventory_cost(
        actual,
        median_stock
    ).sum()

    upper_cost = inventory_cost(
        actual,
        upper_stock
    ).sum()

    savings = (
        (median_cost - upper_cost)
        / median_cost
    ) * 100

    return {
        "median_cost": median_cost,
        "upper_cost": upper_cost,
        "cost_reduction": savings,
    }


# ============================================================
# DEMAND SPIKE ANALYSIS
# ============================================================

def analyze_demand_spikes(
    train,
    test,
    calibrated_lower,
    calibrated_upper,
):

    threshold = np.quantile(
        train["sales"],
        0.90
    )

    actual = test["sales"].to_numpy()

    high_demand = actual >= threshold

    normal_demand = actual < threshold

    inside = (
        (actual >= calibrated_lower)
        & (actual <= calibrated_upper)
    )

    return {
        "high_demand_threshold": threshold,
        "high_demand_days": high_demand.sum(),
        "high_demand_coverage": inside[high_demand].mean(),
        "normal_demand_coverage": inside[normal_demand].mean(),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def save_forecast_plot(
    test,
    predictions,
    calibrated_lower,
    calibrated_upper,
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        test["date"],
        test["sales"],
        label="Actual Demand",
        alpha=0.6,
    )

    plt.plot(
        test["date"],
        predictions[0.5],
        label="Median Forecast",
    )

    plt.fill_between(
        test["date"],
        calibrated_lower,
        calibrated_upper,
        alpha=0.25,
        label="Calibrated 80% Prediction Interval",
    )

    plt.title("Demand Forecast with Prediction Intervals")

    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "forecast_intervals.png",
        dpi=300,
    )

    plt.close()


def save_inventory_plot(inventory_results):

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Median Forecast", "Calibrated Upper Bound"],
        [
            inventory_results["median_cost"],
            inventory_results["upper_cost"],
        ],
    )

    plt.title("Inventory Cost Comparison")

    plt.ylabel("Total Inventory Cost (₹)")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "inventory_comparison.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading dataset...")

    df = load_data()

    df = create_features(df)

    train, calibration, test = split_data(df)

    print("Training quantile models...")

    models = train_models(train)

    calibration_predictions = predict_quantiles(
        models,
        calibration
    )

    test_predictions = predict_quantiles(
        models,
        test
    )

    correction = calibrate_intervals(
        calibration,
        calibration_predictions
    )

    metrics, calibrated_lower, calibrated_upper = evaluate_model(
        test,
        test_predictions,
        correction
    )

    inventory_results = evaluate_inventory(
        test,
        test_predictions,
        calibrated_upper
    )

    spike_results = analyze_demand_spikes(
        train,
        test,
        calibrated_lower,
        calibrated_upper
    )
    # Save evaluation results

    all_results = {
        "forecasting": metrics,
        "inventory": inventory_results,
        "demand_spikes": spike_results,
    }

    # Convert NumPy values into standard Python numbers
    def convert_to_python(value):
        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        raise TypeError(
            f"Cannot serialize {type(value)}"
        )


    with open(
        RESULTS_DIR / "metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4,
            default=convert_to_python
        )
    # Save daily predictions

    daily_predictions = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual_sales": test["sales"].to_numpy(),
        "p10": test_predictions[0.1],
        "p50": test_predictions[0.5],
        "p90": test_predictions[0.9],
        "calibrated_lower": calibrated_lower,
        "calibrated_upper": calibrated_upper,
    })

    daily_predictions.to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False
)

    save_forecast_plot(
        test,
        test_predictions,
        calibrated_lower,
        calibrated_upper
    )

    save_inventory_plot(
        inventory_results
    )

    print("\n--- Forecasting Results ---")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Inventory Results ---")

    for name, value in inventory_results.items():
        print(f"{name}: {value:.4f}")

    print("\n--- Demand Spike Analysis ---")

    for name, value in spike_results.items():
        print(f"{name}: {value:.4f}")

    print("\nResults saved to the results/ folder.")


if __name__ == "__main__":
    main()   