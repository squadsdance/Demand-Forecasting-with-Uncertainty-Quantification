from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/train.csv")
RESULTS_DIR = Path("results")

STORE_ID = 1
ITEM_ID = 1


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    print("\nMissing values:")
    print(df.isnull().sum())

    product = df[
        (df["store"] == STORE_ID)
        & (df["item"] == ITEM_ID)
    ].copy()

    if product.empty:
        raise ValueError(
            "No data found for the selected store and item."
        )

    product["date"] = pd.to_datetime(product["date"])

    product = product.sort_values("date")

    # Historical demand plot

    plt.figure(figsize=(12, 5))

    plt.plot(
        product["date"],
        product["sales"],
        label="Daily Sales"
    )

    plt.title("Historical Demand: Store 1, Item 1")

    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "historical_demand.png",
        dpi=300
    )

    plt.close()

    # Calculate the 30-day rolling average

    product["rolling_30"] = (
        product["sales"]
        .rolling(window=30)
        .mean()
    )

    # Rolling average visualization

    plt.figure(figsize=(12, 5))

    plt.plot(
        product["date"],
        product["sales"],
        alpha=0.3,
        label="Daily Sales"
    )

    plt.plot(
        product["date"],
        product["rolling_30"],
        label="30-Day Rolling Average"
    )

    plt.title("Daily Demand vs. 30-Day Rolling Average")

    plt.xlabel("Date")
    plt.ylabel("Units Sold")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "rolling_average.png",
        dpi=300
    )

    plt.close()

    print("\nExploratory analysis completed.")
    print("Graphs saved to the results/ folder.")


if __name__ == "__main__":
    main()