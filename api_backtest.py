"""
api_backtest.py
---------------
Backtesting pipeline that uses the FastAPI model endpoint to generate
real-time trading signals for each step in the test period.

This script connects:
 - FastAPI model inference endpoint (/predict)
 - Local backtesting engine (backtesting.py)
"""

import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

from data import load_data, preprocess_data
from backtesting import backtest, performance_metrics


API_URL = "http://localhost:8899/predict"
TIMESTEPS = 30  # must match the CNN model window length
DATA_PATH = "data/prices.csv"


def get_signal_from_api(window_df: pd.DataFrame) -> int:
    """
    Send a recent window of OHLCV data to the API and return the trading signal.

    Args:
        window_df (pd.DataFrame): Window of latest candles (must include Open, High, Low, Close, Volume).

    Returns:
        int: Trading signal (-1, 0, 1)
    """
    payload = {
        "data": window_df.reset_index().to_dict(orient="records"),
        "timesteps": TIMESTEPS
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        return int(response.json()["prediction"])
    except Exception as e:
        print(f"⚠️ API call failed: {e}")
        return 0  # fallback to 'hold' if API unavailable


def run_api_backtest():
    """Run a full backtest using live predictions from the FastAPI model."""
    print("🚀 Starting API-driven backtest...\n")

    # 1️⃣ Load and preprocess data
    df = load_data(DATA_PATH)
    df = preprocess_data(df)

    # Ensure necessary columns exist
    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Data missing required columns: {required_cols - set(df.columns)}")

    # 2️⃣ Collect predictions by iterating through test period
    predictions = []
    price_series = df["Close"].iloc[TIMESTEPS:]  # align with prediction start

    print(f"Generating signals from API ({API_URL})...")
    for i in tqdm(range(TIMESTEPS, len(df))):
        window_df = df.iloc[i - TIMESTEPS:i].copy()
        signal = get_signal_from_api(window_df)
        predictions.append(signal)
        time.sleep(0.05)  # small delay to avoid API overload

    # 3️⃣ Align predictions and prices
    predictions = np.array(predictions)
    prices = price_series.values[: len(predictions)]

    # 4️⃣ Backtest using local engine
    print("\n📈 Running local backtesting simulation...")
    results = backtest(predictions, prices)
    metrics = performance_metrics(results["equity_curve"])

    # 5️⃣ Print summary
    print("\n===== API-Driven Backtest Results =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\n✅ Completed API-driven backtest.")
    return results, metrics


if __name__ == "__main__":
    run_api_backtest()
