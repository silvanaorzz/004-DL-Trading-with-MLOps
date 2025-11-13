"""
data.py
-------
Handles loading, cleaning, preprocessing, and splitting of financial price data
from raw CSVs (Investing.com / Yahoo style) for the Deep Learning Trading pipeline.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import re


def _parse_volume(vol_str: str) -> float:
    """
    Convert volume strings like '36.48M' or '1.2K' into numeric values.
    Supports suffixes K, M, B.
    """
    if pd.isna(vol_str):
        return np.nan
    vol_str = str(vol_str).replace(",", "").strip()
    match = re.match(r"([\d\.]+)([KMB]?)", vol_str)
    if not match:
        return np.nan
    value, suffix = match.groups()
    multiplier = {"K": 1e3, "M": 1e6, "B": 1e9}.get(suffix, 1)
    return float(value) * multiplier


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw market CSV and standardize column names.

    Handles columns like:
        'Date', 'Price', 'Open', 'High', 'Low', 'Vol.', 'Change %'

    Returns:
        pd.DataFrame: Cleaned DataFrame with ['Open','High','Low','Close','Volume']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ Data file not found: {filepath}")

    # Read CSV (allow quoted headers)
    df = pd.read_csv(filepath)

    # Normalize column names
    df.columns = [c.strip().replace('"', '').replace(" ", "_") for c in df.columns]

    # Map expected names
    rename_map = {
        "Price": "Close",
        "Vol.": "Volume",
        "Change_%": "ChangePct",
    }
    df.rename(columns=rename_map, inplace=True)

    # Parse Date column
    if "Date" not in df.columns:
        raise ValueError("❌ Missing 'Date' column in data file.")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df.set_index("Date", inplace=True)

    # Parse Volume strings (e.g. '36.48M')
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].apply(_parse_volume)

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop irrelevant or non-numeric columns
    drop_cols = [c for c in df.columns if not any(k in c for k in ["Open", "High", "Low", "Close", "Volume"])]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Sort by date ascending
    df.sort_index(inplace=True)

    print(f"✅ Loaded market data: {df.shape[0]} rows | {df.shape[1]} columns")
    print(f"   Range: {df.index.min().date()} → {df.index.max().date()}")
    return df


def preprocess_data(df: pd.DataFrame, normalize: bool = False, scaler_path: str = None) -> pd.DataFrame:
    """
    Fill missing values and optionally normalize features.

    Args:
        df (pd.DataFrame): Raw data.
        normalize (bool): Apply MinMax normalization.
        scaler_path (str): Optional path to save fitted scaler.

    Returns:
        pd.DataFrame: Cleaned (and optionally normalized) DataFrame.
    """
    df = df.copy()
    df = df.ffill().bfill().dropna()

    if normalize:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values)
        df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
            print(f"💾 Scaler saved to {scaler_path}")

    print(f"✅ Preprocessed data: {len(df)} samples after cleaning.")
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    shuffle: bool = False,
):
    """
    Chronologically (or randomly) split the data into train/val/test sets.

    Returns:
        (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    df = df.copy()
    if shuffle:
        df = df.sample(frac=1, random_state=42)

    n = len(df)
    train_end = int(train_ratio * n)
    val_end = train_end + int(val_ratio * n)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print(f"✅ Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df
