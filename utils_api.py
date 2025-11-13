# utils_api.py
"""
Utility helpers for API preprocessing and model/scaler loading.
Relies on the same feature-engineering functions used during training.
"""

import os
from typing import Tuple, Optional, List, Dict

import joblib
import numpy as np
import pandas as pd

from indicators import add_technical_features, normalize_features


def load_scaler(scaler_path: str = "models/scaler.joblib"):
    """
    Attempt to load a saved sklearn scaler (joblib). If not found, return None.
    """
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception:
        return None


def load_label_mapping(mapping_path: str = "models/label_mapping.joblib"):
    """
    Attempt to load a saved label mapping (joblib). Expected format is a list
    or array mapping index -> label (e.g., [ -1, 0, 1 ]).
    Returns None if not available.
    """
    try:
        mapping = joblib.load(mapping_path)
        return mapping
    except Exception:
        return None


def df_from_pricepoints(data: List[dict]) -> pd.DataFrame:
    """
    Convert list of price dictionaries to a DataFrame and ensure correct dtypes.
    Expects each item to have keys: Date (optional), Open, High, Low, Close, Volume.
    """
    df = pd.DataFrame([d for d in data])
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
        except Exception:
            # ignore date parsing errors; keep integer index
            df.index = pd.RangeIndex(len(df))
    else:
        df.index = pd.RangeIndex(len(df))
    # Ensure numeric types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def prepare_features_for_inference(
    raw_df: pd.DataFrame,
    scaler=None,
    timesteps: int = 30,
    required_min_rows: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Process raw OHLCV DataFrame -> engineered features -> normalized -> CNN windows.

    Returns:
        X_windowed: numpy array shaped (1, timesteps, features)
        feature_cols: list of feature column names (ordered)
        feature_df: the full feature DataFrame after engineering & (optionally) scaling
    """
    if required_min_rows is None:
        required_min_rows = timesteps

    # 1) Feature engineering (use same function used in training)
    df = raw_df.copy()
    df = add_technical_features(df)  # this function drops NA and returns features including 'Close' etc.
    if len(df) < required_min_rows:
        raise ValueError(f"Not enough rows after feature engineering: got {len(df)}, need {required_min_rows}")

    # Drop target columns if present
    if "signal" in df.columns:
        df = df.drop(columns=["signal"])

    # 2) Select only the last `required_min_rows` rows for windowing
    df_window = df.iloc[-required_min_rows:].copy()

    # 3) Normalize using provided scaler if available
    feature_cols = df_window.columns.tolist()
    if scaler is not None:
        # Expect scaler.transform accepts 2D array in the order of feature_cols
        scaled = scaler.transform(df_window.values)
        df_scaled = pd.DataFrame(scaled, index=df_window.index, columns=feature_cols)
    else:
        # Fall back to fitting a MinMax scaler via normalize_features (this will fit on the window)
        df_scaled = normalize_features(df_window)

    # 4) Reshape to (1, timesteps, features)
    X = df_scaled.values.astype(np.float32)
    X = X.reshape(1, X.shape[0], X.shape[1])  # one sample, timesteps, features

    return X, feature_cols, df_scaled


def map_prediction_index_to_label(index: int, label_mapping: Optional[List[int]] = None) -> int:
    """
    Map a predicted class index (0..K-1) to the original label used for trading signals.
    If a label_mapping is provided, it should be index -> label (e.g., [ -1, 0, 1 ]).
    If not provided, a default mapping is returned:
        - For 3 classes -> [-1, 0, 1]
        - For 2 classes -> [0, 1] (returns 0 or 1)
        - For 1 class -> [0]
    """
    if label_mapping is not None:
        try:
            return int(label_mapping[int(index)])
        except Exception:
            pass

    # Default heuristics
    # common trading labels: -1, 0, 1
    if index is None:
        return 0
    return int({0: -1, 1: 0, 2: 1}.get(index, int(index)))