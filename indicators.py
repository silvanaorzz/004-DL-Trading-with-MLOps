"""
indicators.py
-------------
Feature engineering module for financial time-series data.
Uses the `ta` library for reliable and efficient technical indicators.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ta  # Technical Analysis library


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a comprehensive set of technical indicators to the DataFrame.

    Args:
        df (pd.DataFrame): Must contain 'Open', 'High', 'Low', 'Close', 'Volume'.

    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """
    df = df.copy()

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ====== Basic Features ======
    df["returns"] = df["Close"].pct_change()
    df["log_return"] = np.log1p(df["returns"])

    # ====== Momentum Indicators ======
    df["rsi_14"] = ta.momentum.RSIIndicator(close=df["Close"], window=14).rsi()
    df["stoch_k"] = ta.momentum.StochasticOscillator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).stoch()
    df["roc_10"] = ta.momentum.ROCIndicator(close=df["Close"], window=10).roc()
    df["willr_14"] = ta.momentum.WilliamsRIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], lbp=14
    ).williams_r()

    # ====== Trend Indicators ======
    df["ema_12"] = ta.trend.EMAIndicator(close=df["Close"], window=12).ema_indicator()
    df["ema_26"] = ta.trend.EMAIndicator(close=df["Close"], window=26).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(close=df["Close"], window=50).ema_indicator()
    macd = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["adx_14"] = ta.trend.ADXIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).adx()
    df["cci_20"] = ta.trend.CCIIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], window=20
    ).cci()

    # ====== Volatility Indicators ======
    df["atr_14"] = ta.volatility.AverageTrueRange(
        high=df["High"], low=df["Low"], close=df["Close"], window=14
    ).average_true_range()
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["Close"]
    df["bb_percent_b"] = bb.bollinger_pband()

    # ====== Volume Indicators ======
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
    df["cmf_20"] = ta.volume.ChaikinMoneyFlowIndicator(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=20
    ).chaikin_money_flow()
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"], window=20
    ).volume_weighted_average_price()

    # ====== Statistical Features ======
    df["rolling_mean_10"] = df["Close"].rolling(10).mean()
    df["rolling_std_10"] = df["Close"].rolling(10).std()
    df["rolling_mean_30"] = df["Close"].rolling(30).mean()
    df["rolling_std_30"] = df["Close"].rolling(30).std()

    # ====== Cleanup ======
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df


def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize features to [0, 1] range using MinMaxScaler.

    Args:
        df (pd.DataFrame): Feature DataFrame.

    Returns:
        pd.DataFrame: Normalized DataFrame.
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(scaled, index=df.index, columns=df.columns)
    return df_scaled
