"""
backtesting.py
---------------
Simulates a simple long/short trading strategy using model predictions
and computes risk-adjusted performance metrics.
"""

import pandas as pd
import numpy as np


def backtest(
    predictions,
    prices,
    commission: float = 0.00125,
    trade_on_change: bool = True,
):
    """
    Run a simple backtest given model predictions and asset prices.

    Args:
        predictions (array-like): Trading signals (-1 = short, 0 = hold, 1 = long).
        prices (array-like): Corresponding price series (must align in time).
        commission (float): Transaction cost per trade (fractional, e.g. 0.00125 = 0.125%).
        trade_on_change (bool): Apply commission only when position changes.

    Returns:
        pd.DataFrame: Backtest results including equity curve and strategy returns.
    """
    df = pd.DataFrame({"signal": predictions, "price": prices}).copy()

    # --- Compute raw returns ---
    df["return"] = df["price"].pct_change().fillna(0)

    # --- Shift signals to avoid lookahead bias ---
    df["position"] = df["signal"].shift(1).fillna(0)

    # --- Strategy return before commission ---
    df["strategy_return"] = df["position"] * df["return"]

    # --- Apply transaction costs ---
    if trade_on_change:
        df["trade"] = df["signal"].diff().abs().fillna(0)
        df["strategy_return"] -= df["trade"] * commission
    else:
        df["strategy_return"] -= abs(df["position"]) * commission

    # --- Compute equity curve ---
    df["equity_curve"] = (1 + df["strategy_return"]).cumprod()
    df["benchmark_curve"] = (1 + df["return"]).cumprod()

    return df


def performance_metrics(equity_curve: pd.Series, freq: int = 252) -> dict:
    """
    Compute key performance metrics for a trading strategy.

    Args:
        equity_curve (pd.Series): Equity curve (cumulative returns).
        freq (int): Annualization factor (default 252 for daily data).

    Returns:
        dict: Sharpe, Sortino, Calmar ratios and Max Drawdown.
    """
    returns = equity_curve.pct_change().dropna()

    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    downside_std = np.std(returns[returns < 0])
    max_dd = (equity_curve / equity_curve.cummax() - 1).min()

    sharpe = (mean_ret / std_ret) * np.sqrt(freq) if std_ret > 0 else 0
    sortino = (mean_ret / downside_std) * np.sqrt(freq) if downside_std > 0 else 0
    calmar = (mean_ret * freq / abs(max_dd)) if max_dd != 0 else 0

    metrics = {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max_Drawdown": max_dd,
        "Total_Return": equity_curve.iloc[-1] - 1,
    }
    return metrics


def summarize_backtest(df: pd.DataFrame):
    """
    Print a quick summary of backtest results and performance metrics.

    Args:
        df (pd.DataFrame): Output from `backtest()`.

    Returns:
        dict: Performance metrics.
    """
    metrics = performance_metrics(df["equity_curve"])
    print("\n===== BACKTEST SUMMARY =====")
    for k, v in metrics.items():
        print(f"{k:15}: {v: .4f}")
    print(f"Final Equity: {df['equity_curve'].iloc[-1]:.4f}")
    return metrics


def compute_drawdown_series(equity_curve: pd.Series) -> pd.Series:
    """
    Compute drawdown series from equity curve.

    Args:
        equity_curve (pd.Series): Cumulative returns.

    Returns:
        pd.Series: Drawdown values over time.
    """
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    return drawdown
