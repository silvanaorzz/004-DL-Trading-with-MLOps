# Compro vendo o nada. Long (1), hold (0), short (-1)

import pandas as pd


def generate_trading_signals(df: pd.DataFrame, threshold: float = 0.002, horizon: int = 1) -> pd.DataFrame:
    df['future_return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['signal'] = 0
    df.loc[df['future_return'] > threshold, 'signal'] = 1
    df.loc[df['future_return'] < -threshold, 'signal'] = -1
    df.dropna(inplace=True)
    return df
