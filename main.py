"""
main.py
-------
Main execution pipeline for the Deep Learning Trading project.
Handles data loading, preprocessing, feature engineering, model training,
data drift detection, and backtesting of trading performance.
"""

import numpy as np
import pandas as pd
from data import load_data, preprocess_data, split_data
from indicators import add_technical_features, normalize_features
from trading_signal_labels import generate_trading_signals
from data_drift import detect_data_drift
from backtesting import backtest, performance_metrics
from cnn_model import create_cnn_model
from train import train_model
import tensorflow as tf


def reshape_for_cnn(X, timesteps=30):
    """
    Convert tabular features into sequential windows for CNN input.

    Args:
        X (pd.DataFrame or np.ndarray): Input features.
        timesteps (int): Number of lookback steps for each sample.

    Returns:
        np.ndarray: 3D array of shape (samples, timesteps, features)
    """
    X = np.array(X)
    samples = []
    for i in range(len(X) - timesteps):
        samples.append(X[i:i + timesteps])
    return np.array(samples)


def align_labels(y, timesteps=30):
    """Align labels to match the reduced sample size after windowing."""
    return y[timesteps:]


def main():
    print("🚀 Starting Deep Learning Trading Pipeline...\n")

    # 1️⃣ Load and preprocess data
    df = load_data("data/prices.csv")
    df = preprocess_data(df)
    df = add_technical_features(df)
    df = generate_trading_signals(df)
    df = normalize_features(df)

    print(f"Data loaded with {len(df)} records and {df.shape[1]} features.")

    # 2️⃣ Split dataset chronologically
    train_df, val_df, test_df = split_data(df)
    print(f"Train/Val/Test split: {len(train_df)}, {len(val_df)}, {len(test_df)}")

    from sklearn.preprocessing import LabelEncoder
    import joblib

    # 3️⃣ Separate features and labels
    X_train, y_train = train_df.drop("signal", axis=1), train_df["signal"]
    X_val, y_val = val_df.drop("signal", axis=1), val_df["signal"]
    X_test, y_test = test_df.drop("signal", axis=1), test_df["signal"]

    # 4️⃣ Encode labels (-1, 0, 1 → 0, 1, 2)
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_val = encoder.transform(y_val)
    y_test = encoder.transform(y_test)

    # Save encoder for inference use
    joblib.dump(encoder, "models/label_encoder.joblib")
    print("💾 Label encoder saved to models/label_encoder.joblib")

    # 5️⃣ Reshape features for CNN
    timesteps = 30
    X_train_seq = reshape_for_cnn(X_train, timesteps)
    X_val_seq = reshape_for_cnn(X_val, timesteps)
    X_test_seq = reshape_for_cnn(X_test, timesteps)

    # Align labels with windowed samples
    y_train_aligned = align_labels(y_train, timesteps)
    y_val_aligned = align_labels(y_val, timesteps)
    y_test_aligned = align_labels(y_test, timesteps)

    # 6️⃣ Ensure matching shapes
    assert X_train_seq.shape[0] == len(y_train_aligned), "Train X/y mismatch!"
    assert X_val_seq.shape[0] == len(y_val_aligned), "Val X/y mismatch!"
    assert X_test_seq.shape[0] == len(y_test_aligned), "Test X/y mismatch!"

    print("✅ Data shapes OK:")
    print(f"   Train: X={X_train_seq.shape}, y={y_train_aligned.shape}")
    print(f"   Val:   X={X_val_seq.shape}, y={y_val_aligned.shape}")
    print(f"   Test:  X={X_test_seq.shape}, y={y_test_aligned.shape}")

    # 3️⃣ Separate features and labels
    X_train, y_train = train_df.drop("signal", axis=1), train_df["signal"]
    X_val, y_val = val_df.drop("signal", axis=1), val_df["signal"]
    X_test, y_test = test_df.drop("signal", axis=1), test_df["signal"]

    # 4️⃣ Reshape features for CNN
    timesteps = 30
    X_train_seq = reshape_for_cnn(X_train, timesteps)
    X_val_seq = reshape_for_cnn(X_val, timesteps)
    X_test_seq = reshape_for_cnn(X_test, timesteps)

    y_train_aligned = align_labels(y_train.values, timesteps)
    y_val_aligned = align_labels(y_val.values, timesteps)
    y_val_aligned = encoder.transform(y_val_aligned)
    y_test_aligned = align_labels(y_test.values, timesteps)

    # 5️⃣ Create and train CNN model
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    num_classes = len(np.unique(y_train_aligned))
    model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
    model = train_model(model, X_train_seq, y_train_aligned, X_val_seq, y_val_aligned)

    # 6️⃣ Data drift detection (train vs test)
    print("\n📊 Running data drift detection...")
    detect_data_drift(train_df, test_df)

    # 7️⃣ Model evaluation and backtesting
    print("\n📈 Running backtesting...")
    preds = np.argmax(model.predict(X_test_seq), axis=1)
    preds = align_labels(preds, 0)  # same alignment logic
    backtest_results = backtest(preds, test_df["Close"].iloc[-len(preds):])

    metrics = performance_metrics(backtest_results["equity_curve"])
    print("\nFinal Backtest Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")

    print("\n✅ Pipeline complete. Model, metrics, and artifacts logged to MLFlow.")


if __name__ == "__main__":
    main()
