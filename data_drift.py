"""
data_drift.py
-------------
Performs statistical data drift detection between training and test datasets
using the Kolmogorov–Smirnov (KS) test and summary statistics.
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import os


def detect_data_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, threshold: float = 0.05, save_path: str = None):
    """
    Detect data drift between training and test datasets using the KS test.

    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        threshold (float): P-value threshold for drift detection.
        save_path (str): Optional path to save drift summary (CSV or JSON).

    Returns:
        pd.DataFrame: DataFrame with feature-wise drift statistics.
    """
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    drift_records = []

    for col in numeric_cols:
        train_col = train_df[col].dropna()
        test_col = test_df[col].dropna()

        if len(train_col) < 2 or len(test_col) < 2:
            continue  # skip insufficient data

        # KS test for distributional difference
        stat, pval = ks_2samp(train_col, test_col)

        drift_records.append({
            "feature": col,
            "train_mean": train_col.mean(),
            "test_mean": test_col.mean(),
            "mean_diff": train_col.mean() - test_col.mean(),
            "train_std": train_col.std(),
            "test_std": test_col.std(),
            "statistic": stat,
            "p_value": pval,
            "drift_detected": pval < threshold
        })

    drift_df = pd.DataFrame(drift_records).sort_values("p_value").reset_index(drop=True)

    # --- Console summary ---
    n_drifted = drift_df["drift_detected"].sum()
    print("\n===== DATA DRIFT REPORT =====")
    print(f"🔹 Features checked: {len(drift_df)}")
    print(f"⚠️ Drift detected in {n_drifted} features (p < {threshold})\n")

    if n_drifted > 0:
        print("Top drifted features:")
        print(drift_df[drift_df["drift_detected"]].head(5)[["feature", "p_value", "mean_diff"]].to_string(index=False))
    else:
        print("✅ No significant drift detected.")

    # --- Optional save ---
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".csv"):
            drift_df.to_csv(save_path, index=False)
        elif save_path.endswith(".json"):
            drift_df.to_json(save_path, orient="records", indent=2)
        print(f"\n💾 Drift report saved to: {save_path}")

    return drift_df
