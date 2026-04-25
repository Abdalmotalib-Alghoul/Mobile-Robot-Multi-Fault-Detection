#!/usr/bin/env python3
"""
Remove one feature from each high-correlation pair (|r| >= 0.95)

"""

import os
import pandas as pd
import numpy as np

# ───────────────────────────────────────────────
#  CONFIG
# ───────────────────────────────────────────────
INPUT_FILE  = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/splits/train.csv"
TARGET_COL  = "has_fault"
CORR_THRESHOLD = 0.95

OUTPUT_DIR  = "splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Train_no_correlations.csv")
REMOVED_LOG = os.path.join(OUTPUT_DIR, "removed_features_log.txt")

# ───────────────────────────────────────────────
def main():
    print("Loading dataset ...")
    df = pd.read_csv(INPUT_FILE)
    print(f"Original shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    if TARGET_COL not in df.columns:
        print(f"Error: target column '{TARGET_COL}' not found.")
        return


    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # Only numeric features for correlation
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) != X.shape[1]:
        print(f"Warning: {X.shape[1] - len(numeric_cols)} non-numeric columns ignored for correlation")

    X_num = X[numeric_cols].copy()

    print("Computing absolute Pearson correlation matrix ...")
    corr_abs = X_num.corr(method="pearson").abs()

    # Upper triangle → find high corr pairs
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    high_pairs = upper.stack().reset_index()
    high_pairs.columns = ["f1", "f2", "abs_r"]
    high_pairs = high_pairs[high_pairs["abs_r"] >= CORR_THRESHOLD]
    high_pairs = high_pairs.sort_values("abs_r", ascending=False).reset_index(drop=True)

    print(f"Found {len(high_pairs)} pairs with |r| ≥ {CORR_THRESHOLD}\n")

    if len(high_pairs) == 0:
        print("No strong correlations → no removal needed.")
        return

    # ─── removal ───────────────────────────────────────
    to_remove = set()
    kept = set(numeric_cols)

    for _, row in high_pairs.iterrows():
        f1, f2 = row["f1"], row["f2"]


        if f1 in to_remove or f2 in to_remove:
            continue

        # Deterministic choice
        if f1 < f2:
            remove = f1
            keep  = f2
        else:
            remove = f2
            keep  = f1

        to_remove.add(remove)
        kept.discard(remove)

        print(f"Pair {len(to_remove):2d}: |r|={row['abs_r']:.4f} → remove {remove:<36}  keep {keep}")

    print(f"\nTotal features to remove: {len(to_remove)}")
    print(f"Remaining numeric features: {len(kept)}")

    # Build final dataset
    final_cols = list(kept) + [TARGET_COL]
    df_clean = df[final_cols].copy()

    print(f"\nSaving cleaned dataset → {OUTPUT_FILE}")
    df_clean.to_csv(OUTPUT_FILE, index=False)

    # Log removed features
    with open(REMOVED_LOG, "w") as f:
        f.write("Removed features due to |r| >= 0.85 (greedy alphabetical keep):\n\n")
        for feat in sorted(to_remove):
            f.write(f"{feat}\n")
        f.write(f"\nTotal removed: {len(to_remove)}\n")
        f.write(f"Remaining features (excluding target): {len(kept)}\n")

    print(f"Removal log saved → {REMOVED_LOG}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
