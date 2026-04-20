#!/usr/bin/env python3
"""
MEMORY-OPTIMIZED SCIENTIFIC FEATURE RANKER (has_fault version)
Uses XGBoost + Permutation Importance with StratifiedKFold
Handles gapped labels, missing classes in folds
Target is fixed to: has_fault
"""

import argparse
import time
import numpy as np
import pandas as pd
import sys
import os
import gc
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

DEFAULT_INPUT = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/splits/Train_no_correlations.csv"
DEFAULT_OUTPUT = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/splits/Train_ranked_no_normalize.csv"

def main():
    parser = argparse.ArgumentParser(description="Memory-Optimized Feature Ranker with XGBoost (target = has_fault)")
    parser.add_argument("input", nargs='?', default=DEFAULT_INPUT, help="Input CSV")
    parser.add_argument("output", nargs='?', default=DEFAULT_OUTPUT, help="Output CSV")
    parser.add_argument("--top", type=int, default=50, help="Number of top features to keep")
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()

    print("="*70)
    print("MEMORY-OPTIMIZED SCIENTIFIC FEATURE RANKER")
    print(f"XGBoost v{xgb.__version__}")
    print(f"Input:  {args.input}")
    print("Target: has_fault (fixed)")
    print(f"Output: Top {args.top} features ranked")
    print("="*70)

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    df = pd.read_csv(args.input)
    label_col = "has_fault"           # ← fixed target

    if label_col not in df.columns:
        print(f"ERROR: Target column '{label_col}' not found in the dataset")
        sys.exit(1)

    # Identify feature columns (exclude all possible labels/groups)
    target_labels = ['has_fault', 'label', 'group0', 'group1', 'has_fault', 'group3']
    feature_names = [c for c in df.columns if c not in target_labels]
    X = df[feature_names].values
    y_raw = df[label_col].values

 


    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    print(f"Loaded {len(df)} samples, {len(feature_names)} features")
    print(f"Target: {label_col} → {n_classes} classes after encoding")
    print(f"Class mapping: {dict(enumerate(le.classes_))}")

    xgb_params = {
        'n_estimators': args.n_estimators,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'n_jobs': -1,
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss'
    }

    # ─── Ranking with StratifiedKFold + Permutation Importance ───
    print(f"\nRanking with {args.folds}-fold StratifiedKFold + Permutation Importance...")
    skf = StratifiedKFold(n_splits=args.folds)
    fold_importances = []

    for i, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"  Fold {i}/{args.folds}...", end='\r')

        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # Handle missing classes with dummy samples
        unique_y_tr = np.unique(y_tr)
        missing = [c for c in range(n_classes) if c not in unique_y_tr]
        if missing:
            dummy_X = np.tile(X_tr[0], (len(missing), 1))
            dummy_y = np.array(missing)
            dummy_w = np.zeros(len(missing))
            X_tr = np.vstack([X_tr, dummy_X])
            y_tr = np.concatenate([y_tr, dummy_y])
            w_tr = compute_sample_weight('balanced', y_tr)
            w_tr = np.concatenate([w_tr, dummy_w])
        else:
            w_tr = compute_sample_weight('balanced', y_tr)

        # Train
        model = XGBClassifier(**xgb_params, num_class=n_classes)
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)

        # Permutation importance
        res = permutation_importance(
            model, X_val, y_val,
            scoring='f1_macro',
            n_repeats=15,
            random_state=42,
            n_jobs=1
        )
        fold_importances.append(np.maximum(0, res.importances_mean))

        del model
        gc.collect()

    # Aggregate
    mean_imp = np.mean(fold_importances, axis=0)
    ranked_idx = np.argsort(mean_imp)[::-1]
    top_idx = ranked_idx[:args.top]

    ranked_features = [feature_names[i] for i in ranked_idx]
    top_features = [feature_names[i] for i in top_idx]

    print(f"\nTop {args.top} features by mean permutation importance (for has_fault):")
    for rank, fname in enumerate(top_features, 1):
        print(f"  {rank:2d}. {fname}")

    # Build output dataset (top features + label + groups)
    output_cols = top_features + [label_col]
    for c in target_labels:
        if c in df.columns and c != label_col:
            output_cols.append(c)

    df_output = df[output_cols].copy()
    df_output.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")
    print(f"  - Features kept: {len(top_features)} (top ranked)")
    print(f"  - Total columns: {len(output_cols)} (features + has_fault + other groups)")
    print("Done.")

if __name__ == "__main__":
    main()
