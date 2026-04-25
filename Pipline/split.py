#!/usr/bin/env python3

import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

# ========================= CONFIG =========================
INPUT_PATH = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/causal_60_step30_all_features_combined_final.csv"
OUTPUT_DIR = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/splits/"
TRAIN_PATH = os.path.join(OUTPUT_DIR, "train.csv")
TEST_PATH  = os.path.join(OUTPUT_DIR, "test.csv")
TARGET_COL = "has_fault"
TEST_SIZE  = 0.20
RANDOM_STATE = 42
# ========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load data
logger.info("Loading dataset...")
df = pd.read_csv(INPUT_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Column '{TARGET_COL}' not found in the dataset!")

logger.info(f"Performing stratified random split by '{TARGET_COL}' (top-tier practice for multi-path data)...")
logger.info(f"→ Train: 80% | Test: 20% | Seed={RANDOM_STATE} → paths mixed fairly")

# Stratified split (sklearn handles per-class stratification perfectly)
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    stratify=df[TARGET_COL],
    random_state=RANDOM_STATE
)

# Save
train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH,  index=False)

# Summary
logger.info("="*70)
logger.info("SPLIT COMPLETED SUCCESSFULLY — STRATIFIED RANDOM (PATH-FAIR)")
logger.info(f"Total samples     : {len(df)}")
logger.info(f"Train samples     : {len(train_df)} ({len(train_df)/len(df):.1%})")
logger.info(f"Test samples      : {len(test_df)}  ({len(test_df)/len(df):.1%})")
logger.info(f"Train saved to    : {TRAIN_PATH}")
logger.info(f"Test saved to     : {TEST_PATH}")
logger.info("="*70)

# Class distribution check
print("\nClass distribution in Train:")
print(train_df[TARGET_COL].value_counts().sort_index())
print("\nClass distribution in Test:")
print(test_df[TARGET_COL].value_counts().sort_index())
