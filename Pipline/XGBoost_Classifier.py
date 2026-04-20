#!/usr/bin/env python3
"""
Unified XGBoost Training Script v8 - API COMPATIBILITY FIXES
============================================================================
For CoDIT 2026

FIXES IN v8:
-----------
1. FIXED SHAP SAMPLE SIZE: Ensures minimum interpretability sample size.
   When SHAP uses all available samples (small dataset), interpretability
   functions still work with available data.

2. FIXED PDP MULTICLASS: Added `target` parameter for multiclass classification.
   PDP now correctly computes partial dependence for each class separately.

3. FIXED LIME API: `as_pyplot_figure()` no longer accepts `ax` parameter.
   Now creates figure separately and optionally adds to existing axes.

4. FIXED SHAP WATERFALL: Modern SHAP API changed. Now uses `shap.Explanation`
   object instead of passing `feature_names` directly to `waterfall()`.

5. IMPROVED ERROR HANDLING: All plotting functions have better error handling
   with specific warnings for API incompatibilities.

KEY IMPROVEMENTS (from v7):
---------------------------
- Stabilized sample matching for interpretability
- Improved plot_local_shap_per_class with pre-computed SHAP values
- Robust fallback when SHAP is disabled
- Cleaner code with clear console output

CLASS WEIGHT TUNING:
-------------------
NORMAL_WEIGHT = 1.7 (default) - Adjust between 1.0-2.5 based on validation performance
"""

import numpy as np
import pandas as pd
import time
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import seaborn as sns
import shap
import random
import optuna
import lime
import lime.lime_tabular
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Reproducibility
np.random.seed(7)
random.seed(7)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# CONFIGURATION
# =====================================================================
DEFAULT_TRAIN = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/splits/Train_ranked_no_normalize.csv"
DEFAULT_TEST = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/splits/test.csv"

N_FOLDS = 5
N_OPTUNA_TRIALS = 40  # 40 trials per fold for thorough HPO
EARLY_STOPPING_ROUNDS = 30
FINAL_VAL_SIZE = 0.10

# =====================================================================
# CLASS WEIGHT CONFIGURATION - CUSTOM WEIGHTS FOR CLASS 0 (NORMAL)
# =====================================================================
# NOTE: This weight can be tuned between 1.0-2.5 based on validation performance
# Higher values (>1.0) improve recall for Normal class at cost of other classes
NORMAL_WEIGHT = 2.0

# =====================================================================
# SHAP & INTERPRETABILITY CONFIGURATION
# =====================================================================
RUN_SHAP_ANALYSIS = True  # Set to False to skip SHAP analysis (faster training)
INTERPRETABILITY_SAMPLE_SIZE = 1500  # Max samples for interpretability plots

# =====================================================================
# CLASS MAPPING
# =====================================================================
CLASS_MAPPING = {
    0: 'Normal',
    2: 'Dropout',
    3: 'G_Noise',
    4: 'Lidar_Drift',
    5: 'Lidar_Offset',
    7: 'ACC_Fault',
    9: 'ANG_Fault',
    10: 'Beam_loss'
}

# =====================================================================
# LABEL PROCESSING
# =====================================================================
def process_labels(y_raw):
    """Process raw labels to map to consecutive indices for classification."""
    unique_in_data = np.unique(y_raw)
    present_classes = [l for l in unique_in_data if l in CLASS_MAPPING]
    if not present_classes:
        raise ValueError("No valid fault classes found for classification!")

    valid_mask = np.isin(y_raw, list(CLASS_MAPPING.keys()))
    y_filtered = y_raw[valid_mask]

    all_possible_names = []
    for l in sorted(CLASS_MAPPING.keys()):
        name = CLASS_MAPPING[l]
        if name not in all_possible_names:
            all_possible_names.append(name)

    present_names = [name for name in all_possible_names
                     if any(CLASS_MAPPING[l] == name for l in unique_in_data if l in CLASS_MAPPING)]

    name_to_idx = {name: i for i, name in enumerate(present_names)}
    y_mapped = np.array([name_to_idx[CLASS_MAPPING[l]] for l in y_filtered])

    print(f"\n[Classification Mode] Classes: {present_names}")
    return y_mapped, present_names, valid_mask, present_classes

# =====================================================================
# CUSTOM CLASS WEIGHT FUNCTION
# =====================================================================
def get_custom_sample_weights(y, normal_weight=NORMAL_WEIGHT):
    """Compute sample weights with balanced base + tunable Class 0 (Normal) multiplier."""
    base_weights = compute_sample_weight(class_weight='balanced', y=y)
    class_0_mask = (y == 0)
    final_weights = base_weights.copy()
    final_weights[class_0_mask] = base_weights[class_0_mask] * normal_weight
    return final_weights

# =====================================================================
# ANALYSIS FUNCTIONS
# =====================================================================
def run_shap_analysis(model, X_scaled, y_scaled, feature_names, prefix, n_samples=1000):
    """
    Compute SHAP values using modern SHAP API.

    Returns:
        shap_values: SHAP values (list or 3D array depending on model)
        X_sample: Sampled features used for SHAP computation
        y_sample: Corresponding labels for the sampled features
        sample_indices: Original indices of sampled data
    """
        # ====================== STRATIFIED SAMPLING (MATLAB-style) ======================
    if len(X_scaled) > n_samples:
        print(f"Selecting {n_samples} samples using **stratified sampling** "
              f"(preserves class distribution)...")
        
        from sklearn.model_selection import train_test_split
        
        _, sample_indices = train_test_split(
            np.arange(len(X_scaled)), 
            test_size=n_samples / len(X_scaled),
            stratify=y_scaled,
            random_state=7
        )
        X_sample = X_scaled[sample_indices]
        y_sample = y_scaled[sample_indices]
    else:
        print(f"Using ALL {len(X_scaled)} available samples (dataset is small)")
        X_sample = X_scaled
        y_sample = y_scaled
        sample_indices = np.arange(len(X_scaled))













    print(f"Computing SHAP values on {len(X_sample)} samples...")

    try:
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation='interventional',
            data=X_sample
        )
        explanation = explainer(X_sample, check_additivity=False)
        shap_values = explanation.values

    except Exception as e:
        print(f"  SHAP warning (modern API): {e}")
        print("  Falling back to legacy shap_values() method...")

        try:
            shap_values = explainer.shap_values(X_sample, check_additivity=False)
        except Exception as e2:
            print(f"  SHAP warning (legacy): {e2}")
            print("  Using basic TreeExplainer without feature perturbation...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample, check_additivity=False)

    # Calculate shap_importance for global ranking
    if isinstance(shap_values, list):
        shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        shap_for_plot = shap_values
    else:
        if len(shap_values.shape) == 3:
            shap_importance = np.abs(shap_values).mean(axis=(0, 2))
            shap_for_plot = shap_values.mean(axis=2)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
            shap_for_plot = shap_values

    # Save SHAP feature importance CSV
    shap_ranking = sorted(zip(feature_names, shap_importance), key=lambda x: -x[1])
    pd.DataFrame(shap_ranking, columns=['feature', 'shap_importance']).to_csv(
        f"shap_feature_importance_{prefix}.csv", index=False)

    # Plot SHAP summary
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_for_plot, X_sample, feature_names=feature_names,
                      show=False, max_display=5)
    plt.savefig(f"shap_summary_{prefix}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP analysis completed (saved shap_summary_{prefix}.png)")

    return shap_values, X_sample, y_sample, sample_indices

def calculate_metrics(y_true, y_pred, prefix):
    """Calculate comprehensive classification metrics."""
    accuracy = accuracy_score(y_true, y_pred) * 100
    error_rate = (1 - accuracy_score(y_true, y_pred)) * 100
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0) * 100
    weighted_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0) * 100
    weighted_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0) * 100
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) * 100

    return {
        f'Accuracy % ({prefix})': accuracy,
        f'Total Cost ({prefix})': len(y_true) - np.sum(y_true == y_pred),
        f'Error Rate % ({prefix})': error_rate,
        f'Macro Precision % ({prefix})': macro_precision,
        f'Micro Precision % ({prefix})': micro_precision,
        f'Weighted Precision % ({prefix})': weighted_precision,
        f'Macro Recall % ({prefix})': macro_recall,
        f'Micro Recall % ({prefix})': micro_recall,
        f'Weighted Recall % ({prefix})': weighted_recall,
        f'Macro F1 Score % ({prefix})': macro_f1,
        f'Micro F1 Score % ({prefix})': micro_f1,
        f'Weighted F1 Score % ({prefix})': weighted_f1
    }

# =====================================================================
# CONFERENCE FIGURE PLOTTING FUNCTIONS
# =====================================================================
def plot_confusion_matrix(y_true, y_pred, class_names, title, filename, normalize=False):
    """Plot confusion matrix heatmap for conference paper."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'},
                linewidths=0.5, linecolor='gray')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_roc_curves(y_true, y_pred_proba, class_names, title, filename):
    """Plot ROC curves (one-vs-all) for conference paper."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})', linewidth=2)

    # Micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, 'k--', linewidth=2,
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    return roc_auc

def plot_precision_recall_curves(y_true, y_pred_proba, class_names, title, filename):
    """Plot Precision-Recall curves for conference paper."""
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 8))

    precision = {}
    recall = {}
    avg_precision = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        ax.plot(recall[i], precision[i], linewidth=2,
                label=f'{class_names[i]} (AP = {avg_precision[i]:.3f})')

    # Micro-average PR curve
    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    avg_precision_micro = average_precision_score(y_true_bin.ravel(), y_pred_proba.ravel())
    ax.plot(recall_micro, precision_micro, 'k--', linewidth=2,
            label=f'Micro-average (AP = {avg_precision_micro:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    return avg_precision

def plot_class_distribution(y, class_names, title, filename):
    """Plot class distribution bar chart for conference paper."""
    unique, counts = np.unique(y, return_counts=True)
    counts_dict = {class_names[i]: counts[i] for i in unique}

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(counts_dict.keys(), counts_dict.values(),
                  color=sns.color_palette("husl", len(counts_dict)))

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    ax.set_xlabel('Fault Type', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

def plot_feature_importance(model, feature_names, title, filename, top_n=5):
    """Plot XGBoost feature importance for conference paper."""
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance_df['feature'], importance_df['importance'],
                   color=sns.color_palette("Blues_d", len(importance_df)))
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    return importance_df

# =====================================================================
# INTERPRETABILITY PLOTTING FUNCTIONS - FIXED FOR API COMPATIBILITY
# =====================================================================

def plot_permutation_importance(model, X, y, feature_names, title, filename, top_n=5, n_repeats=10):
    """
    Plot Permutation Importance - Bar plot showing the relative importance of each predictor
    by measuring how much the model's performance drops when that predictor's values are
    randomly permuted.
    """
    print(f"  Computing Permutation Importance on {len(X)} samples...")
    try:
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=7, n_jobs=-1
        )

        # Sort by mean importance
        sorted_idx = perm_importance.importances_mean.argsort()[::-1][:top_n]

        importance_df = pd.DataFrame({
            'feature': [feature_names[i] for i in sorted_idx],
            'importance_mean': perm_importance.importances_mean[sorted_idx],
            'importance_std': perm_importance.importances_std[sorted_idx]
        })

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            importance_df['feature'],
            importance_df['importance_mean'],
            xerr=importance_df['importance_std'],
            color=sns.color_palette("RdYlGn_r", len(importance_df)),
            capsize=3
        )
        ax.invert_yaxis()
        ax.set_xlabel('Permutation Importance (Mean Decrease in F1)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
        return importance_df
    except Exception as e:
        print(f"  Warning: Could not compute permutation importance: {e}")
        return None

def plot_partial_dependence(model, X, feature_names, title, filename, class_idx=0, top_features=None):
    """
    FIXED v8: Plot Partial Dependence Plot (PDP) for multiclass.
    
    For multiclass, computes PDP for a specific class (default: class_idx=0).
    Uses `target` parameter for sklearn's PartialDependenceDisplay.
    """
    print(f"  Computing Partial Dependence Plots on {len(X)} samples (class {class_idx})...")

    try:
        if top_features is None:
            importance = model.feature_importances_
            top_indices = importance.argsort()[-5:][::-1]
        else:
            top_indices = [i for i, fn in enumerate(feature_names) if fn in top_features][:5]

        n_features = len(top_indices)
        n_rows = int(np.ceil(n_features / 2))

        # Individual PDPs (1-way)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, feat_idx in enumerate(top_indices):
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X, features=[feat_idx],
                    feature_names=feature_names,
                    ax=axes[idx],
                    kind='average',
                    target=class_idx,  # FIXED: Added target parameter for multiclass
                    line_kw={'color': 'blue', 'linewidth': 2}
                )
                axes[idx].set_title(f'PDP: {feature_names[feat_idx]} (Class {class_idx})', 
                                   fontsize=12, fontweight='bold')
            except Exception as e:
                print(f"    Warning: Could not plot PDP for {feature_names[feat_idx]}: {e}")
                axes[idx].text(0.5, 0.5, f'PDP for {feature_names[feat_idx]}\n unavailable',
                              ha='center', va='center', transform=axes[idx].transAxes)

        for idx in range(len(top_indices), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{title} (Class {class_idx})', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

        # 2-way PDP (interaction plot) for top 2 features
        if len(top_indices) >= 2:
            fig2, ax2 = plt.subplots(figsize=(12, 10))
            try:
                PartialDependenceDisplay.from_estimator(
                    model, X, features=[(top_indices[0], top_indices[1])],
                    feature_names=feature_names,
                    ax=ax2,
                    kind='average',
                    target=class_idx  # FIXED: Added target parameter for multiclass
                )
                ax2.set_title(f'2-Way PDP: {feature_names[top_indices[0]]} x {feature_names[top_indices[1]]} (Class {class_idx})',
                             fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(filename.replace('.png', '_interaction.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {filename.replace('.png', '_interaction.png')}")
            except Exception as e:
                print(f"    Warning: Could not plot 2-way PDP: {e}")
    except Exception as e:
        print(f"  Warning: Could not compute partial dependence: {e}")




def plot_shap_importance(shap_values, X_sample, feature_names, title, filename, class_names, top_n=10):
    """UPDATED: MATLAB-style stacked bar chart with DISTINCT, clear colors."""
    try:
        print(f" → Creating MATLAB-style STACKED SHAP importance plot with distinct colors...")

        # Compute mean absolute SHAP per feature per class
        if isinstance(shap_values, list):
            mean_abs_per_class = np.array([np.abs(sv).mean(axis=0) for sv in shap_values]).T
        elif len(shap_values.shape) == 3:
            mean_abs_per_class = np.abs(shap_values).mean(axis=0)
        else:
            mean_abs_per_class = np.abs(shap_values).mean(axis=0)[:, np.newaxis]

        # Total importance for sorting
        total_importance = mean_abs_per_class.sum(axis=1)

        # Top N features
        sorted_idx = np.argsort(total_importance)[::-1][:top_n]
        top_features = [feature_names[i] for i in sorted_idx]
        top_per_class = mean_abs_per_class[sorted_idx]

        # === DISTINCT & CLEAR COLORS (much better than husl) ===
        distinct_colors = [
            '#e41a1c',  # Red       - Normal
            '#377eb8',  # Blue      - Dropout
            '#4daf4a',  # Green     - G_Noise
            '#984ea3',  # Purple    - Lidar_Drift
            '#ff9f1c',  # Orange    - Lidar_Offset
            '#ffff33',  # Yellow    - ACC_Fault
            '#a65628',  # Brown     - ANG_Fault
            '#f781bf'   # Pink      - Beam_loss
        ]

        # Use only as many colors as we have classes
        colors = distinct_colors[:len(class_names)]

        # === MATLAB-style stacked horizontal bar chart ===
        fig, ax = plt.subplots(figsize=(14, max(9, top_n * 0.75)))

        left = np.zeros(len(top_features))
        for c in range(len(class_names)):
            ax.barh(top_features, top_per_class[:, c], left=left,
                    color=colors[c], 
                    label=class_names[c],
                    edgecolor='white', 
                    linewidth=0.8,
                    height=0.82)

            left += top_per_class[:, c]

        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value| (stacked by class contribution)', fontsize=13)
        ax.set_ylabel('Feature', fontsize=13)
        ax.set_title(title + '\n(MATLAB-style: Per-Class Contribution)', 
                     fontsize=15, fontweight='bold', pad=25)

        # Legend with better visibility
        ax.legend(title='Class', loc='lower right', fontsize=10.5, 
                  frameon=True, ncol=2 if len(class_names) > 6 else 1)

        plt.tight_layout()

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f" Saved: {filename} ← STACKED with DISTINCT colors")

        # Save per-class CSV (very useful)
        per_class_df = pd.DataFrame(top_per_class,
                                    columns=[f"{name}" for name in class_names],
                                    index=top_features)
        per_class_df['Total'] = per_class_df.sum(axis=1)
        per_class_df = per_class_df.sort_values('Total', ascending=False)
        per_class_df.to_csv(f"shap_feature_importance_per_class_{filename.replace('.png','')}.csv", index=True)
        print(f" → Also saved: shap_feature_importance_per_class_...csv")

        return per_class_df

    except Exception as e:
        print(f" Warning: Could not plot stacked SHAP importance: {e}")
        return None
        
        
        
        
def plot_shap_summary(shap_values, X_sample, feature_names, title, filename, top_n=5):
    """Shapley Summary - Swarm/dot plot and violin plot."""
    try:
        if isinstance(shap_values, list):
            shap_for_plot = shap_values[0]
        elif len(shap_values.shape) == 3:
            shap_for_plot = shap_values.mean(axis=2)
        else:
            shap_for_plot = shap_values

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_for_plot, X_sample, feature_names=feature_names,
            show=False, max_display=top_n, plot_type="dot"
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

        # Violin plot version
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_for_plot, X_sample, feature_names=feature_names,
            show=False, max_display=top_n, plot_type="violin"
        )
        plt.title(title.replace('Summary', 'Summary (Violin)'), fontsize=14, fontweight='bold')
        plt.tight_layout()
        violin_filename = filename.replace('.png', '_violin.png')
        plt.savefig(violin_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {violin_filename}")
    except Exception as e:
        print(f"  Warning: Could not plot SHAP summary: {e}")

def plot_shap_dependence(shap_values, X_sample, feature_names, title, filename, top_n=5):
    """Shapley Dependence - Scatter plot showing feature value vs SHAP value."""
    try:
        if isinstance(shap_values, list):
            shap_for_plot = np.mean(shap_values, axis=0)
        elif len(shap_values.shape) == 3:
            shap_for_plot = shap_values.mean(axis=2)
        else:
            shap_for_plot = shap_values

        importance = np.abs(shap_for_plot).mean(axis=0)
        top_indices = np.argsort(importance)[::-1][:top_n]

        n_features = len(top_indices)
        n_rows = int(np.ceil(n_features / 2))

        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, feat_idx in enumerate(top_indices):
            plt.sca(axes[idx])
            shap.dependence_plot(
                feat_idx,
                shap_for_plot,
                X_sample,
                feature_names=feature_names,
                show=False,
                ax=axes[idx]
            )
            axes[idx].set_title(f'SHAP Dependence: {feature_names[feat_idx]}', fontsize=11, fontweight='bold')

        for idx in range(len(top_indices), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
    except Exception as e:
        print(f"  Warning: Could not plot SHAP dependence: {e}")

def plot_lime_explanation(model, X_sample, feature_names, class_names, title, filename, sample_idx=0):
    """
    FIXED v8: LIME Plot - Local interpretable explanation for a single sample.
    
    Fixed: `as_pyplot_figure()` no longer accepts `ax` parameter.
    Creates figure separately and sets title after.
    """
    print(f"  Computing LIME explanation for sample {sample_idx}...")
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_sample.astype(float),
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

        exp = explainer.explain_instance(
            X_sample[sample_idx].astype(float),
            model.predict_proba,
            num_features=5
        )

        # FIXED: Create figure separately, don't pass ax to as_pyplot_figure()
        fig = exp.as_pyplot_figure()
        ax = fig.get_axes()[0] if fig.get_axes() else None
        if ax:
            ax.set_title(title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
        return exp
    except Exception as e:
        print(f"  Warning: Could not compute LIME explanation: {e}")
        return None

def plot_local_shap(model, X_sample, feature_names, title, filename, sample_idx=0, class_idx=None):
    """
    FIXED v8: Local Shapley Plot - SHAP values for a single prediction.
    
    Fixed: Modern SHAP waterfall() API changed - now uses shap.Explanation
    object instead of passing feature_names directly.
    """
    print(f"  Computing Local SHAP for sample {sample_idx}...")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values_single = explainer.shap_values(X_sample[sample_idx:sample_idx+1], check_additivity=False)

        # Handle multiclass format
        if isinstance(shap_values_single, list):
            if class_idx is None:
                pred_proba = model.predict_proba(X_sample[sample_idx:sample_idx+1])
                class_idx = np.argmax(pred_proba[0])
            shap_vals = shap_values_single[class_idx][0]
            class_name = f"Class {class_idx}"
            expected_val = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value
        elif len(shap_values_single.shape) == 3:
            if class_idx is None:
                pred_proba = model.predict_proba(X_sample[sample_idx:sample_idx+1])
                class_idx = np.argmax(pred_proba[0])
            shap_vals = shap_values_single[0, :, class_idx]
            class_name = f"Class {class_idx}"
            expected_val = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, list) else explainer.expected_value
        else:
            shap_vals = shap_values_single[0]
            class_name = ""
            expected_val = explainer.expected_value

        sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

        # Bar plot of local SHAP values
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['green' if v > 0 else 'red' for v in shap_vals[sorted_idx]]
        ax.barh(
            [feature_names[i] for i in sorted_idx],
            shap_vals[sorted_idx],
            color=colors
        )
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(f'{title} - {class_name}', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")

        # FIXED v8: Waterfall plot using modern SHAP API
        plt.figure(figsize=(12, 8))
        try:
            # Create Explanation object (modern SHAP API)
            exp_obj = shap.Explanation(
                values=shap_vals,
                base_values=expected_val,
                data=X_sample[sample_idx],
                feature_names=feature_names
            )
            shap.plots.waterfall(exp_obj, show=False, max_display=5)
            plt.title(f'Waterfall - Local SHAP - {class_name}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            waterfall_filename = filename.replace('.png', '_waterfall.png')
            plt.savefig(waterfall_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {waterfall_filename}")
        except Exception as e:
            print(f"    Warning: Could not create waterfall plot: {e}")
            # Fallback: just save bar plot without waterfall
    except Exception as e:
        print(f"  Warning: Could not compute local SHAP: {e}")

def plot_local_shap_per_class(shap_values, X_sample, y_sample, feature_names, class_names,
                               title, filename_prefix, top_n=5):
    """
    Local Shapley Plot PER CLASS - Shows top N features for each class.

    Reuses pre-computed SHAP values instead of recomputing.
    Properly handles both list format and 3D array (n_samples, n_features, n_classes).

    Outputs:
        - fig_local_shap_sample_combined.png (all classes in subplot grid)
        - fig_local_shap_sample_class_{idx}_{ClassName}.png (one per class)

    Color coding:
        - Green bars: Features that push prediction TOWARD this class
        - Red bars: Features that push prediction AWAY from this class
    """
    print(f"\n  Generating Local SHAP per class (top {top_n} features)...")

    try:
        n_classes = len(class_names)
        n_samples = len(X_sample)

        # Handle multiclass SHAP values format
        shap_per_class = {}

        if isinstance(shap_values, list):
            for class_idx in range(len(shap_values)):
                shap_per_class[class_idx] = shap_values[class_idx]
        elif len(shap_values.shape) == 3:
            for class_idx in range(shap_values.shape[2]):
                shap_per_class[class_idx] = shap_values[:, :, class_idx]
        else:
            shap_per_class[0] = shap_values

        # Generate combined subplot figure
        n_rows = int(np.ceil(n_classes / 2))
        n_cols = 2 if n_classes > 1 else 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
        if n_classes == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        individual_plots_data = {}

        for class_idx in range(n_classes):
            ax = axes[class_idx]

            class_mask = (y_sample == class_idx)
            n_class_samples = np.sum(class_mask)

            if n_class_samples == 0:
                ax.text(0.5, 0.5, f'No samples for\n{class_names[class_idx]}',
                       ha='center', va='center', fontsize=12, transform=ax.transAxes)
                ax.set_title(f'{class_names[class_idx]}', fontsize=12, fontweight='bold')
                continue

            class_shap = shap_per_class[class_idx][class_mask]
            mean_abs_shap = np.abs(class_shap).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
            top_features = [feature_names[i] for i in top_indices]
            top_values = mean_abs_shap[top_indices]
            mean_shap = class_shap.mean(axis=0)[top_indices]

            individual_plots_data[class_idx] = {
                'top_features': top_features,
                'top_values': top_values,
                'mean_shap': mean_shap,
                'n_samples': n_class_samples
            }

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
            ax.barh(top_features, top_values, color=colors)
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
            ax.set_ylabel('Feature', fontsize=10)
            ax.set_title(f'{class_names[class_idx]} (n={n_class_samples})', fontsize=11, fontweight='bold')

        for idx in range(n_classes, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{title}\n(Top {top_n} Features per Class)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        combined_filename = f'{filename_prefix}_combined.png'
        plt.savefig(combined_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {combined_filename}")

        # Generate individual plots per class
        for class_idx, data in individual_plots_data.items():
            fig, ax = plt.subplots(figsize=(10, 8))

            top_features = data['top_features']
            top_values = data['top_values']
            mean_shap = data['mean_shap']
            n_class_samples = data['n_samples']

            colors = ['forestgreen' if v > 0 else 'crimson' for v in mean_shap]
            bars = ax.barh(top_features, top_values, color=colors,
                          edgecolor='black', linewidth=0.5)
            ax.invert_yaxis()
            ax.set_xlabel('Mean |SHAP Value| (Average Impact on Prediction)', fontsize=12)
            ax.set_ylabel('Feature', fontsize=12)
            ax.set_title(f'Local SHAP - {class_names[class_idx]}\n'
                        f'(Top {top_n} Features for Class Prediction, n={n_class_samples})',
                        fontsize=14, fontweight='bold')

            for bar, val in zip(bars, top_values):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{val:.4f}', va='center', fontsize=10)

            plt.tight_layout()
            class_filename = (f'{filename_prefix}_class_{class_idx}_'
                            f'{class_names[class_idx].replace(" ", "_").replace("/", "_")}.png')
            plt.savefig(class_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {class_filename}")

        print(f"  Completed Local SHAP per class plots")
        return True

    except Exception as e:
        print(f"  Warning: Could not compute local SHAP per class: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_all_figures(y_cv_true, y_cv_pred, y_cv_proba, y_test_true, y_test_pred, y_test_proba,
                     class_names, model, feature_names, y_train, X_train_scaled=None,
                     shap_values=None, X_sample=None, y_sample=None):
    """
    Save all figures for conference report including all interpretability plots.

    IMPROVED v8: Better handling of small sample sizes and API compatibility fixes.
    """
    print(f"\n{'='*70}")
    print(" SAVING CONFERENCE FIGURES")
    print(f"{'='*70}")

    # Performance Plots
    plot_confusion_matrix(y_cv_true, y_cv_pred, class_names,
                          'Confusion Matrix - Cross-Validation (Normalized)',
                          'fig_confusion_matrix_cv_normalized.png', normalize=True)
    plot_confusion_matrix(y_test_true, y_test_pred, class_names,
                          'Confusion Matrix - Test Set (Normalized)',
                          'fig_confusion_matrix_test_normalized.png', normalize=True)
    plot_confusion_matrix(y_cv_true, y_cv_pred, class_names,
                          'Confusion Matrix - Cross-Validation',
                          'fig_confusion_matrix_cv.png', normalize=False)
    plot_confusion_matrix(y_test_true, y_test_pred, class_names,
                          'Confusion Matrix - Test Set',
                          'fig_confusion_matrix_test.png', normalize=False)
    plot_roc_curves(y_cv_true, y_cv_proba, class_names,
                     'ROC Curves - Cross-Validation', 'fig_roc_curves_cv.png')
    plot_roc_curves(y_test_true, y_test_proba, class_names,
                     'ROC Curves - Test Set', 'fig_roc_curves_test.png')
    plot_precision_recall_curves(y_cv_true, y_cv_proba, class_names,
                                'Precision-Recall Curves - Cross-Validation',
                                'fig_pr_curves_cv.png')
    plot_precision_recall_curves(y_test_true, y_test_proba, class_names,
                                'Precision-Recall Curves - Test Set',
                                'fig_pr_curves_test.png')
    plot_class_distribution(y_train, class_names,
                           'Training Set Class Distribution',
                           'fig_class_distribution_train.png')
    plot_feature_importance(model, feature_names,
                            'Top 5 Feature Importance (XGBoost)',
                            'fig_feature_importance_xgboost.png')

    # =====================================================================
    # INTERPRETABILITY PLOTS
    # =====================================================================
    print(f"\n{'='*70}")
    print(" SAVING INTERPRETABILITY PLOTS")
    print(f"{'='*70}")

    # Determine data source for interpretability
    if X_sample is not None and y_sample is not None:
        X_interp = X_sample
        y_interp = y_sample
        print(f"\n  Using SHAP-computed samples: {len(X_interp)} samples")
    elif X_train_scaled is not None:
        sample_size = min(INTERPRETABILITY_SAMPLE_SIZE, len(X_train_scaled))
        indices = np.random.choice(len(X_train_scaled), sample_size, replace=False)
        X_interp = X_train_scaled[indices]
        y_interp = y_train[indices]
        print(f"\n  Using random sample for interpretability: {len(X_interp)} samples")
    else:
        X_interp = None
        y_interp = None
        print("\n  Warning: No interpretability data available")

    # 1. Permutation Importance
    if y_interp is not None and X_interp is not None and len(X_interp) >= 10:
        plot_permutation_importance(
            model, X_interp, y_interp, feature_names,
            'Permutation Importance (Top 10)',
            'fig_permutation_importance.png', top_n=10
        )
    elif X_interp is not None:
        print(f"  Skipping permutation importance (only {len(X_interp)} samples)")

    # 2. Partial Dependence Plots (FIXED: added target parameter)
    if X_interp is not None and len(X_interp) >= 50:
        # Generate PDP for each class
        for class_idx in range(len(class_names)):
            plot_partial_dependence(
                model, X_interp, feature_names,
                f'Partial Dependence Plots (Top 5 Features)',
                f'fig_partial_dependence_class_{class_idx}.png',
                class_idx=class_idx
            )
    elif X_interp is not None:
        print(f"  Skipping PDP (only {len(X_interp)} samples, need >= 50)")

    # 3. SHAP-based interpretability plots
    if shap_values is not None and X_interp is not None:
        print(f"\n  Generating SHAP-based interpretability plots...")

        plot_shap_importance(
            shap_values, X_interp, feature_names,
            'Shapley Importance (Mean |SHAP Value|)',
            'fig_shap_importance.png',
            class_names,          # ← ADD THIS LINE
            top_n=10
        )
        plot_shap_summary(
            shap_values, X_interp, feature_names,
            'Shapley Summary - Feature Value Impact Distribution',
            'fig_shap_summary.png', top_n=10
        )
        plot_shap_dependence(
            shap_values, X_interp, feature_names,
            'Shapley Dependence - Feature Value vs SHAP Value',
            'fig_shap_dependence.png', top_n=10
        )

        # Local interpretation plots
        if len(X_interp) >= 3:
            sample_indices = [0, min(len(X_interp)//2, 10), min(len(X_interp)-1, 20)]
            sample_indices = [i for i in sample_indices if i < len(X_interp)]

            for sample_idx in sample_indices:
                plot_lime_explanation(
                    model, X_interp, feature_names, class_names,
                    f'LIME Explanation - Sample {sample_idx}',
                    f'fig_lime_sample_{sample_idx}.png', sample_idx=sample_idx
                )
                plot_local_shap(
                    model, X_interp, feature_names,
                    f'Local SHAP Values - Sample {sample_idx}',
                    f'fig_local_shap_sample_{sample_idx}.png', sample_idx=sample_idx
                )
        else:
            print(f"  Skipping LIME/Local SHAP (only {len(X_interp)} samples)")

        # Local SHAP per class
        if y_sample is not None and len(y_sample) == len(X_interp):
            print("\n  Generating Local SHAP plots per class (top 5 features)...")
            plot_local_shap_per_class(
                shap_values, X_interp, y_sample, feature_names, class_names,
                'Local SHAP Values per Class',
                'fig_local_shap_sample', top_n=5
            )

    elif X_interp is not None and len(X_interp) >= 3:
        print(f"\n  SHAP disabled - generating LIME explanations only...")
        sample_indices = [0, min(len(X_interp)//2, 10), min(len(X_interp)-1, 20)]
        sample_indices = [i for i in sample_indices if i < len(X_interp)]
        for sample_idx in sample_indices:
            plot_lime_explanation(
                model, X_interp, feature_names, class_names,
                f'LIME Explanation - Sample {sample_idx}',
                f'fig_lime_sample_{sample_idx}.png', sample_idx=sample_idx
            )

    print(f"\n{'='*70}")
    print(" All conference figures saved successfully!")
    print("=" * 70)
    print("Generated figures:")
    print("  [Performance Plots]")
    print("  - fig_confusion_matrix_cv_normalized.png")
    print("  - fig_confusion_matrix_test_normalized.png")
    print("  - fig_confusion_matrix_cv.png")
    print("  - fig_confusion_matrix_test.png")
    print("  - fig_roc_curves_cv.png")
    print("  - fig_roc_curves_test.png")
    print("  - fig_pr_curves_cv.png")
    print("  - fig_pr_curves_test.png")
    print("  - fig_class_distribution_train.png")
    print("  - fig_feature_importance_xgboost.png (Top 5)")
    print("  [Interpretability Plots - Top 5 Features]")
    print("  - fig_permutation_importance.png (Permutation Importance)")
    if shap_values is not None:
        print("  - fig_partial_dependence_class_*.png (1-way PDP per class)")
        print("  - fig_partial_dependence_class_*_interaction.png (2-way PDP)")
        print("  - fig_shap_importance.png (Shapley Importance)")
        print("  - fig_shap_summary.png (Shapley Summary - Dot)")
        print("  - fig_shap_summary_violin.png (Shapley Summary - Violin)")
        print("  - fig_shap_dependence.png (Shapley Dependence)")
        print("  - fig_lime_sample_*.png (LIME for selected samples)")
        print("  - fig_local_shap_sample_*.png (Local SHAP for selected samples)")
        print("  [Local SHAP per Class - Top 5 Features]")
        print("  - fig_local_shap_sample_combined.png (All classes combined)")
        print("  - fig_local_shap_sample_class_*_*.png (One per class)")
    else:
        print("  - fig_lime_sample_*.png (LIME only - SHAP disabled)")
    print("=" * 70)

# =====================================================================
# OPTUNA OBJECTIVE FUNCTION
# =====================================================================
def objective(trial, X_train, y_train, X_val, y_val, weights_train):
    """Optuna objective for hyperparameter optimization."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 400, 1400, step=100),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'eval_metric': 'mlogloss',
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'tree_method': 'hist',
        'random_state': 7,
        'n_jobs': -1,
    }
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=weights_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    pred = model.predict(X_val)
    return f1_score(y_val, pred, average='macro')

# =====================================================================
# TRAINING FUNCTION
# =====================================================================
def train_model(df_mode, y, class_names, save_prefix):
    """Train XGBoost model with proper overfitting prevention."""
    print(f"\n{'='*90}")
    print(" TRAINING: Fault Detection - CORRECT OVERFITTING PREVENTION (NO LEAKAGE)")
    print(f"{'='*90}")

    print(f"\n[Class Weight Configuration]")
    print(f"  Base: sklearn balanced weights (all classes)")
    print(f"  Class 0 (Normal) multiplier: {NORMAL_WEIGHT}")
    print("-" * 90)

    exclude_cols = ['has_fault']
    feature_names_all = [c for c in df_mode.columns if c not in exclude_cols]
    X = df_mode[feature_names_all].values
    print(f"Using all {len(feature_names_all)} pre-ranked features.")

    print(f"\nCreating {FINAL_VAL_SIZE*100:.1f}% hold-out validation set...")
    X_hpo, X_val_es, y_hpo, y_val_es = train_test_split(
        X, y, test_size=FINAL_VAL_SIZE, stratify=y, random_state=7, shuffle=True
    )
    print(f"   -> HPO + CV uses {len(y_hpo)} samples")
    print(f"   -> Final early-stopping validation uses {len(y_val_es)} samples")

    total_samples = len(y)
    unique, counts = np.unique(y, return_counts=True)
    class_distribution = {int(k): int(v) for k, v in zip(unique, counts)}
    print(f"\n[Class Distribution]")
    print("-" * 70)
    for idx, name in enumerate(class_names):
        count = class_distribution.get(idx, 0)
        pct = 100 * count / total_samples if total_samples > 0 else 0
        print(f" {name:<20}: {count:>8} samples ({pct:>5.1f}%)")
    print(f" {'TOTAL':<20}: {total_samples:>8}")
    print("-" * 70)

    # Step 1: Nested HPO with 5-Fold CV
    print(f"\nStep 1: {N_FOLDS}-Fold Nested Optuna HPO ({N_OPTUNA_TRIALS} trials per fold)...")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=7)
    all_y_true, all_y_pred = [], []
    fold_f1 = []
    best_params_per_fold = []
    actual_n_estimators_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_hpo), 1):
        X_train, X_val = X_hpo[train_idx], X_hpo[val_idx]
        y_train, y_val = y_hpo[train_idx], y_hpo[val_idx]

        weights_train = get_custom_sample_weights(y_train, normal_weight=NORMAL_WEIGHT)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=7))
        study.optimize(
            lambda trial: objective(trial, X_train_scaled, y_train, X_val_scaled, y_val, weights_train),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=True
        )

        best_params = study.best_params
        best_params['tree_method'] = 'hist'
        best_params['random_state'] = 7
        best_params['n_jobs'] = -1
        best_params['eval_metric'] = 'mlogloss'
        best_params['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS
        best_params_per_fold.append(best_params)

        print(f" Fold {fold}: Best Macro F1 = {study.best_value:.4f}")

        model = XGBClassifier(**best_params)
        model.fit(
            X_train_scaled, y_train,
            sample_weight=weights_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )
        actual_trees = model.best_iteration + 1 if hasattr(model, 'best_iteration') and model.best_iteration is not None else 500
        actual_n_estimators_per_fold.append(actual_trees)

        y_pred = model.predict(X_val_scaled)
        f1 = f1_score(y_val, y_pred, average='macro')
        fold_f1.append(f1)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    mean_f1 = np.mean(fold_f1)
    std_f1 = np.std(fold_f1)
    validation_metrics = calculate_metrics(all_y_true, all_y_pred, "Validation")

    print(f"\nCV Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f" -> Average trees used in CV folds: {int(round(np.mean(actual_n_estimators_per_fold)))}")

    # Step 2: Final Model Training
    print(f"\nStep 2: Training Final Model on 90% data with proper early stopping...")
    training_start_time = time.time()

    avg_params = {}
    for key in best_params_per_fold[0].keys():
        values = [p[key] for p in best_params_per_fold]
        if isinstance(values[0], (int, float)):
            avg_params[key] = float(np.mean(values))
        else:
            avg_params[key] = values[0]

    for key in ['max_depth', 'min_child_weight', 'random_state', 'n_jobs']:
        if key in avg_params:
            avg_params[key] = int(round(avg_params[key]))

    avg_params['n_estimators'] = 2000
    avg_params['eval_metric'] = 'mlogloss'
    avg_params['early_stopping_rounds'] = EARLY_STOPPING_ROUNDS

    final_scaler = StandardScaler()
    X_hpo_scaled = final_scaler.fit_transform(X_hpo)
    X_val_es_scaled = final_scaler.transform(X_val_es)

    weights_hpo = get_custom_sample_weights(y_hpo, normal_weight=NORMAL_WEIGHT)

    final_model = XGBClassifier(**avg_params)
    final_model.fit(
        X_hpo_scaled, y_hpo,
        sample_weight=weights_hpo,
        eval_set=[(X_val_es_scaled, y_val_es)],
        verbose=False
    )

    optimal_trees = final_model.best_iteration + 1 if hasattr(final_model, 'best_iteration') and final_model.best_iteration is not None else 500
    actual_final_trees = optimal_trees
    training_time = time.time() - training_start_time

    print(f" -> Early stopping determined optimal trees: {optimal_trees}")
    print(f" -> Re-training final model with exact {optimal_trees} trees...")

    avg_params['n_estimators'] = optimal_trees
    avg_params.pop('early_stopping_rounds', None)

    final_model_fixed = XGBClassifier(**avg_params)
    final_model_fixed.fit(
        X_hpo_scaled, y_hpo,
        sample_weight=weights_hpo,
        verbose=False
    )

    final_model = final_model_fixed
    print(f" -> Final model trained with {actual_final_trees} trees")

    # Save models and artifacts
    joblib.dump(final_model, f'model_{save_prefix}.joblib')
    joblib.dump(final_scaler, f'scaler_{save_prefix}.joblib')
    joblib.dump(feature_names_all, f'features_{save_prefix}.joblib')
    joblib.dump(class_names, f'classes_{save_prefix}.joblib')

    # Step 3: SHAP Analysis (if enabled)
    shap_values = None
    X_sample = None
    y_sample = None

    if RUN_SHAP_ANALYSIS:
        print(f"\nStep 3: Running SHAP Interpretability Analysis...")
        shap_values, X_sample, y_sample, sample_indices = run_shap_analysis(
            final_model, X_hpo_scaled, y_hpo, feature_names_all, save_prefix
        )
    else:
        print(f"\nStep 3: SKIPPING SHAP Analysis (set RUN_SHAP_ANALYSIS=True to enable)")

    print(f"\n{'='*70}")
    print(" CLASSIFICATION REPORT (CV Aggregated)")
    print(f"{'='*70}")
    print(classification_report(all_y_true, all_y_pred, target_names=class_names, zero_division=0))
    print(f"\nConfusion Matrix:\n{confusion_matrix(all_y_true, all_y_pred)}")

    return {
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'validation_metrics': validation_metrics,
        'cv_y_true': all_y_true,
        'cv_y_pred': all_y_pred,
        'final_model': final_model,
        'final_scaler': final_scaler,
        'X_train_scaled': X_hpo_scaled,
        'feature_names': feature_names_all,
        'actual_final_trees': actual_final_trees,
        'best_hyperparameters': avg_params,
        'training_time': training_time,
        'final_train_samples': len(y_hpo),
        'y_train': y_hpo,
        'normal_weight': NORMAL_WEIGHT,
        'shap_values': shap_values,
        'X_sample': X_sample,
        'y_sample': y_sample
    }

# =====================================================================
# MAIN FUNCTION
# =====================================================================
def main():
    total_start_time = time.time()
    print("=" * 100)
    print(" UNIFIED FAULT DETECTION TRAINING v8 - API COMPATIBILITY FIXES")
    print(f" Config: {N_FOLDS}-Fold CV, {N_OPTUNA_TRIALS} Optuna trials, {EARLY_STOPPING_ROUNDS} early stopping")
    print(f"         Normal Weight: {NORMAL_WEIGHT}, SHAP Analysis: {'Enabled' if RUN_SHAP_ANALYSIS else 'Disabled'}")
    print("=" * 100)

    # Load data
    df_train = pd.read_csv(DEFAULT_TRAIN)
    df_test = pd.read_csv(DEFAULT_TEST)

    has_fault_col = 'has_fault' if 'has_fault' in df_train.columns else next(
        (c for c in df_train.columns if 'has_fault' in c.lower()), None
    )
    if has_fault_col is None:
        raise ValueError("No has_fault column found")

    train_feature_cols = [c for c in df_train.columns if c != has_fault_col]
    missing_in_test = [c for c in train_feature_cols if c not in df_test.columns]
    if missing_in_test:
        print(f"Warning: Filling {len(missing_in_test)} missing columns in test with 0.")
        for c in missing_in_test:
            df_test[c] = 0
    df_test = df_test[train_feature_cols + [has_fault_col]]

    y_raw_train = df_train[has_fault_col].values.astype(int)
    y_mapped_train, class_names, mask_train, present_classes = process_labels(y_raw_train)
    df_mode_train = df_train[mask_train].reset_index(drop=True)

    results = train_model(df_mode_train, y_mapped_train, class_names, save_prefix="fault_detection")

    # External Test Set Evaluation
    print(f"\n{'='*70}")
    print(" EXTERNAL TEST SET PERFORMANCE")
    print(f"{'='*70}")
    y_raw_test = df_test[has_fault_col].values.astype(int)
    valid_mask_test = np.isin(y_raw_test, present_classes)
    df_mode_test = df_test[valid_mask_test].reset_index(drop=True)
    if len(df_mode_test) == 0:
        raise ValueError("No matching fault values in external test set!")

    name_to_idx = {name: i for i, name in enumerate(class_names)}
    y_test_ext = np.array([name_to_idx[CLASS_MAPPING[l]] for l in df_mode_test[has_fault_col].values])
    X_test_ext = df_mode_test[results['feature_names']].values
    X_test_ext_scaled = results['final_scaler'].transform(X_test_ext)
    y_pred_test = results['final_model'].predict(X_test_ext_scaled)

    print(classification_report(y_test_ext, y_pred_test, target_names=class_names, zero_division=0))
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test_ext, y_pred_test)}")
    test_metrics = calculate_metrics(y_test_ext, y_pred_test, "External_Test")

    print("\nGenerating probability predictions for ROC/PR curves...")
    y_cv_proba = results['final_model'].predict_proba(results['X_train_scaled'])
    y_test_proba = results['final_model'].predict_proba(X_test_ext_scaled)

    # Save all figures
    save_all_figures(
        y_cv_true=results['cv_y_true'],
        y_cv_pred=results['cv_y_pred'],
        y_cv_proba=y_cv_proba,
        y_test_true=y_test_ext,
        y_test_pred=y_pred_test,
        y_test_proba=y_test_proba,
        class_names=class_names,
        model=results['final_model'],
        feature_names=results['feature_names'],
        y_train=results['y_train'],
        X_train_scaled=results['X_train_scaled'],
        shap_values=results['shap_values'],
        X_sample=results['X_sample'],
        y_sample=results['y_sample']
    )

    print("\n" + "=" * 120)
    print(" PERFORMANCE SUMMARY")
    print("=" * 120)
    print(f"CV Macro F1: {results['mean_f1']:.4f} +/- {results['std_f1']:.4f}")
    print(f"External Test Macro F1: {f1_score(y_test_ext, y_pred_test, average='macro'):.4f}")
    print(f"Final model trained on {results['final_train_samples']} samples + early-stopped on unseen 10%")
    print(f"Final model used {results['actual_final_trees']} trees")
    print(f"Normal class weight: {results['normal_weight']}")
    print(f"Best Hyperparameters: {results['best_hyperparameters']}")
    print("-" * 120)

    total_time = time.time() - total_start_time
    print(f"Total execution time: {total_time:.1f} seconds")

    csv_results = {"predictors": ', '.join(results['feature_names'])}
    csv_results.update(results['validation_metrics'])
    csv_results.update(test_metrics)
    csv_results['CV Macro F1'] = f"{results['mean_f1']:.4f} +/- {results['std_f1']:.4f}"
    csv_results['Normal Weight'] = results['normal_weight']
    csv_results['Best Hyperparameters'] = str(results['best_hyperparameters'])
    df_results = pd.DataFrame([csv_results])
    df_results.to_csv("training_results.csv", index=False)
    print("\nResults saved to training_results.csv")

if __name__ == "__main__":
    main()

