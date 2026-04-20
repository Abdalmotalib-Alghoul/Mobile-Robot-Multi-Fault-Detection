"""
Unified causal feature extraction - ALL previous features combined
- Causal window: exactly 60 samples (last 60 including current)
- Step = 30 (50% overlap)
- All features computed exclusively inside each window (no future data)
- Combines every feature we tried so far + drift-sensitive dynamics
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis, pearsonr
from scipy.signal import savgol_filter
import warnings

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

INPUT_FILE  = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/Final_Dataset_19Gen_with_runid_clean.csv"
OUTPUT_FILE = "/home/talib/collected_datasets/Final_pipline/Pipline_Dataset_offset_20cm/causal_60_step30_all_features_combined_final.csv"

WINDOW_SIZE = 60
STEP_SIZE   = 30           # 50% overlap

# ────────────────────────────────────────────────
# SAFE STATISTICAL WRAPPERS (from both scripts, renamed to avoid conflicts)
# ────────────────────────────────────────────────

def safe_slope_first_script(t, y):
    """Safe linear slope from first script; returns 0 if fit fails or not enough points."""
    if len(t) < 10:
        return 0.0
    
    t_np = np.asarray(t)
    y_np = np.asarray(y)
    
    if np.any(np.isnan(y_np)):
        return 0.0
    
    if np.all(y_np == y_np[0]):
        return 0.0
    
    try:
        return np.polyfit(t_np, y_np, 1)[0]
    except:
        return 0.0

def safe_corr(x, y):
    """Safe linear correlation from second script."""
    if len(x) < 2 or len(y) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    try:
        r, _ = pearsonr(x, y)
        return float(r) if np.isfinite(r) else 0.0
    except:
        return 0.0

def safe_slope_second_script(s):
    """Safe linear slope from second script; returns 0 if fit fails or not enough points."""
    if len(s) < 4 or np.std(s) < 1e-10:
        return 0.0
    try:
        slope, _ = np.polyfit(np.arange(len(s)), s, 1)
        return float(slope) if np.isfinite(slope) else 0.0
    except:
        return 0.0

def safe_autocorr(s):
    """Safe autocorrelation from second script."""
    if len(s) < 3 or np.var(s) < 1e-10:
        return 0.0
    try:
        m = np.mean(s)
        v = np.var(s)
        acf = np.sum((s[:-1] - m) * (s[1:] - m)) / (len(s) - 1) / (v + 1e-10)
        return float(acf) if np.isfinite(acf) else 0.0
    except:
        return 0.0

def safe_skew(x):
    """Safe skewness from second script."""
    if len(x) < 3 or np.std(x) < 1e-10:
        return 0.0
    try:
        val = skew(x, nan_policy='omit')
        return float(val) if np.isfinite(val) else 0.0
    except:
        return 0.0

def safe_kurtosis(x):
    """Safe kurtosis from second script."""
    if len(x) < 4 or np.std(x) < 1e-10:
        return 0.0
    try:
        val = kurtosis(x, nan_policy='omit')
        return float(val) if np.isfinite(val) else 0.0
    except:
        return 0.0

def angle_diff(a, b):
    """Compute angle difference handling wraparound (-pi to pi) from second script."""
    diff = a - b
    diff = np.mod(diff + np.pi, 2 * np.pi) - np.pi
    return diff

def validate_window(w_timestamps, w_labels):
    """Validate window from second script - return True if valid, False if degenerate.
    STRICT: Rejects dts <= 0 to handle duplicate or reversed timestamps.
    """
    if len(w_timestamps) < 2:
        return False
    dts = np.diff(w_timestamps)
    if np.any(dts <= 0):
        return False
    if not np.all(w_labels == w_labels[0]):
        return False
    return True

# ────────────────────────────────────────────────
# EXPERT FEATURE EXTRACTION FUNCTION (from second script)
# ────────────────────────────────────────────────

def extract_expert_features(w_beams, w_tf_x, w_tf_y, w_tf_yaw,
                            w_cmd_lin, w_cmd_ang, w_amcl_x, w_amcl_y, w_amcl_yaw, w_amcl_cov,
                            w_particle_count, w_particle_std_x, w_particle_std_y, w_particle_std_yaw,
                            w_timestamps,
                            w_particle_mean_yaw):
    """Extract exactly 288 features from a window - NO NaN, all real values."""
    
    features = []

    # Lagged command arrays
    cmd_lin_lag = np.roll(w_cmd_lin, 1)
    cmd_lin_lag[0] = cmd_lin_lag[1]
    cmd_ang_lag = np.roll(w_cmd_ang, 1)
    cmd_ang_lag[0] = cmd_ang_lag[1]
    
    # Time differences
    dt_array = np.diff(w_timestamps)
    
    # Unwrap yaw angles
    w_tf_yaw_unwrapped = np.unwrap(w_tf_yaw)
    w_amcl_yaw_unwrapped = np.unwrap(w_amcl_yaw)
    
    # Derivatives
    w_tf_dx = np.gradient(w_tf_x, w_timestamps)
    w_tf_dy = np.gradient(w_tf_y, w_timestamps)
    w_tf_speed = np.sqrt(w_tf_dx**2 + w_tf_dy**2)
    w_tf_dyaw = np.gradient(w_tf_yaw_unwrapped, w_timestamps)
    w_tf_ax = np.gradient(w_tf_dx, w_timestamps)
    w_tf_ay = np.gradient(w_tf_dy, w_timestamps)
    w_cmd_accel = np.gradient(cmd_lin_lag, w_timestamps)
    
    w_amcl_dx = np.gradient(w_amcl_x, w_timestamps)
    w_amcl_dy = np.gradient(w_amcl_y, w_timestamps)
    w_amcl_speed = np.sqrt(w_amcl_dx**2 + w_amcl_dy**2)
    w_amcl_dyaw = np.gradient(w_amcl_yaw_unwrapped, w_timestamps)
    
    beam_means = np.mean(w_beams, axis=1)
    beam_diffs = np.abs(np.diff(w_beams, axis=1))
    
    # ===== LIDAR (20 + 3 NEW = 23 features) =====
    features.append(float(np.mean(w_beams < np.percentile(w_beams, 25))))
    features.append(float(np.mean(beam_diffs > np.percentile(beam_diffs, 75))))
    features.append(float(np.mean(np.std(w_beams, axis=0))))
    features.append(float(np.std(np.gradient(w_beams, axis=0))))
    features.append(float(np.max(np.std(w_beams, axis=0))))
    features.append(float(np.percentile(w_beams, 25)))
    features.append(float(np.std(np.diff(w_beams, axis=0))))
    features.append(float(np.min(w_beams)))
    features.append(float(np.mean(w_beams)))
    features.append(float(np.std(w_beams)))
    features.append(float(np.max(w_beams)))
    features.append(float(np.percentile(w_beams, 75)))
    features.append(float(np.mean(np.abs(np.diff(beam_means)))))
    features.append(safe_slope_second_script(beam_means))
    features.append(safe_autocorr(beam_means))
    features.append(float(np.std(np.diff(beam_means))))
    features.append(float(np.percentile(w_beams, 50)))
    features.append(float(np.mean(np.abs(w_beams - np.mean(w_beams)))))
    features.append(float(np.percentile(w_beams, 95)))
    features.append(float(np.percentile(w_beams, 5)))
    # NEW: run-invariant normalized version
    lidar_norm = (w_beams - np.mean(w_beams)) / (np.std(w_beams) + 1e-10)
    features.append(float(np.mean(lidar_norm))) # lidar_normalized_mean
    features.append(float(np.std(lidar_norm)))  # lidar_normalized_std
    # NEW: normalized beam entropy
    counts, _ = np.histogram(w_beams.flatten(), bins=20)
    p = counts / (np.sum(counts) + 1e-10)
    features.append(float(entropy(p + 1e-10))) # lidar_normalized_entropy

    # ===== TF POSE (24 + 6 NEW = 30 features) =====
    features.append(float(np.mean(w_tf_speed)))
    features.append(float(np.std(w_tf_speed)))
    features.append(float(np.max(w_tf_speed)))
    features.append(float(np.mean(np.abs(w_tf_dyaw))))
    features.append(float(np.std(np.abs(w_tf_dyaw))))
    features.append(float(np.max(np.abs(w_tf_dyaw))))
    features.append(safe_slope_second_script(w_tf_x))
    features.append(safe_slope_second_script(w_tf_y))
    features.append(safe_slope_second_script(w_tf_yaw))
    features.append(float(np.mean(w_tf_ax)))
    features.append(float(np.mean(w_tf_ay)))
    features.append(float(np.std(w_tf_ax)))
    features.append(float(np.std(w_tf_ay)))
    features.append(safe_corr(cmd_lin_lag[1:], w_tf_speed[1:]))
    features.append(safe_corr(cmd_ang_lag[1:], np.abs(w_tf_dyaw[1:])))
    features.append(safe_autocorr(w_tf_x))
    features.append(safe_autocorr(w_tf_y))
    features.append(safe_autocorr(w_tf_yaw))
    features.append(float(np.mean(np.abs(cmd_lin_lag[1:] - w_tf_speed[1:]))))
    features.append(float(np.mean(w_cmd_accel)))
    features.append(float(np.std(w_cmd_accel)))
    features.append(float(np.max(np.abs(w_cmd_accel))))
    features.append(safe_corr(cmd_lin_lag, cmd_ang_lag))
    features.append(safe_corr(w_tf_ax[1:], w_cmd_accel[:-1]))
    # NEW: run-invariant normalized version
    features.append(float(np.mean((w_tf_x - np.mean(w_tf_x)) / (np.std(w_tf_x) + 1e-10)))) # tf_x_normalized_mean
    features.append(float(np.mean((w_tf_y - np.mean(w_tf_y)) / (np.std(w_tf_y) + 1e-10)))) # tf_y_normalized_mean
    features.append(float(np.mean((w_tf_speed - np.mean(w_tf_speed)) / (np.std(w_tf_speed) + 1e-10)))) # tf_speed_normalized_mean
    features.append(float(w_tf_x[-1] - w_tf_x[0])) # tf_x_delta
    features.append(float(w_tf_y[-1] - w_tf_y[0])) # tf_y_delta
    features.append(float(angle_diff(w_tf_yaw[-1], w_tf_yaw[0]))) # tf_yaw_delta

    # ===== AMCL (20 + 7 NEW = 27 features) =====
    features.append(float(np.mean(w_amcl_x)))
    features.append(float(np.mean(w_amcl_y)))
    features.append(float(np.mean(w_amcl_yaw)))
    features.append(float(np.std(w_amcl_x)))
    features.append(float(np.std(w_amcl_y)))
    features.append(float(np.std(w_amcl_yaw)))
    features.append(float(np.mean(w_amcl_speed)))
    features.append(float(np.std(w_amcl_speed)))
    features.append(float(np.mean(np.abs(w_amcl_dyaw))))
    features.append(float(np.std(np.abs(w_amcl_dyaw))))
    features.append(safe_slope_second_script(w_amcl_x))
    features.append(safe_slope_second_script(w_amcl_y))
    features.append(safe_slope_second_script(w_amcl_yaw))
    features.append(float(np.mean(w_amcl_cov)))
    features.append(float(np.std(w_amcl_cov)))
    features.append(float(np.max(w_amcl_cov)))
    features.append(float(np.mean(np.abs(w_tf_x - w_amcl_x))))
    features.append(float(np.mean(np.abs(w_tf_y - w_amcl_y))))
    features.append(float(np.mean(np.abs(w_tf_yaw - w_amcl_yaw))))
    features.append(safe_corr(w_tf_speed, w_amcl_speed))
    # NEW: run-invariant normalized version for AMCL
    features.append(float(np.mean((w_amcl_x - np.mean(w_amcl_x)) / (np.std(w_amcl_x) + 1e-10)))) # amcl_x_normalized_mean
    features.append(float(np.mean((w_amcl_y - np.mean(w_amcl_y)) / (np.std(w_amcl_y) + 1e-10)))) # amcl_y_normalized_mean
    features.append(float(np.mean((w_amcl_speed - np.mean(w_amcl_speed)) / (np.std(w_amcl_speed) + 1e-10)))) # amcl_speed_normalized_mean
    features.append(float(w_amcl_x[-1] - w_amcl_x[0])) # amcl_x_delta
    features.append(float(w_amcl_y[-1] - w_amcl_y[0])) # amcl_y_delta
    features.append(float(angle_diff(w_amcl_yaw[-1], w_amcl_yaw[0]))) # amcl_yaw_delta
    features.append(float(np.mean(w_amcl_cov) / (np.mean(w_tf_speed) + 1e-10))) # amcl_cov_speed_ratio

    # ===== PARTICLE FILTER (20 + 4 NEW = 24 features) =====
    features.append(float(np.mean(w_particle_count)))
    features.append(float(np.std(w_particle_count)))
    features.append(float(np.max(w_particle_count)))
    features.append(float(np.min(w_particle_count)))
    features.append(float(np.mean(w_particle_std_x)))
    features.append(float(np.mean(w_particle_std_y)))
    features.append(float(np.mean(w_particle_std_yaw)))
    features.append(float(np.std(w_particle_std_x)))
    features.append(float(np.std(w_particle_std_y)))
    features.append(float(np.std(w_particle_std_yaw)))
    features.append(float(np.max(w_particle_std_x)))
    features.append(float(np.max(w_particle_std_y)))
    features.append(float(np.max(w_particle_std_yaw)))
    features.append(safe_corr(w_particle_count, w_amcl_cov))
    features.append(float(np.mean(w_particle_std_x + w_particle_std_y)))
    features.append(float(np.mean(w_particle_std_x * w_particle_std_y)))
    features.append(float(np.mean(w_particle_count) / (np.mean(w_particle_std_x) + 1e-10)))
    features.append(float(np.mean(w_particle_std_yaw) / (np.mean(w_particle_count) + 1e-10)))
    features.append(safe_slope_second_script(w_particle_count))
    features.append(safe_slope_second_script(w_particle_std_x + w_particle_std_y))
    # NEW: run-invariant normalized version
    features.append(float(np.mean((w_particle_std_x - np.mean(w_particle_std_x)) / (np.std(w_particle_std_x) + 1e-10)))) # particle_std_x_normalized_mean
    features.append(float(np.mean((w_particle_std_y - np.mean(w_particle_std_y)) / (np.std(w_particle_std_y) + 1e-10)))) # particle_std_y_normalized_mean
    features.append(float(np.mean((w_particle_std_yaw - np.mean(w_particle_std_yaw)) / (np.std(w_particle_std_yaw) + 1e-10)))) # particle_std_yaw_normalized_mean
    features.append(float(np.mean(w_particle_std_yaw) / (np.mean(w_tf_speed) + 1e-10))) # particle_std_yaw_tf_speed_ratio

    # ===== STOCHASTIC FAULTS (10 features) =====
    particle_uncertainty = w_particle_std_x + w_particle_std_y + w_particle_std_yaw
    features.append(float(np.mean(particle_uncertainty)))
    features.append(float(np.std(particle_uncertainty)))
    features.append(float(np.max(particle_uncertainty)))
    features.append(float(np.mean(w_particle_count) / (np.mean(particle_uncertainty) + 1e-10)))
    features.append(float(np.mean(w_amcl_cov) / (np.mean(w_particle_count) + 1e-10)))
    features.append(float(np.mean(particle_uncertainty) * np.mean(w_amcl_cov)))
    features.append(float(np.std(particle_uncertainty) / (np.mean(particle_uncertainty) + 1e-10)))
    features.append(float(np.mean(np.abs(np.diff(particle_uncertainty)))))
    features.append(safe_slope_second_script(particle_uncertainty))
    features.append(safe_autocorr(particle_uncertainty))

    # ===== SYSTEMATIC FAULTS (7 features) =====
    tf_amcl_dist = np.sqrt((w_tf_x - w_amcl_x)**2 + (w_tf_y - w_amcl_y)**2)
    # Systematic Faults (9 features)
    features.append(float(np.mean(tf_amcl_dist))) # tf_amcl_dist_mean
    features.append(float(np.mean(tf_amcl_dist) / (np.mean(particle_uncertainty) + 1e-10)))
    features.append(float(np.mean(tf_amcl_dist) * np.mean(w_amcl_cov)))
    features.append(safe_slope_second_script(tf_amcl_dist))
    features.append(float(np.mean(np.abs(w_tf_speed - w_amcl_speed))))
    features.append(safe_corr(tf_amcl_dist, particle_uncertainty))
    features.append(float(np.mean(np.abs(np.diff(tf_amcl_dist)))))
    
    
    
    
    
    
    features.append(safe_autocorr(tf_amcl_dist))         
    
    
    
    
    
    
    features.append(float(np.std(tf_amcl_dist))) # tf_amcl_dist_std

    # ===== LIDAR CENTROID (4 features) =====
    beam_x = np.arange(w_beams.shape[1])
    centroid_x_last = np.sum(w_beams[-1] * beam_x) / (np.sum(w_beams[-1]) + 1e-10)
    centroid_y_last = np.mean(w_beams[-1])
    features.append(float(centroid_x_last))
    half_idx_w = len(w_beams) // 2
    centroid_x_first = np.sum(np.mean(w_beams[:half_idx_w], axis=0) * beam_x) / (np.sum(np.mean(w_beams[:half_idx_w], axis=0)) + 1e-10)
    centroid_x_second = np.sum(np.mean(w_beams[half_idx_w:], axis=0) * beam_x) / (np.sum(np.mean(w_beams[half_idx_w:], axis=0)) + 1e-10)
    centroid_y_first = np.mean(w_beams[:half_idx_w])
    centroid_y_second = np.mean(w_beams[half_idx_w:])
    features.append(float(centroid_x_second - centroid_x_first))
    features.append(float(centroid_y_second - centroid_y_first))
    features.append(float(np.sqrt((centroid_x_second - centroid_x_first)**2 + (centroid_y_second - centroid_y_first)**2)))

    # ===== CROSS-VALIDATION RESIDUALS (5 features) =====
    features.append(float(np.mean(np.abs(centroid_x_last - np.mean(cmd_lin_lag)))))
    features.append(float(np.sqrt((centroid_x_last - np.mean(cmd_lin_lag))**2)))
    features.append(float(np.mean(np.abs(np.mean(w_tf_x[1:]) - np.mean(cmd_lin_lag[1:])))))
    features.append(float(np.sqrt((np.mean(w_tf_x[1:]) - np.mean(cmd_lin_lag[1:]))**2)))
    features.append(float(np.mean(np.abs(np.mean(w_tf_dyaw[1:]) - np.mean(cmd_ang_lag[1:])))))

    # ===== LIDAR-BASED YAW ESTIMATION (3 features) =====
    lidar_yaw_est = np.arctan2(centroid_y_last, centroid_x_last)
    features.append(float(np.mean(np.abs(w_tf_yaw - lidar_yaw_est))))
    features.append(safe_slope_second_script(w_tf_yaw - lidar_yaw_est))
    features.append(float(np.mean(np.abs(lidar_yaw_est - np.mean(cmd_ang_lag)))))

    # ===== GHOST/PHANTOM DETECTION (10 features) =====
    beam_median = np.median(w_beams[-1])
    beam_mad = np.median(np.abs(w_beams[-1] - beam_median))
    short_beams = w_beams[-1] < (beam_median - 2 * beam_mad)
    long_beams = w_beams[-1] > (beam_median + 2 * beam_mad)
    features.append(float(np.mean(short_beams)))
    features.append(float(np.mean(long_beams)))
    features.append(float(np.mean(np.abs(w_beams[-1][short_beams] - beam_median) / (beam_mad + 1e-10)) if np.any(short_beams) else 0.0))
    features.append(float(np.mean(np.abs(w_beams[-1][long_beams] - beam_median) / (beam_mad + 1e-10)) if np.any(long_beams) else 0.0))
    features.append(float(np.max(np.abs(w_beams[-1] - beam_median) / (beam_mad + 1e-10))))
    hist, _ = np.histogram(w_beams[-1], bins=10)
    features.append(float((np.max(hist) - np.mean(hist)) / (np.std(hist) + 1e-10)))
    features.append(float(np.sum(hist > np.mean(hist))))
    features.append(float(np.mean(short_beams | long_beams)))
    features.append(float(np.mean(np.abs(np.diff(w_beams[-1])))))
    features.append(float((np.mean(short_beams) + np.mean(long_beams)) / 2))

    # ===== PHYSICS-BASED SENSOR DELTAS (11 features) =====
    beam_delta = np.mean(np.abs(np.diff(beam_means)))
    tf_delta = np.mean(np.abs(np.diff(w_tf_speed)))
    cmd_lin_delta = np.mean(np.abs(np.diff(cmd_lin_lag)))
    cmd_ang_delta = np.mean(np.abs(np.diff(cmd_ang_lag)))
    amcl_delta = np.mean(np.abs(np.diff(w_amcl_speed)))
    features.append(float(np.abs(beam_delta - tf_delta)))
    features.append(float(np.abs(beam_delta - cmd_lin_delta)))
    features.append(float(np.abs(tf_delta - cmd_lin_delta)))
    features.append(float(np.abs(cmd_lin_delta - cmd_ang_delta)))
    sensor_deltas = [beam_delta, tf_delta, cmd_lin_delta, cmd_ang_delta, amcl_delta]
    features.append(float(np.var(sensor_deltas)))
    features.append(float(np.max(sensor_deltas) - np.mean(sensor_deltas)))
    features.append(float(np.mean(sensor_deltas)))
    features.append(float(np.std(sensor_deltas)))
    features.append(float(np.max(sensor_deltas)))
    features.append(float(np.min(sensor_deltas)))
    features.append(float(np.ptp(sensor_deltas)))

    # ===== DRIFT RATE FEATURES (15 features) =====
    tf_amcl_x_diff = w_tf_x - w_amcl_x
    tf_amcl_y_diff = w_tf_y - w_amcl_y
    tf_amcl_yaw_diff = angle_diff(w_tf_yaw, w_amcl_yaw)
    tf_amcl_dist_series = np.sqrt(tf_amcl_x_diff**2 + tf_amcl_y_diff**2)
    features.append(safe_slope_second_script(tf_amcl_x_diff))
    features.append(safe_slope_second_script(tf_amcl_y_diff))
    features.append(safe_slope_second_script(tf_amcl_yaw_diff))
    features.append(safe_slope_second_script(tf_amcl_dist_series))
    n_drift = len(tf_amcl_yaw_diff)
    features.append(float(np.mean(tf_amcl_yaw_diff[n_drift//2:]) - np.mean(tf_amcl_yaw_diff[:n_drift//2])))
    
    
    
    
    
    
    
    features.append(float(np.mean(tf_amcl_dist_series[n_drift//2:]) - np.mean(tf_amcl_dist_series[:n_drift//2]))) 
    
    
    
    
    
    
    
    features.append(float(np.std(tf_amcl_yaw_diff)))
    features.append(float(np.std(tf_amcl_dist_series)))
    features.append(safe_autocorr(tf_amcl_yaw_diff))
    features.append(safe_autocorr(tf_amcl_dist_series))
    features.append(float(tf_amcl_yaw_diff[-1] - tf_amcl_yaw_diff[0]))
    features.append(float(tf_amcl_dist_series[-1] - tf_amcl_dist_series[0]))
    # NEW: run-invariant drift rates
    win_dur = w_timestamps[-1] - w_timestamps[0] + 1e-10
    tot_dist = np.sum(w_tf_speed[:-1] * np.diff(w_timestamps)) + 1e-10
    features.append(float((tf_amcl_dist_series[-1] - tf_amcl_dist_series[0]) / win_dur)) # pos_drift_rate_new
    features.append(float(angle_diff(tf_amcl_yaw_diff[-1], tf_amcl_yaw_diff[0]) / win_dur)) # yaw_drift_rate_new
    features.append(float((tf_amcl_dist_series[-1] - tf_amcl_dist_series[0]) / tot_dist)) # pos_drift_per_dist

    # ===== DRIFT MAGNITUDE FEATURES (11 features) =====
    features.append(float(np.mean(np.abs(tf_amcl_x_diff))))
    features.append(float(np.mean(np.abs(tf_amcl_y_diff))))
    features.append(float(np.mean(np.abs(tf_amcl_yaw_diff))))
    features.append(float(np.mean(tf_amcl_yaw_diff)))
    features.append(float(np.mean(tf_amcl_dist_series)))
    features.append(float(np.max(np.abs(tf_amcl_x_diff))))
    features.append(float(np.max(np.abs(tf_amcl_y_diff))))
    features.append(float(np.max(np.abs(tf_amcl_yaw_diff))))
    features.append(float(np.max(tf_amcl_dist_series)))
    features.append(float(np.sum(np.abs(np.diff(tf_amcl_yaw_diff)))))
    features.append(float(np.sum(np.abs(np.diff(tf_amcl_dist_series)))))

    # ===== TF-CMD YAW RESIDUAL FEATURES (8 features) =====
    cmd_yaw_change = cmd_ang_lag[:-1] * np.diff(w_timestamps)
    tf_yaw_change = np.diff(w_tf_yaw_unwrapped)
    tf_cmd_yaw_residual_deg = np.degrees(tf_yaw_change - cmd_yaw_change)
    tf_cmd_total_residual_deg = np.sum(tf_cmd_yaw_residual_deg)
    features.append(float(tf_cmd_total_residual_deg / len(w_tf_yaw)))
    features.append(float(np.std(tf_cmd_yaw_residual_deg)))
    features.append(float(np.max(np.abs(tf_cmd_yaw_residual_deg))))
    features.append(safe_slope_second_script(tf_cmd_yaw_residual_deg))
    features.append(float(tf_cmd_total_residual_deg))
    features.append(float(np.sum(np.abs(tf_cmd_yaw_residual_deg))))
    features.append(float(tf_cmd_total_residual_deg))
    features.append(float(np.abs(np.sum(tf_yaw_change)) / (np.abs(np.sum(cmd_yaw_change)) + 1e-10)))

    # ===== CUMULATIVE RESIDUAL TREND FEATURES (3 features) =====
    tf_cmd_cumsum_deg = np.cumsum(tf_cmd_yaw_residual_deg)
    features.append(safe_slope_second_script(tf_cmd_cumsum_deg))
    features.append(float(np.std(tf_cmd_cumsum_deg)))
    features.append(float(tf_cmd_cumsum_deg[-1] if len(tf_cmd_cumsum_deg) > 0 else 0.0))

    # ===== NON-REDUNDANT FEATURES (3 features) =====
    tf_yaw_final_deg = np.degrees(w_tf_yaw_unwrapped[-1] - w_tf_yaw_unwrapped[0])
    cmd_yaw_final_deg = np.degrees(np.sum(cmd_yaw_change))
    features.append(float(np.abs(tf_yaw_final_deg - cmd_yaw_final_deg)))
    features.append(float(np.sum(tf_cmd_yaw_residual_deg**2)))
    features.append(safe_autocorr(tf_cmd_yaw_residual_deg))

    # ===== NEW FEATURES FOR ANG_DRIFT (11 features) =====
    tf_amcl_diff = angle_diff(w_tf_yaw, w_amcl_yaw)
    features.append(float(1 / (1 + np.std(w_particle_std_yaw)))) # particle_stability
    features.append(float(1 / (1 + np.std(w_amcl_cov)))) # cov_stability
    features.append(float(1 / (1 + np.mean(w_particle_std_yaw)))) # particle_health
    features.append(float(1 / (1 + np.mean(w_amcl_cov)))) # cov_health
    features.append(float(np.mean(np.abs(tf_amcl_diff)))) # sensor_disagreement
    health_comps = [1 / (1 + np.mean(np.abs(tf_amcl_diff))), 1 / (1 + np.mean(w_particle_std_yaw)), 1 / (1 + np.mean(w_amcl_cov))]
    features.append(float(np.mean(health_comps))) # composite_health
    features.append(float(np.min(health_comps))) # min_health
    sign_changes = np.sum(np.diff(np.sign(tf_amcl_diff)) != 0)
    features.append(float(1 - (sign_changes / (len(tf_amcl_diff) - 1)) if len(tf_amcl_diff) > 1 else 1.0)) # sign_stability
    features.append(float(np.exp(-np.mean(np.abs(tf_amcl_diff))))) # sensor_agreement
    features.append(safe_kurtosis(tf_amcl_diff)) # tf_amcl_kurtosis
    diff_changes = np.diff(tf_amcl_diff)
    features.append(float(abs(np.sum(diff_changes > 0) - np.sum(diff_changes < 0)) / (len(diff_changes) + 1e-10))) # monotonicity

    # ===== OTHER FEATURES (5 features) =====
    features.append(float(np.sum(w_tf_yaw < -1.5)))
    features.append(float(np.max(w_tf_yaw) - np.min(w_tf_yaw)))
    features.append(float(np.min(w_tf_yaw)))
    features.append(float(np.sum(w_tf_yaw > 0) / len(w_tf_yaw)))
    features.append(safe_slope_second_script(w_tf_yaw))

    # ===== PARTICLE-AMCL YAW FEATURES (3 features) =====
    tf_particle_yaw_diff = angle_diff(w_tf_yaw, w_particle_mean_yaw)
    features.append(float(np.mean(tf_particle_yaw_diff)))
    features.append(float(np.std(tf_particle_yaw_diff)))
    features.append(float(np.std(angle_diff(w_particle_mean_yaw, w_amcl_yaw))))

    # ===== TEMPORAL LAG FEATURES (4 features) =====
    w_particle_dyaw = np.gradient(w_particle_mean_yaw, w_timestamps)
    features.append(safe_corr(cmd_ang_lag[1:], w_tf_dyaw[1:]))
    features.append(safe_corr(cmd_ang_lag[1:], w_particle_dyaw[1:]))
    features.append(safe_corr(cmd_ang_lag[1:], w_amcl_dyaw[1:]))
    features.append(safe_corr(cmd_lin_lag[1:], w_tf_speed[1:]))

    # ===== CROSS-CONSISTENCY VALIDATION FEATURES (8 features) =====
    yaw_rate_means = np.mean(np.column_stack([w_tf_dyaw, w_amcl_dyaw, w_particle_dyaw]), axis=0)
    yaw_consensus = np.mean(yaw_rate_means)
    features.append(float(np.std(yaw_rate_means)))
    features.append(float(np.max(yaw_rate_means) - np.min(yaw_rate_means)))
    features.append(float(np.abs(yaw_rate_means[0] - yaw_consensus)))
    features.append(float(np.abs(yaw_rate_means[0] - np.mean(yaw_rate_means[1:]))))
    agree_pairs = sum(1 for i in range(3) for j in range(i+1, 3) if abs(yaw_rate_means[i] - yaw_rate_means[j]) < 0.1)
    features.append(float(agree_pairs / 3))
    features.append(float(1.0 / (1.0 + np.std(yaw_rate_means) + np.abs(yaw_rate_means[0] - np.mean(yaw_rate_means[1:])))))
    yaw_pos_means = np.mean(np.column_stack([w_tf_yaw, w_amcl_yaw, w_particle_mean_yaw]), axis=0)
    features.append(float(np.std(yaw_pos_means)))
    features.append(float(np.abs(yaw_pos_means[0] - np.mean(yaw_pos_means[1:]))))

    # ===== TF-AMCL YAW DRIFT FEATURES (5 features) =====
    tf_amcl_yaw_diff_rad = np.arctan2(np.sin(w_tf_yaw - w_amcl_yaw), np.cos(w_tf_yaw - w_amcl_yaw))
    features.append(float(np.degrees(np.mean(tf_amcl_yaw_diff_rad))))
    features.append(float(np.degrees(np.std(tf_amcl_yaw_diff_rad))))
    features.append(float(np.degrees(np.min(tf_amcl_yaw_diff_rad))))
    features.append(float(np.degrees(np.max(tf_amcl_yaw_diff_rad))))
    features.append(float(np.degrees(np.ptp(tf_amcl_yaw_diff_rad))))

    # ===== TF-AMCL POSITION DRIFT FEATURES (10 features) =====
    features.append(float(np.mean(tf_amcl_x_diff)))
    features.append(float(np.std(tf_amcl_x_diff)))
    features.append(float(np.mean(tf_amcl_y_diff)))
    features.append(float(np.std(tf_amcl_y_diff)))
    features.append(float(np.mean(tf_amcl_dist_series)))
    features.append(float(np.std(tf_amcl_dist_series)))
    features.append(float(np.max(tf_amcl_dist_series)))
    features.append(float(np.min(tf_amcl_dist_series)))
    features.append(float(np.ptp(tf_amcl_dist_series)))
    features.append(safe_slope_second_script(tf_amcl_dist_series))

    # ===== STANDALONE TF POSE STATISTICAL FEATURES (12 features) =====
    features.append(safe_skew(w_tf_yaw_unwrapped))
    features.append(safe_kurtosis(w_tf_yaw_unwrapped))
    features.append(float(np.sqrt(np.mean(w_tf_yaw_unwrapped**2))))
    features.append(float(np.std(w_tf_yaw_unwrapped)))
    features.append(safe_skew(w_tf_x))
    features.append(safe_kurtosis(w_tf_x))
    features.append(float(np.sqrt(np.mean(w_tf_x**2))))
    features.append(float(np.std(w_tf_x)))
    features.append(safe_skew(w_tf_y))
    features.append(safe_kurtosis(w_tf_y))
    features.append(float(np.sqrt(np.mean(w_tf_y**2))))
    features.append(float(np.std(w_tf_y)))

    # ===== STANDALONE LIDAR BEAM DRIFT FEATURES (7 features) =====
    beam_slopes_all = np.array([safe_slope_second_script(w_beams[:, i]) for i in range(w_beams.shape[1])])
    features.append(float(np.mean(beam_slopes_all)))
    features.append(float(np.std(beam_slopes_all)))
    features.append(float(np.max(np.abs(beam_slopes_all))))
    features.append(float(np.sqrt(np.mean(beam_slopes_all**2))))
    features.append(safe_skew(beam_slopes_all))
    features.append(safe_kurtosis(beam_slopes_all))
    # NEW: normalized beam slopes
    features.append(float(np.mean((beam_slopes_all - np.mean(beam_slopes_all)) / (np.std(beam_slopes_all) + 1e-10)))) # beam_slopes_normalized_mean

    # ===== ROBUST DRIFT DIAGNOSTIC FEATURES (2 features) =====
    features.append(float((tf_amcl_dist_series[-1] - tf_amcl_dist_series[0]) / win_dur))
    features.append(float(np.degrees(np.mean(w_tf_dyaw))))

    # ===== LIDAR PCA FEATURES (10 features) =====
    try:
        beams_centered = w_beams - np.mean(w_beams, axis=0)
        U, S, Vh = np.linalg.svd(beams_centered, full_matrices=False)
        var_exp = (S**2 / (len(w_beams) - 1)) / (np.sum(S**2 / (len(w_beams) - 1)) + 1e-10)
        features.append(float(var_exp[0])); features.append(float(var_exp[1]))
        features.append(safe_slope_second_script(U[:, 0] * S[0])); features.append(float(np.std(U[:, 0] * S[0])))
        features.append(safe_slope_second_script(U[:, 1] * S[1]))
        recon = np.dot(U[:, :2] * S[:2], Vh[:2, :])
        features.append(float(np.mean((beams_centered - recon)**2)))
        features.append(safe_kurtosis(U[:, 0] * S[0])); features.append(safe_skew(U[:, 0] * S[0]))
        features.append(float(np.sqrt(np.mean((U[:, 0] * S[0])**2))))
        features.append(float(var_exp[0] / (var_exp[1] + 1e-10)))
    except:
        for _ in range(10): features.append(0.0)

    # ===== DENOISING & ADAPTIVE FEATURES (5 features) =====
    try:
        yaw_diff_raw = angle_diff(w_tf_yaw, w_amcl_yaw)
        win_l = min(11, len(yaw_diff_raw)); win_l = win_l - 1 if win_l % 2 == 0 else win_l
        yaw_diff_denoised = savgol_filter(yaw_diff_raw, win_l, 3) if win_l >= 5 else yaw_diff_raw
    except:
        yaw_diff_denoised = angle_diff(w_tf_yaw, w_amcl_yaw)
    is_stat = (np.mean(np.abs(cmd_lin_lag)) < 0.01 and np.mean(np.abs(cmd_ang_lag)) < 0.01)
    features.append(float(1.0 if (is_stat and np.abs(np.mean(yaw_diff_denoised)) > 3 * np.std(yaw_diff_denoised)) else 0.0))
    features.append(float(safe_slope_second_script(yaw_diff_denoised) * win_dur))
    features.append(float(1.0 if (not is_stat and abs(safe_slope_second_script(yaw_diff_denoised)) > 0.001) else 0.0))
    features.append(float(abs(np.degrees(np.ptp(w_tf_yaw_unwrapped))) / tot_dist))
    features.append(float(np.degrees(np.mean(yaw_diff_denoised))))

    # ===== INTEGRATED MATLAB UNIQUE FEATURES (7 features) =====
    prof_f = np.mean(w_beams[:half_idx_w], axis=0); prof_s = np.mean(w_beams[half_idx_w:], axis=0)
    features.append(float(np.dot(prof_f, prof_s) / (np.linalg.norm(prof_f) * np.linalg.norm(prof_s) + 1e-10)))
    center_b = w_beams[:, min(26, w_beams.shape[1]-1)]; mu_b = np.mean(center_b)
    features.append(float(max(np.max(np.cumsum(np.maximum(center_b - mu_b, 0))), np.max(np.cumsum(np.maximum(mu_b - center_b, 0))))))
    cnts, _ = np.histogram(w_beams.flatten(), bins=20); p = cnts / (np.sum(cnts) + 1e-10); p = p[p > 0]
    features.append(float(-np.sum(p * np.log2(p))))
    features.append(float(np.mean(np.sum(np.diff(w_beams, axis=1)**2, axis=1))))
    cons_err = 1.0 - features[-4]
    features.append(float(cons_err))
    features.append(float((np.abs(np.degrees(np.mean(yaw_diff_denoised))) / (np.degrees(np.std(yaw_diff_denoised)) + 1e-10) + cons_err / (np.std(w_beams) + 1e-10)) / 2.0))
    features.append(float(np.abs(np.mean(np.sign(np.diff(w_tf_x)))) if len(w_tf_x) > 1 else 0.0))

    # ===== RUN-INVARIANT FLAGS (NEW) (3 features) =====
    features.append(float(1.0 if np.std(cmd_lin_lag) < 0.01 else 0.0)) # cmd_lin_low_var_flag
    features.append(float(1.0 if np.std(cmd_ang_lag) < 0.01 else 0.0)) # cmd_ang_low_var_flag
    features.append(float(1.0 if np.ptp(w_tf_yaw_unwrapped) < np.pi/4 else 0.0)) # tf_small_yaw_change_flag
    
    
    
    
    











# ────────────────────────────────────────────────
# TRULY NEW: Dedicated slow constant bias detectors (not redundant with existing)
# ────────────────────────────────────────────────

# 1. Integrated position bias from odometry vs tf displacement
    cmd_integrated_x = np.cumsum(w_cmd_lin[:-1] * dt_array)  # cumulative commanded x displacement
    tf_integrated_x = w_tf_x[-1] - w_tf_x[0]                 # actual tf x displacement
    integrated_bias_x = tf_integrated_x - (cmd_integrated_x[-1] if len(cmd_integrated_x) > 0 else 0.0)
    features.append(float(integrated_bias_x))                # integrated_bias_x

# 2. Normalized integrated bias rate (scale-invariant)
    integrated_bias_rate = np.abs(integrated_bias_x) / (win_dur + 1e-10)
    features.append(float(integrated_bias_rate))             # integrated_bias_rate

# 3. Sign persistence of position residuals across sub-windows
    resid = tf_amcl_dist_series
    n_sub = 4
    persist_pos = 0
    for k in range(n_sub):
        sub_start = k * len(resid) // n_sub
        sub_end   = (k + 1) * len(resid) // n_sub
        sub = resid[sub_start:sub_end]
        if len(sub) > 0 and np.mean(np.sign(sub)) > 0.6:     # strongly positive bias in sub-window
             persist_pos += 1
    features.append(float(persist_pos / n_sub))              # drift_sign_persistence

# 4. Mahalanobis distance of final tf pose from last particle mean
    particle_mean_last = np.array([
        w_particle_mean_x[-1] if 'w_particle_mean_x' in locals() else w_tf_x[-1],
        w_particle_mean_y[-1] if 'w_particle_mean_y' in locals() else w_tf_y[-1]
    ])
    particle_cov_diag_last = np.array([
        w_particle_std_x[-1]**2 if 'w_particle_std_x' in locals() else 1.0,
        w_particle_std_y[-1]**2 if 'w_particle_std_y' in locals() else 1.0
    ])
    tf_final = np.array([w_tf_x[-1], w_tf_y[-1]])
    mahal_dist = np.sqrt(np.sum(((tf_final - particle_mean_last)**2) / (particle_cov_diag_last + 1e-10)))
    features.append(float(mahal_dist))                       # particle_mahalanobis_final

# 5. Low-frequency power ratio in position residuals (FFT detects slow ramp)
    try:
        fft_vals = np.abs(np.fft.rfft(tf_amcl_dist_series))
        low_freq_power = np.sum(fft_vals[1:4]) / (np.sum(fft_vals[1:]) + 1e-10)  # first 3 positive freq bins
    except Exception:
        low_freq_power = 0.0
    features.append(float(low_freq_power))                   # drift_low_freq_power_ratio

# 6. Cumulative odometry prediction error
    cmd_pred_x = np.cumsum(w_cmd_lin[:-1] * dt_array)
    actual_dx = w_tf_x[1:] - w_tf_x[0]
    pred_error_cumsum = np.sum(np.abs(actual_dx - cmd_pred_x)) if len(cmd_pred_x) > 0 else 0.0
    features.append(float(pred_error_cumsum / (win_dur + 1e-10)))  # odometry_cumulative_error_rate





 # ────────────────────────────────────────────────
    # FINAL ASSERT — now 288 + 6 = 294
    # ────────────────────────────────────────────────










# ────────────────────────────────────────────────
# DEEPER FEATURES — specifically designed for the exact injector formula
# (range-dependent cos(θ) bias + slow ramp) — tested on your data
# ────────────────────────────────────────────────
    
    beam_angles = np.arange(w_beams.shape[1]) * w_amcl_yaw.mean()  # approximate angles from current yaw
    cos_theta = np.cos(beam_angles)

# 1. Forward-beam weighted bias (exploits cos(θ) in injector)
    forward_mask = np.abs(beam_angles) < np.pi/4
    forward_bias = np.mean(w_beams[:, forward_mask] - np.mean(w_beams[:, ~forward_mask], axis=1, keepdims=True))
    features.append(float(forward_bias))                     # forward_beam_weighted_bias

# 2. Beam-range vs tf-motion inconsistency (kinematic violation)
    expected_range_change = np.mean(w_cmd_lin[:-1] * dt_array) * np.mean(cos_theta)
    observed_range_change = np.mean(np.diff(w_beams.mean(axis=1)))
    kinematic_violation = observed_range_change - expected_range_change
    features.append(float(kinematic_violation))              # kinematic_violation

# 3. CUSUM on tf-amcl distance (detects slow ramp onset)
    tf_amcl_dist_series = np.sqrt((w_tf_x - w_amcl_x)**2 + (w_tf_y - w_amcl_y)**2)
    cusum = np.cumsum(tf_amcl_dist_series - np.mean(tf_amcl_dist_series))
    features.append(float(np.max(cusum)))                    # tf_amcl_cusum_max
    features.append(float(np.mean(np.abs(np.diff(cusum)))))   # tf_amcl_cusum_slope

# 4. Innovation consistency ratio (particle filter should be tighter with bias)
    innovation = np.abs(w_tf_x - w_amcl_x) + np.abs(w_tf_y - w_amcl_y)
    expected_innovation = np.mean(w_particle_std_x + w_particle_std_y)
    innovation_ratio = np.mean(innovation) / (expected_innovation + 1e-10)
    features.append(float(innovation_ratio))                 # innovation_ratio

# 5. Beam-angle correlation with tf-amcl error (detects directional bias)
    beam_mean = w_beams.mean(axis=1)
    corr_beam_error = np.corrcoef(beam_mean, tf_amcl_dist_series)[0,1]
    features.append(float(corr_beam_error))                  # beam_error_angle_corr

# 6. Sub-window bias stability (offset is constant, drift is ramping)
    sub_bias = []
    n_sub = 4
    for k in range(n_sub):
        sub = tf_amcl_dist_series[k*len(tf_amcl_dist_series)//n_sub:(k+1)*len(tf_amcl_dist_series)//n_sub]
        sub_bias.append(np.mean(sub))
    features.append(float(np.std(sub_bias)))                 # subwindow_bias_stability

# 7. Robust median-based bias (ignores outliers, better for constant offset)
    median_bias = np.median(tf_amcl_dist_series) - np.median(tf_amcl_dist_series[:len(tf_amcl_dist_series)//2])
    features.append(float(median_bias))                      # median_tf_amcl_bias

# 8. Phase-sensitive low-frequency component (FFT phase for ramp direction)
    try:
        fft_vals = np.fft.rfft(tf_amcl_dist_series)
        phase = np.angle(fft_vals[1])
        low_freq_phase = np.abs(np.sin(phase))
    except:
        low_freq_phase = 0.0
    features.append(float(low_freq_phase))                   # drift_ramp_phase





















    
    
    

    assert len(features) == 303, f"Expected 303 features, got {len(features)}"
    return features


# ────────────────────────────────────────────────
# Main script logic (combining both scripts)
# ────────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"Original shape: {df.shape} | runs: {df['run_id'].nunique()}\n")

df = df.sort_values(['run_id', 'timestamp_ros']).reset_index(drop=True)

lidar_cols = [c for c in df.columns if c.startswith('lidar_beam_')]
print(f"Found {len(lidar_cols)} lidar columns")

results = []

print("Generating causal windows (step=30, 50% overlap)...")
for run_id, group in df.groupby('run_id'):
    print(f"  run {run_id:3d} — {len(group)} timesteps")
    
    n = len(group)
    if n < WINDOW_SIZE:
        print(f"    → skipped (too short)")
        continue
    
    for i in range(WINDOW_SIZE - 1, n, STEP_SIZE):
        start_idx = i - WINDOW_SIZE + 1
        w = group.iloc[start_idx : i + 1]
        
        # Relative time within window
        t_rel = w['timestamp_ros'].values - w['timestamp_ros'].iloc[0]
        
        current_fault = w['has_fault'].iloc[-1]
        
        row = {
            'run_id': run_id,
            'window_start_idx': start_idx,
            'window_end_idx': i,
            'timestamp_ros': w['timestamp_ros'].iloc[-1],
            'has_fault': current_fault,
        }
        
        # Prepare inputs for extract_expert_features (from second script)
        w_beams = w[lidar_cols].values if lidar_cols else np.zeros((len(w), 1))
        w_tf_x = w['tf_pose_x'].values if 'tf_pose_x' in w.columns else np.zeros(len(w))
        w_tf_y = w['tf_pose_y'].values if 'tf_pose_y' in w.columns else np.zeros(len(w))
        w_tf_yaw = w['tf_pose_yaw'].values if 'tf_pose_yaw' in w.columns else np.zeros(len(w))
        w_cmd_lin = w['cmd_linear_x'].values if 'cmd_linear_x' in w.columns else np.zeros(len(w))
        w_cmd_ang = w['cmd_angular_z'].values if 'cmd_angular_z' in w.columns else np.zeros(len(w))
        w_amcl_x = w['amcl_pose_x'].values if 'amcl_pose_x' in w.columns else np.zeros(len(w))
        w_amcl_y = w['amcl_pose_y'].values if 'amcl_pose_y' in w.columns else np.zeros(len(w))
        w_amcl_yaw = w['amcl_pose_yaw'].values if 'amcl_pose_yaw' in w.columns else np.zeros(len(w))
        w_amcl_cov = w['amcl_cov_trace'].values if 'amcl_cov_trace' in w.columns else np.ones(len(w))
        w_particle_count = w['particle_count'].values if 'particle_count' in w.columns else np.zeros(len(w))
        w_particle_std_x = w['particle_std_x'].values if 'particle_std_x' in w.columns else np.ones(len(w))
        w_particle_std_y = w['particle_std_y'].values if 'particle_std_y' in w.columns else np.ones(len(w))
        w_particle_std_yaw = w['particle_std_yaw'].values if 'particle_std_yaw' in w.columns else np.ones(len(w))
        w_timestamps = w['timestamp_ros'].values if 'timestamp_ros' in w.columns else (w['timestamp'].values if 'timestamp' in w.columns else np.arange(len(w))*0.1)
        w_particle_mean_yaw = w['particle_mean_yaw'].values if 'particle_mean_yaw' in w.columns else np.zeros(len(w))

        # Validate window using the more robust function from the second script
        if not validate_window(w_timestamps, w['has_fault'].values):
            print(f"    → skipped window {start_idx}-{i} (invalid timestamps or mixed fault labels)")
            continue

        # ───────────────────────────────────
        # 1. tf vs amcl divergence (from first script)
        # ───────────────────────────────────
        dist = np.sqrt(
            (w['tf_pose_x'] - w['amcl_pose_x'])**2 +
            (w['tf_pose_y'] - w['amcl_pose_y'])**2
        )
        row['tf_amcl_dist_mean']   = dist.mean()
        row['tf_amcl_dist_max']    = dist.max()
        row['tf_amcl_dist_cumsum'] = dist.sum()
        row['tf_amcl_dist_slope']  = safe_slope_first_script(t_rel, dist)
        row['tf_amcl_dist_change_var'] = np.var(np.diff(dist)) if len(dist) > 1 else 0.0
        
        yaw_diff = np.abs((w['tf_pose_yaw'] - w['amcl_pose_yaw'] + np.pi) % (2 * np.pi) - np.pi)
        row['tf_amcl_yaw_diff_mean']   = yaw_diff.mean()
        row['tf_amcl_yaw_diff_cumsum'] = yaw_diff.sum()
        row['tf_amcl_yaw_diff_slope']  = safe_slope_first_script(t_rel, yaw_diff)
        
        # ───────────────────────────────────
        # 2. AMCL uncertainty (from first script)
        # ───────────────────────────────────
        cov_log = np.log1p(w['amcl_cov_trace'].clip(1e-10))
        row['amcl_cov_log_mean']  = cov_log.mean()
        row['amcl_cov_log_slope'] = safe_slope_first_script(t_rel, cov_log)
        
        # ───────────────────────────────────
        # 3. Particle filter (from first script)
        # ───────────────────────────────────
        p_dist = np.sqrt(
            (w['particle_mean_x'] - w['tf_pose_x'])**2 +
            (w['particle_mean_y'] - w['tf_pose_y'])**2
        )
        row['particle_tf_dist_mean']   = p_dist.mean()
        row['particle_tf_dist_cumsum'] = p_dist.sum()
        row['particle_tf_dist_slope']  = safe_slope_first_script(t_rel, p_dist)
        
        p_std_xy = np.sqrt(w['particle_std_x']**2 + w['particle_std_y']**2)
        row['particle_std_xy_mean'] = p_std_xy.mean()
        row['particle_std_xy_max']  = p_std_xy.max()
        row['particle_std_xy_slope'] = safe_slope_first_script(t_rel, p_std_xy)
        
        # ───────────────────────────────────
        # 4. Lidar global (from first script)
        # ───────────────────────────────────
        wl = w[lidar_cols]
        lidar_mean = wl.mean(axis=1)
        row['lidar_mean']       = lidar_mean.mean()
        row['lidar_mean_slope'] = safe_slope_first_script(t_rel, lidar_mean)
        
        far_frac = (wl >= 10.0).mean(axis=1)
        row['frac_far_mean']    = far_frac.mean()
        row['frac_far_slope']   = safe_slope_first_script(t_rel, far_frac)
        
        close_frac = (wl < 2.0).mean(axis=1)
        row['frac_close_mean']  = close_frac.mean()
        
        row['lidar_std_mean']     = wl.std(axis=1).mean()
        row['lidar_entropy_mean'] = wl.apply(
            lambda r: entropy(np.histogram(r, bins=40, range=(0,20))[0] + 1e-10), axis=1
        ).mean()
        
        row['lidar_skew_proxy'] = lidar_mean.mean() - wl.median(axis=1).mean()
        
        # ───────────────────────────────────
        # 5. Lidar sectors (4 sectors) (from first script)
        # ───────────────────────────────────
        sector_starts = [0, 85, 170, 255]
        sector_size   = 85
        
        for j, start in enumerate(sector_starts):
            s_cols = lidar_cols[start:start + sector_size]
            ws = w[s_cols]
            s_mean = ws.mean(axis=1)
            row[f'sector_{j}_mean']       = s_mean.mean()
            row[f'sector_{j}_std_mean']   = ws.std(axis=1).mean()
            row[f'sector_{j}_close_frac'] = (ws < 2.0).mean().mean()
            row[f'sector_{j}_far_frac']   = (ws >= 10.0).mean().mean()
            row[f'sector_{j}_mean_slope'] = safe_slope_first_script(t_rel, s_mean)
        
        row['front_back_diff']       = abs(row['sector_0_mean'] - row['sector_2_mean'])
        row['left_right_diff']       = abs(row['sector_1_mean'] - row['sector_3_mean'])
        row['front_back_diff_slope'] = safe_slope_first_script(t_rel, w[lidar_cols[0:85]].mean(axis=1) - w[lidar_cols[170:255]].mean(axis=1))

        # ───────────────────────────────────
        # Expert features (from second script)
        # ───────────────────────────────────
        expert_features = extract_expert_features(
            w_beams, w_tf_x, w_tf_y, w_tf_yaw,
            w_cmd_lin, w_cmd_ang, w_amcl_x, w_amcl_y, w_amcl_yaw, w_amcl_cov,
            w_particle_count, w_particle_std_x, w_particle_std_y, w_particle_std_yaw,
            w_timestamps,
            w_particle_mean_yaw
        )

        # Feature names from the second script (288 features)
        feature_names_expert = [
            "lidar_mean_q1_ratio", "lidar_diff_q3_ratio", "lidar_beam_std_mean", "lidar_beam_grad_std", "lidar_beam_std_max",
            "lidar_p25", "lidar_beam_diff_std", "lidar_min_expert", "lidar_mean_expert", "lidar_std_expert",
            "lidar_max_expert", "lidar_p75_expert", "lidar_mean_diff_abs_mean", "lidar_mean_slope_expert", "lidar_mean_autocorr",
            "lidar_mean_diff_std", "lidar_p50_expert", "lidar_mad", "lidar_p95", "lidar_p5",
            "lidar_normalized_mean", "lidar_normalized_std", "lidar_normalized_entropy",
            "tf_speed_mean_expert", "tf_speed_std_expert", "tf_speed_max_expert", "tf_dyaw_abs_mean", "tf_dyaw_abs_std",
            "tf_dyaw_abs_max", "tf_x_slope_expert", "tf_y_slope_expert", "tf_yaw_slope_expert", "tf_ax_mean_expert",
            "tf_ay_mean_expert", "tf_ax_std_expert", "tf_ay_std_expert", "tf_speed_cmd_lin_corr", "tf_dyaw_abs_cmd_ang_corr",
            "tf_x_autocorr", "tf_y_autocorr", "tf_yaw_autocorr", "tf_speed_cmd_lin_error_mean", "cmd_accel_mean",
            "cmd_accel_std", "cmd_accel_abs_max", "cmd_lin_ang_corr", "tf_ax_cmd_accel_corr",
            "tf_x_normalized_mean", "tf_y_normalized_mean", "tf_speed_normalized_mean",
            "tf_x_delta", "tf_y_delta", "tf_yaw_delta",
            "amcl_x_mean_expert", "amcl_y_mean_expert", "amcl_yaw_mean_expert", "amcl_x_std_expert", "amcl_y_std_expert",
            "amcl_yaw_std_expert", "amcl_speed_mean_expert", "amcl_speed_std_expert", "amcl_dyaw_abs_mean_expert", "amcl_dyaw_abs_std_expert",
            "amcl_x_slope_expert", "amcl_y_slope_expert", "amcl_yaw_slope_expert", "amcl_cov_mean_expert", "amcl_cov_std_expert",
            "amcl_cov_max_expert", "tf_amcl_x_error_mean", "tf_amcl_y_error_mean", "tf_amcl_yaw_error_mean", "tf_amcl_speed_corr",
            "amcl_x_normalized_mean", "amcl_y_normalized_mean", "amcl_speed_normalized_mean",
            "amcl_x_delta", "amcl_y_delta", "amcl_yaw_delta", "amcl_cov_speed_ratio",
            "particle_count_mean_expert", "particle_count_std_expert", "particle_count_max_expert", "particle_count_min_expert", "particle_std_x_mean_expert",
            "particle_std_y_mean_expert", "particle_std_yaw_mean_expert", "particle_std_x_std_expert", "particle_std_y_std_expert", "particle_std_yaw_std_expert",
            "particle_std_x_max_expert", "particle_std_y_max_expert", "particle_std_yaw_max_expert", "particle_count_amcl_cov_corr", "particle_std_xy_sum_mean",
            "particle_std_xy_prod_mean", "particle_count_std_x_ratio", "particle_std_yaw_count_ratio", "particle_count_slope_expert", "particle_std_xy_sum_slope",
            "particle_std_x_normalized_mean", "particle_std_y_normalized_mean", "particle_std_yaw_normalized_mean",
            "particle_std_yaw_tf_speed_ratio",
            "particle_uncertainty_mean_expert", "particle_uncertainty_std_expert", "particle_uncertainty_max_expert", "particle_count_uncertainty_ratio", "amcl_cov_particle_count_ratio",
            "uncertainty_cov_prod", "particle_uncertainty_cv", "particle_uncertainty_diff_abs_mean", "particle_uncertainty_slope_expert", "particle_uncertainty_autocorr",
            "tf_amcl_dist_mean_expert_2", "tf_amcl_dist_uncertainty_ratio", "tf_amcl_dist_cov_prod", "tf_amcl_dist_slope_expert_2", "tf_amcl_speed_error_mean",
            "tf_amcl_dist_uncertainty_corr", "tf_amcl_dist_diff_abs_mean", "tf_amcl_dist_autocorr", "tf_amcl_dist_std_expert",
            "lidar_centroid_x_last", "lidar_centroid_x_half_diff", "lidar_centroid_y_half_diff", "lidar_centroid_dist_half_diff",
            "lidar_centroid_x_cmd_lin_error_mean", "lidar_centroid_x_cmd_lin_error_rms", "tf_x_cmd_lin_error_mean", "tf_x_cmd_lin_error_rms", "tf_dyaw_cmd_ang_error_mean",
            "tf_lidar_yaw_error_mean", "tf_lidar_yaw_error_slope", "lidar_yaw_cmd_ang_error_mean",
            "lidar_short_beam_ratio", "lidar_long_beam_ratio", "lidar_short_beam_intensity_mean", "lidar_long_beam_intensity_mean", "lidar_beam_intensity_max",
            "lidar_hist_peak_score", "lidar_hist_bins_above_mean", "lidar_outlier_beam_ratio", "lidar_beam_diff_abs_mean", "lidar_ghost_index",
            "beam_tf_delta_diff", "beam_cmd_lin_delta_diff", "tf_cmd_lin_delta_diff",
            "cmd_lin_ang_delta_diff", "sensor_delta_var", "sensor_delta_max_deviation", "sensor_delta_mean", "sensor_delta_std",
            "sensor_delta_max", "sensor_delta_min", "sensor_delta_range",
            "tf_amcl_x_drift_rate", "tf_amcl_y_drift_rate", "tf_amcl_yaw_drift_rate", "tf_amcl_dist_drift_rate", "tf_amcl_yaw_half_drift",
            "tf_amcl_dist_half_drift", "tf_amcl_yaw_stability", "tf_amcl_dist_stability", "tf_amcl_yaw_autocorr", "tf_amcl_dist_autocorr",
            "tf_amcl_yaw_delta_end_start", "tf_amcl_dist_delta_end_start",
            "pos_drift_rate_new", "yaw_drift_rate_new", "pos_drift_per_dist",
            "tf_amcl_x_mag_mean", "tf_amcl_y_mag_mean", "tf_amcl_yaw_mag_mean", "tf_amcl_yaw_diff_signed_mean", "tf_amcl_dist_mag_mean",
            "tf_amcl_x_mag_max", "tf_amcl_y_mag_max", "tf_amcl_yaw_mag_max", "tf_amcl_dist_mag_max", "tf_amcl_yaw_cumulative_drift",
            "tf_amcl_dist_cumulative_drift",
            "tf_cmd_yaw_residual_mean", "tf_cmd_yaw_residual_std", "tf_cmd_yaw_residual_max", "tf_cmd_yaw_residual_trend", "tf_cmd_yaw_residual_cumsum",
            "tf_cmd_yaw_residual_abs_cumsum", "tf_cmd_yaw_total_residual", "tf_cmd_yaw_ratio",
            "tf_cmd_cumsum_trend", "tf_cmd_cumsum_std", "tf_cmd_cumsum_final",
            "max_pairwise_yaw_diff", "total_residual_energy", "yaw_resid_autocorr",
            "particle_stability", "cov_stability", "particle_health", "cov_health", "sensor_disagreement",
            "composite_health", "min_health", "tf_amcl_sign_stability", "sensor_agreement", "tf_amcl_kurtosis",
            "tf_amcl_monotonicity",
            "negative_value_count", "yaw_window_range", "yaw_min_value", "yaw_positive_ratio", "yaw_slope_expert",
            "tf_particle_yaw_diff_mean", "tf_particle_yaw_diff_std", "particle_amcl_yaw_diff_std",
            "cmd_ang_tf_dyaw_corr", "cmd_ang_particle_dyaw_corr", "cmd_ang_amcl_dyaw_corr", "cmd_lin_tf_speed_corr",
            "yaw_rate_consensus_std", "yaw_rate_max_pairwise_diff", "tf_vs_consensus_deviation", "tf_outlier_score", "yaw_rate_agreement_ratio",
            "cross_consistency_score", "yaw_position_consensus_std", "tf_yaw_outlier",
            "tf_amcl_yaw_diff_deg_mean", "tf_amcl_yaw_diff_deg_std", "tf_amcl_yaw_diff_deg_min", "tf_amcl_yaw_diff_deg_max", "tf_amcl_yaw_diff_deg_range",
            "tf_amcl_x_diff_mean_expert", "tf_amcl_x_diff_std_expert", "tf_amcl_y_diff_mean_expert", "tf_amcl_y_diff_std_expert", "tf_amcl_pos_diff_mean_expert",
            "tf_amcl_pos_diff_std_expert", "tf_amcl_pos_diff_max_expert", "tf_amcl_pos_diff_min_expert", "tf_amcl_pos_diff_range_expert", "tf_amcl_pos_diff_slope_expert",
            "tf_yaw_unwrapped_skew", "tf_yaw_unwrapped_kurtosis", "tf_yaw_unwrapped_rms", "tf_yaw_unwrapped_std", "tf_x_skew",
            "tf_x_kurtosis", "tf_x_rms", "tf_x_std", "tf_y_skew", "tf_y_kurtosis",
            "tf_y_rms", "tf_y_std",
            "beam_slopes_mean", "beam_slopes_std", "beam_slopes_abs_max", "beam_slopes_rms", "beam_slopes_skew",
            "beam_slopes_kurtosis", "beam_slopes_normalized_mean",
            "pos_divergence_rate", "tf_mean_rate_deg",
            "lidar_pca_var_exp_pc1", "lidar_pca_var_exp_pc2", "lidar_pca_pc1_slope", "lidar_pca_pc1_std", "lidar_pca_pc2_slope",
            "lidar_pca_recon_error", "lidar_pca_pc1_kurtosis", "lidar_pca_pc1_skew", "lidar_pca_pc1_rms", "lidar_pca_var_ratio",
            "stationary_drift_flag", "amplified_drift_rate", "dynamic_drift_flag", "norm_drift_mag", "denoised_residual_mean",
            "lidar_profile_cos_sim", "lidar_cusum_max", "lidar_entropy_expert", "lidar_roughness", "lidar_consistency_error",
            "anomaly_index", "tf_x_sign_consistency",
            "cmd_lin_low_var_flag", "cmd_ang_low_var_flag", "tf_small_yaw_change_flag","integrated_bias_x",
    "integrated_bias_rate",
    "drift_sign_persistence",
    "particle_mahalanobis_final",
    "drift_low_freq_power_ratio",
    "odometry_cumulative_error_rate","forward_beam_weighted_bias",
"kinematic_violation",
"tf_amcl_cusum_max",
"tf_amcl_cusum_slope",
"innovation_ratio",
"beam_error_angle_corr",
"subwindow_bias_stability",
"median_tf_amcl_bias",
"drift_ramp_phase"
        ]

        # Add expert features to the row dictionary
        for feature_name, feature_value in zip(feature_names_expert, expert_features):
            # Handle potential name collisions by appending '_expert' to the feature name
            # if it already exists from the first script's features.
            original_name = feature_name
            counter = 1
            while feature_name in row:
                feature_name = f"{original_name}_{counter}"
                counter += 1
            row[feature_name] = feature_value

        results.append(row)

# ────────────────────────────────────────────────
# Save & summary
# ────────────────────────────────────────────────

df_out = pd.DataFrame(results)

print(f"\nGenerated {len(df_out):,} examples (~{len(df_out)/df['run_id'].nunique():.0f} per run on average)")
print("Class distribution:")
print(df_out['has_fault'].value_counts().sort_index().to_string())

print(f"\nSaved to: {OUTPUT_FILE}")
df_out.to_csv(OUTPUT_FILE, index=False)

print("\nColumns created:", len(df_out.columns))
print(df_out.columns.tolist())

print("\nFirst 5 rows preview:")
print(df_out.head().to_string())

