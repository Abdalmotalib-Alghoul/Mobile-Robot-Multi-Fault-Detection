#!/usr/bin/env python3
"""
Drift Injector - Distance offset and Gaussian Noise (Cumulative)
===============================================================

Simulates cumulative distance offset and Gaussian noise in LiDAR sensors using .

"""

import rospy
import numpy as np
import pandas as pd
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header, String, Bool
import os
import time
from datetime import datetime

class DriftInjector:
    def __init__(self):
        """Initialize the drift injector."""
        rospy.init_node('drift_injector', anonymous=True)
        
        self.min_range = 0.05  # meters
        self.max_range = 12.0  # meters
        
        # Get parameters
        self.initial_distance_offset_m = float(rospy.get_param('~distance_offset_m', 0.1))
        self.experiment_duration = float(rospy.get_param('~experiment_duration', 60))
        self.drift_rate_mps = float(rospy.get_param('~offset_rate_mps', 0.0))
        self.noise_std_m = float(rospy.get_param('~noise_std_m', 0.0))
        
        # Initialize data storage
        self.scan_count = 0
        self.start_timestamp_ros = None
        self.experiment_active = True
        self.current_distance_drift_m = self.initial_distance_offset_m
        
        # ROS setup
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.drift_publisher = rospy.Publisher('/scan_anomalous', LaserScan, queue_size=10)
        self.fault_labels_pub = rospy.Publisher("/fault_labels", String, queue_size=10)
        
        # Shutdown subscriber
        self.shutdown_sub = rospy.Subscriber('/shutdown_signal', Bool, self.on_external_shutdown, queue_size=1)
        
        # Log initialization
        rospy.loginfo("📏 Drift Injector initialized (Cumulative Mode)")
        rospy.loginfo(f"   Initial distance drift: {self.initial_distance_offset_m:.6f}m")
        rospy.loginfo(f"   Drift rate: {self.drift_rate_mps:.6f}m/s")
        if self.noise_std_m > 0:
            rospy.loginfo(f"   Gaussian noise std: {self.noise_std_m:.6f}m")
        rospy.loginfo(f"   Experiment duration: {self.experiment_duration:.2f}s")
        rospy.loginfo("   Formula: r'ᵢ = rᵢ + (initial_drift + drift_rate * t)⋅cos(θᵢ)")
        if self.noise_std_m > 0:
            rospy.loginfo("   + N(0, noise_std)")
        rospy.loginfo("📏 Running (Cumulative Mode)...")
        rospy.loginfo(f"   Publishing to /scan_anomalous")

    def on_external_shutdown(self, msg):
        if msg.data:
            rospy.loginfo(f"{rospy.get_name()} received shutdown signal - cleaning up.")
            rospy.signal_shutdown("External graceful shutdown")

    def build_fault_labels(self, has_offset=False, offset_count=0, has_drift=False, drift_count=0, has_noise=False, noise_count=0):
        labels = {
            'has_outlier': 0, 'has_dropout': 0, 'has_noise': 1 if has_noise else 0, 'has_drift': 1 if has_drift else 0,
            'has_offset': 1 if has_offset else 0, 'has_acc_offset': 0, 'has_acc_drift': 0,
            'has_ang_offset': 0, 'has_ang_drift': 0, 'has_crosstalk': 0,
            'offset_count': offset_count, 'drift_count': drift_count, 'noise_count': noise_count
        }
        return ','.join([f"{k}:{v}" for k, v in labels.items()])

    def apply_distance_drift(self, ranges, angle_min, angle_increment, elapsed_time):
        """Apply cumulative distance drift."""
        ranges_array = np.array(ranges)

        num_ranges = len(ranges)
        beam_angles = np.array([angle_min + i * angle_increment for i in range(num_ranges)])
        self.current_distance_drift_m = self.initial_distance_offset_m + self.drift_rate_mps * elapsed_time
        drift_ranges = ranges_array + self.current_distance_drift_m * np.cos(beam_angles)
        
        return drift_ranges

    def apply_gaussian_noise(self, ranges, sigma):
        """Apply Gaussian noise to ranges."""
        noise = np.random.normal(0, sigma, len(ranges))
        return ranges + noise

    def scan_callback(self, msg):
        """Process LiDAR scan and apply drift and noise."""
        scan_timestamp_ros = msg.header.stamp.to_sec()
        
        if self.start_timestamp_ros is None:
            self.start_timestamp_ros = scan_timestamp_ros
            rospy.loginfo("🚀 Starting cumulative distance drift experiment...")
        
        elapsed_time_ros = scan_timestamp_ros - self.start_timestamp_ros
        
        # Check termination
        if elapsed_time_ros > self.experiment_duration:
            if self.experiment_active:
                rospy.loginfo("⏰ Experiment duration completed")
                self.experiment_active = False
                rospy.signal_shutdown("Experiment completed")
            return
        
        self.scan_count += 1
        
        # Store data
        drift_ranges = self.apply_distance_drift(msg.ranges, msg.angle_min, msg.angle_increment, elapsed_time_ros)
        
        # Apply noise if enabled
        if self.noise_std_m > 0:
            drift_ranges = self.apply_gaussian_noise(drift_ranges, self.noise_std_m)
        
        num_beams = len(drift_ranges)

        # Determine labels
        has_offset = self.initial_distance_offset_m != 0
        has_drift = self.drift_rate_mps != 0
        has_noise = self.noise_std_m > 0
        offset_count = num_beams if has_offset else 0
        drift_count = num_beams if has_drift else 0
        noise_count = num_beams if has_noise else 0

        labels_str = self.build_fault_labels(has_offset=has_offset, offset_count=offset_count,
                                             has_drift=has_drift, drift_count=drift_count,
                                             has_noise=has_noise, noise_count=noise_count)

        labels_msg = String()
        labels_msg.data = labels_str
        self.fault_labels_pub.publish(labels_msg)
        
        # Publish drift scan
        drift_scan = LaserScan()
        drift_scan.header = Header()
        drift_scan.header.stamp = rospy.Time.now()
        drift_scan.header.frame_id = msg.header.frame_id
        drift_scan.angle_min = msg.angle_min
        drift_scan.angle_max = msg.angle_max
        drift_scan.angle_increment = msg.angle_increment
        drift_scan.time_increment = msg.time_increment
        drift_scan.scan_time = msg.scan_time
        drift_scan.range_min = msg.range_min
        drift_scan.range_max = msg.range_max
        drift_scan.ranges = drift_ranges.tolist()
        drift_scan.intensities = msg.intensities
        self.drift_publisher.publish(drift_scan)
        
        # Progress logging
        if self.scan_count % 50 == 0:
            remaining_time = max(0, self.experiment_duration - elapsed_time_ros)
            log_msg = f"📊 Scan {self.scan_count}: {elapsed_time_ros:.3f}s elapsed, " \
                      f"{remaining_time:.3f}s remaining, " \
                      f"Current drift: {self.current_distance_drift_m:.6f}m"
            if self.noise_std_m > 0:
                log_msg += f", Noise std: {self.noise_std_m:.6f}m"
            rospy.loginfo(log_msg)
            rospy.loginfo(f"Published labels: {labels_str}")

if __name__ == '__main__':
    try:
        injector = DriftInjector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 ROS interrupt received")
    except Exception as e:
        rospy.logerr(f"❌ Failed to start: {e}")
