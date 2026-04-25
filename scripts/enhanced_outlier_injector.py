#!/usr/bin/env python3
"""
This script was initially developed to inject Cross-Talk faults. Later, it was modified to inject Beam Loss faults, so the terminology used here belongs to Cross-Talk. However, the core function injects Beam Loss. The terminology was not edited to avoid bugs and to allow reuse of the script for future Cross-Talk injection.
"""

import rospy
import numpy as np
import random
import time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool

class PhysicsBasedCrosstalkOutlierInjector:
    def __init__(self):
        rospy.init_node("enhanced_crosstalk_injector", anonymous=True)
        
        # Parameters
        self.outlier_percentage = rospy.get_param("~outlier_percentage", 10.0)
        self.experiment_duration = rospy.get_param("~experiment_duration", 3500)
        self.sensor_max_range = rospy.get_param("~sensor_max_range", 12.0)
        self.sensor_min_range = rospy.get_param("~sensor_min_range", 0.15)
        
        # Indoor-specific ghost parameters 
        self.ghost_min = rospy.get_param("~ghost_min_distance", 0.5)      # Close interferer 
        self.ghost_max = rospy.get_param("~ghost_max_distance", 6.0)      # Typical indoor interference distance
        self.ghost_noise = rospy.get_param("~ghost_noise_std", 0.3)       # Small natural variation
        
        # Statistics
        self.total_scans_received = 0
        self.total_points_processed = 0
        self.total_outliers_injected = 0
        self.experiment_start_time = time.time()
        
        # ROS interface
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.scan_pub = rospy.Publisher("/scan_anomalous", LaserScan, queue_size=10)
        self.fault_labels_pub = rospy.Publisher("/fault_labels", String, queue_size=10)
        
        # Shutdown subscriber
        self.shutdown_sub = rospy.Subscriber('/shutdown_signal', Bool, self.on_external_shutdown, queue_size=1)
        
        # Burst state for consecutive scans
        self.burst_state = {'active': False, 'remaining_scans': 0, 'block': None, 'stall_values': None}
        
        rospy.loginfo("🎯 Indoor Robot Crosstalk Outlier Injector Started")
        rospy.loginfo(f"   Outlier Percentage: {self.outlier_percentage}%")
        rospy.loginfo(f"   Ghost Distance: {self.ghost_min}–{self.ghost_max}m (±{self.ghost_noise}m noise)")

    def on_external_shutdown(self, msg):
        if msg.data:
            rospy.loginfo(f"{rospy.get_name()} received shutdown signal - cleaning up.")
            rospy.signal_shutdown("External graceful shutdown")

    def build_fault_labels(self, has_crosstalk=False, crosstalk_count=0):
        labels = {
            'has_outlier': 0, 'has_dropout': 0, 'has_noise': 0, 'has_drift': 0,
            'has_offset': 0, 'has_acc_offset': 0, 'has_acc_drift': 0,
            'has_ang_offset': 0, 'has_ang_drift': 0, 'has_crosstalk': 1 if has_crosstalk else 0,
            'crosstalk_count': crosstalk_count
        }
        return ','.join([f"{k}:{v}" for k, v in labels.items()])

    def inject_crosstalk_outliers(self, ranges):
        ranges_array = np.array(ranges)
        num_points = len(ranges)
        
        outlier_percentage_clamped = max(0.0, min(100.0, self.outlier_percentage))
        total_outliers = int((outlier_percentage_clamped / 100.0) * num_points)
        block_size = total_outliers
        
        if block_size == 0:
            return ranges_array.tolist(), 0
        
        outlier_count = 0
        start_idx = end_idx = 0
        stall_values = []
        
        if self.burst_state['active'] and self.burst_state['remaining_scans'] > 0:
            start_idx, end_idx = self.burst_state['block']
            stall_values = self.burst_state['stall_values']
            self.burst_state['remaining_scans'] -= 1
            if self.burst_state['remaining_scans'] == 0:
                self.burst_state['active'] = False
        else:
            # Build available blocks
            num_blocks = num_points // block_size
            available_blocks = []
            for i in range(num_blocks):
                s = i * block_size
                e = s + block_size
                if e <= num_points:
                    available_blocks.append((s, e))
            leftover = num_points % block_size
            if leftover > 0:
                available_blocks.append((num_points - leftover, num_points))
            
            if not available_blocks:
                return ranges_array.tolist(), 0
            
            start_idx, end_idx = random.choice(available_blocks)
            
            # One main short ghost distance for the burst (indoor nearby interferer)
            main_ghost_distance = random.uniform(self.ghost_min, self.ghost_max)
            
            stall_values = []
            for idx in range(start_idx, end_idx):
                true_range = ranges_array[idx]
                if true_range >= self.sensor_max_range:
                    stall_values.append(20.0)
                else:
                    ghost_range = main_ghost_distance + random.uniform(-self.ghost_noise, self.ghost_noise)
                    ghost_range = max(self.sensor_min_range, min(ghost_range, self.sensor_max_range))
                    stall_values.append(20.0)
            
            self.burst_state = {
                'active': True,
                'remaining_scans': random.randint(1, 4),
                'block': (start_idx, end_idx),
                'stall_values': stall_values
            }
        
        # Apply stall values
        for i, idx in enumerate(range(start_idx, end_idx)):
                ranges_array[idx] = stall_values[i]
                outlier_count += 1

        return ranges_array.tolist(), outlier_count
        
    def lidar_callback(self, msg):
        self.total_scans_received += 1
        
        output_msg = LaserScan()
        output_msg.header = msg.header
        output_msg.angle_min = msg.angle_min
        output_msg.angle_max = msg.angle_max
        output_msg.angle_increment = msg.angle_increment
        output_msg.time_increment = msg.time_increment
        output_msg.scan_time = msg.scan_time
        output_msg.range_min = msg.range_min
        output_msg.range_max = msg.range_max
        
        anomalous_ranges, outlier_count = self.inject_crosstalk_outliers(msg.ranges)
        self.total_outliers_injected += outlier_count
        self.total_points_processed += len(msg.ranges)
        
        output_msg.ranges = anomalous_ranges
        output_msg.intensities = msg.intensities
        
        self.scan_pub.publish(output_msg)
        
        labels_str = self.build_fault_labels(has_crosstalk=bool(outlier_count > 0), crosstalk_count=outlier_count)
        labels_msg = String()
        labels_msg.data = labels_str
        self.fault_labels_pub.publish(labels_msg)
        
        if self.total_scans_received % 50 == 0:
            actual_percentage = (self.total_outliers_injected / max(self.total_points_processed, 1)) * 100
            num_points = len(msg.ranges)
            total_outliers = int((self.outlier_percentage / 100.0) * num_points)
            rospy.loginfo(f"Scan {self.total_scans_received}: {outlier_count} crosstalk outliers ({total_outliers} beams), Total: {actual_percentage:.1f}%")
            rospy.loginfo(f"Published labels: {labels_str}")
        
        if self.experiment_duration > 0 and (time.time() - self.experiment_start_time) > self.experiment_duration:
            rospy.loginfo("Experiment duration reached. Shutting down.")
            rospy.signal_shutdown("Experiment completed")

if __name__ == "__main__":
    try:
        injector = PhysicsBasedCrosstalkOutlierInjector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Injector node interrupted.")
