#!/usr/bin/env python3
"""
Minimal LiDAR Dropout Injector - Behaves EXACTLY like enhanced_outlier_injector.py
- No file saving
- No in-memory data collection
- Only injects dropout and publishes /scan_anomalous and /fault_labels
- Uses periodic dropout events with per-ray probability
"""

import rospy
import numpy as np
import random
import time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Bool

class MinimalLidarDropoutInjector:
    def __init__(self):
        rospy.init_node("enhanced_lidar_dropout_injector", anonymous=True)






        # Parameters
        self.dropout_mode = str(rospy.get_param("~dropout_mode", "periodic"))
        self.dropout_interval = float(rospy.get_param("~dropout_interval", 1.0))
        self.dropout_duration = float(rospy.get_param("~dropout_duration", 10.0))
        self.per_ray_dropout_prob = float(rospy.get_param("~per_ray_dropout_prob", 1.0))
        self.experiment_duration = float(rospy.get_param("~experiment_duration", 300))


        # State
        self.experiment_start_time = rospy.Time.now().to_sec()
        self.is_dropping_out = False
        self.dropout_start_time = None
        self.next_dropout_time = None

        # ROS interface
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.lidar_callback)
        self.scan_pub = rospy.Publisher("/scan_anomalous", LaserScan, queue_size=10)
        self.fault_labels_pub = rospy.Publisher("/fault_labels", String, queue_size=10)
        self.dropout_status_pub = rospy.Publisher("/dropout_status", Bool, queue_size=10)

        # Shutdown subscriber
        self.shutdown_sub = rospy.Subscriber('/shutdown_signal', Bool, self.on_external_shutdown, queue_size=1)

        # Disable if no dropout
        if (self.dropout_interval <= 0 or self.dropout_duration <= 0 or 
            self.per_ray_dropout_prob <= 0.0 or self.dropout_mode == "none"):
            rospy.loginfo("Dropout disabled (baseline mode)")
            self.dropout_mode = "none"
        else:
            self.schedule_next_dropout()

        rospy.loginfo("Minimal LiDAR Dropout Injector Started")
        rospy.loginfo(f"   Mode: {self.dropout_mode}")
        rospy.loginfo(f"   Interval: {self.dropout_interval}s")
        rospy.loginfo(f"   Duration: {self.dropout_duration}s")
        rospy.loginfo(f"   Per-Ray Prob: {self.per_ray_dropout_prob*100:.0f}%")

    def on_external_shutdown(self, msg):
        if msg.data:
            rospy.loginfo("Received shutdown signal - injector shutting down.")
            rospy.signal_shutdown("External shutdown")

    def schedule_next_dropout(self):
        """Schedule next dropout event after interval"""
        if self.dropout_mode != "periodic":
            return
        delay = self.dropout_interval
        self.next_dropout_time = rospy.Time.now().to_sec() + delay
        rospy.Timer(rospy.Duration(delay), self.start_dropout_event, oneshot=True)

    def start_dropout_event(self, event=None):
        """Start a dropout event"""
        if self.is_dropping_out:
            return
        self.is_dropping_out = True
        self.dropout_start_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"DROPOUT EVENT STARTED at {self.dropout_start_time - self.experiment_start_time:.1f}s")

        # Schedule end
        rospy.Timer(rospy.Duration(self.dropout_duration), self.end_dropout_event, oneshot=True)

    def end_dropout_event(self, event=None):
        """End current dropout event"""
        if not self.is_dropping_out:
            return
        self.is_dropping_out = False
        end_time = rospy.Time.now().to_sec()
        rospy.loginfo(f"DROPOUT EVENT ENDED at {end_time - self.experiment_start_time:.1f}s")

        # Schedule next
        self.schedule_next_dropout()

    def build_fault_labels(self, has_dropout=False):
        """Generate unified fault labels"""
        labels = {
            'has_dropout': 1 if has_dropout else 0,
            'has_outlier': 0, 'has_noise': 0, 'has_drift': 0,
            'has_offset': 0, 'has_acc_offset': 0, 'has_acc_drift': 0,
            'has_ang_offset': 0, 'has_ang_drift': 0, 'has_crosstalk': 0
        }
        return ','.join([f"{k}:{v}" for k, v in labels.items()])

    def lidar_callback(self, msg):
        """Process scan and apply dropout if active"""
        current_time = rospy.Time.now().to_sec()
        if self.experiment_duration > 0 and (current_time - self.experiment_start_time) > self.experiment_duration:
            return  # Silent exit after duration

        # Check if dropout active
        dropout_active = False
        if self.is_dropping_out:
            elapsed = current_time - self.dropout_start_time
            if elapsed >= self.dropout_duration:
                self.end_dropout_event()
            else:
                dropout_active = True

        # Copy message
        anomalous_msg = LaserScan()
        anomalous_msg.header = msg.header
        anomalous_msg.angle_min = msg.angle_min
        anomalous_msg.angle_max = msg.angle_max
        anomalous_msg.angle_increment = msg.angle_increment
        anomalous_msg.time_increment = msg.time_increment
        anomalous_msg.scan_time = msg.scan_time
        anomalous_msg.range_min = msg.range_min
        anomalous_msg.range_max = msg.range_max
        anomalous_msg.intensities = msg.intensities

        ranges = np.array(msg.ranges)
        if dropout_active and self.per_ray_dropout_prob > 0.0:
            drop_mask = np.random.random(len(ranges)) < self.per_ray_dropout_prob
            ranges[drop_mask] = float('inf')

        anomalous_msg.ranges = ranges.tolist()
        self.scan_pub.publish(anomalous_msg)

        # Publish labels
        labels_str = self.build_fault_labels(dropout_active)
        self.fault_labels_pub.publish(String(data=labels_str))

        # Publish status
        self.dropout_status_pub.publish(Bool(data=dropout_active))

        # Log periodically
        if int(current_time) % 50 == 0:
            rays_dropped = np.sum(drop_mask) if dropout_active else 0
            rospy.loginfo(f"Scan processed | Dropout: {dropout_active} | Rays dropped: {rays_dropped}")

        # Auto-shutdown after duration
        if self.experiment_duration > 0 and (current_time - self.experiment_start_time) > self.experiment_duration:
            rospy.loginfo("Experiment duration reached. Shutting down.")
            rospy.signal_shutdown("Experiment completed")

if __name__ == "__main__":
    try:
        injector = MinimalLidarDropoutInjector()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Dropout injector node interrupted.")
