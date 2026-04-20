#!/usr/bin/env python3
"""
FINAL ROBUST HIGH-QUALITY DATASET COLLECTOR FOR LIDAR & ODOMETRY FAULT DETECTION
================================================================================
- Collects raw sensor data for offline feature extraction
- Targets: odometry faults (/tf_anomalous) only
- Saves 340 downsampled lidar beams from /scan + all key sensor fields
- Includes ground truth, AMCL, particle stats for strong indirect signals
- Robust: freshness checks, fallbacks, intermediate/final saves, progress logs
- Professional: clean code, error handling, real-time safe
- No engineered features (raw only) — perfect for your next-stage ML
"""

import rospy
import numpy as np
import pandas as pd
import tf
import scipy.stats as stats
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseArray, PoseWithCovarianceStamped
from std_msgs.msg import String
import os
from datetime import datetime
from threading import Lock
import time
from tf.transformations import euler_from_quaternion

class UnifiedFaultCollector:
    def __init__(self):
        rospy.init_node('unified_fault_collector', anonymous=True)

        # Parameters
        self.experiment_duration = float(rospy.get_param('~experiment_duration', 600.0))
        self.scenario_name = rospy.get_param('~scenario_name', 'fault_injection')
        self.run_id = rospy.get_param('~run_id', '1')
        self.logging_frequency = float(rospy.get_param('~logging_frequency', 10.0))
        self.save_dir = rospy.get_param('~save_dir', "/home/talib/catkin_ws/Plot/fault_analysis")
        self.beam_count = int(rospy.get_param('~lidar_beam_count', 340))

        os.makedirs(self.save_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%d-%b-%Y_%H_%M_%S")

        # State
        self.data_lock = Lock()
        self.dataset = []
        self.start_time = None
        self.last_progress_log = 0.0
        self.last_intermediate_save = 0.0

        # Latest messages
        self.latest_imu = None
        self.latest_odom = None
        self.latest_cmd_vel = None
        self.latest_scan = None
        self.latest_ground_truth = None
        self.latest_amcl_pose = None
        self.latest_particlecloud = None
        
        # TF Listener
        self.tf_listener = tf.TransformListener()

        # Timestamps
        self.last_stamps = {}

        # Fault labels
        self.current_faults = {
            'has_outlier': 0, 'has_dropout': 0, 'has_noise': 0, 'has_drift': 0,
            'has_offset': 0, 'has_pose_x_offset': 0, 'has_acc_drift': 0,
            'has_pose_ang_z_offset': 0, 'has_ang_drift': 0, 'has_crosstalk': 0
        }
        self.fault_to_id = {
            'has_outlier': 1, 'has_dropout': 2, 'has_noise': 3, 'has_drift': 4,
            'has_offset': 5, 'has_pose_x_offset': 6, 'has_acc_drift': 7,
            'has_pose_ang_z_offset': 8, 'has_ang_drift': 9, 'has_crosstalk': 10
        }

        self.setup_subscribers()
        rospy.on_shutdown(self.shutdown_handler)
        rospy.loginfo("UNIFIED FAULT COLLECTOR STARTED - Collecting raw data for odometry fault detection")

    def setup_subscribers(self):
        rospy.Subscriber('/fault_labels', String, self.fault_labels_callback)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback)
        rospy.Subscriber('/stretch_diff_drive_controller/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/stretch_diff_drive_controller/cmd_vel', Twist, self.cmd_vel_callback)
        rospy.Subscriber('/scan_anomalous', LaserScan, self.scan_callback)
        rospy.Subscriber('/ground_truth', Odometry, self.ground_truth_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        rospy.Subscriber('/particlecloud', PoseArray, self.particle_callback)

    def fault_labels_callback(self, msg):
        with self.data_lock:
            try:
                parts = msg.data.split(',')
                for part in parts:
                    if ':' in part:
                        key, val_str = part.split(':', 1)
                        key = key.strip()
                        if key in self.current_faults:
                            self.current_faults[key] = int(val_str.strip())
            except Exception as e:
                rospy.logwarn_throttle(10, f"Fault label parse error: {e}")

    def imu_callback(self, msg):
        with self.data_lock:
            self.latest_imu = msg
            self.last_stamps['imu'] = msg.header.stamp

    def odom_callback(self, msg):
        with self.data_lock:
            self.latest_odom = msg
            self.last_stamps['odom'] = msg.header.stamp

    def cmd_vel_callback(self, msg):
        with self.data_lock:
            self.latest_cmd_vel = msg

    def scan_callback(self, msg):
        with self.data_lock:
            self.latest_scan = msg
            self.last_stamps['scan'] = msg.header.stamp

    def ground_truth_callback(self, msg):
        with self.data_lock:
            self.latest_ground_truth = msg
            self.last_stamps['ground_truth'] = msg.header.stamp

    def amcl_pose_callback(self, msg):
        with self.data_lock:
            self.latest_amcl_pose = msg
            self.last_stamps['amcl_pose'] = msg.header.stamp

    def particle_callback(self, msg):
        with self.data_lock:
            self.latest_particlecloud = msg
            self.last_stamps['particlecloud'] = msg.header.stamp

    def is_data_ready(self):
        now = rospy.Time.now()
        core = ['imu', 'odom', 'scan', 'tf_anomalous']
        ready = all((now - self.last_stamps.get(k, rospy.Time(0))).to_sec() <= 0.5 for k in core)
        if not ready:
            rospy.logdebug_throttle(20, f"Waiting for core data: {[k for k in core if (now - self.last_stamps.get(k, rospy.Time(0))).to_sec() > 0.5]}")
        return ready

    def get_sensor_ranges(self):
        now = rospy.Time.now()
        if self.latest_scan and (now - self.last_stamps.get('scan', rospy.Time(0))).to_sec() <= 0.5:
            return np.array(self.latest_scan.ranges, dtype=float)
        return np.full(self.beam_count, 22222.1)

    def quaternion_to_yaw(self, q):
        try:
            return euler_from_quaternion([q['qx'], q['qy'], q['qz'], q['qw']])[2]
        except:
            return 0.0

    def downsample_ranges(self, ranges):
        ranges = np.where(np.isinf(ranges), 20.0, ranges)
        ranges = np.nan_to_num(ranges, nan=20.0)
        if len(ranges) <= self.beam_count:
            padded = np.full(self.beam_count, 20.1)
            padded[:len(ranges)] = ranges
            return padded
        indices = np.linspace(0, len(ranges) - 1, self.beam_count, dtype=int)
        return ranges[indices]

    def compute_particle_stats(self):
        if not self.latest_particlecloud or len(self.latest_particlecloud.poses) == 0:
            return {'particle_count': 0, 'particle_mean_x': 0.0, 'particle_mean_y': 0.0, 'particle_mean_yaw': 0.0,
                    'particle_std_x': 0.0, 'particle_std_y': 0.0, 'particle_std_yaw': 0.0}
        try:
            xs = np.array([p.position.x for p in self.latest_particlecloud.poses])
            ys = np.array([p.position.y for p in self.latest_particlecloud.poses])
            yaws = np.array([euler_from_quaternion([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])[2]
                             for p in self.latest_particlecloud.poses])
            
            if len(yaws) > 0:
                mean_yaw = stats.circmean(yaws, low=-np.pi, high=np.pi)
                std_yaw = stats.circstd(yaws, low=-np.pi, high=np.pi)
            else:
                mean_yaw = std_yaw = 0.0

            return {
                'particle_count': len(xs),
                'particle_mean_x': float(np.mean(xs)),
                'particle_mean_y': float(np.mean(ys)),
                'particle_mean_yaw': float(mean_yaw),
                'particle_std_x': float(np.std(xs)),
                'particle_std_y': float(np.std(ys)),
                'particle_std_yaw': float(std_yaw)
            }
        except Exception as e:
            rospy.logwarn_throttle(30, f"Particle stats error: {e}")
            return {'particle_count': 0, 'particle_mean_x': 0.0, 'particle_mean_y': 0.0, 'particle_mean_yaw': 0.0,
                    'particle_std_x': 0.0, 'particle_std_y': 0.0, 'particle_std_yaw': 0.0}

    def log_data(self):
        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time()
        
        # Attempt TF lookup - only update timestamp on success
        tf_pose = None
        try:
            self.tf_listener.waitForTransform('odom', 'base_link', rospy.Time(0), rospy.Duration(0.1))
            (trans, rot) = self.tf_listener.lookupTransform('odom', 'base_link', rospy.Time(0))
            tf_pose = {
                'x': trans[0], 'y': trans[1], 'z': trans[2],
                'qx': rot[0], 'qy': rot[1], 'qz': rot[2], 'qw': rot[3]
            }
            self.last_stamps['tf_anomalous'] = rospy.Time.now()  # Only on success
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn_throttle(10, "TF lookup failed - skipping row")
            # Return None → skip row entirely

        if tf_pose is None:
            return  # Critical: block row on failure

        if not self.is_data_ready():
            return

        with self.data_lock:
            row = {}

            now = rospy.Time.now()
            row['timestamp_ros'] = now.to_sec()
            row['wall_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            # Downsampled LiDAR beams
            ds_ranges = self.downsample_ranges(self.get_sensor_ranges())
            for j in range(self.beam_count):
                row[f'lidar_beam_{j:03d}'] = float(ds_ranges[j])

            # IMU
            row.update({
                'imu_accel_x': float(self.latest_imu.linear_acceleration.x if self.latest_imu else 0.0),
                'imu_accel_y': float(self.latest_imu.linear_acceleration.y if self.latest_imu else 0.0),
                'imu_accel_z': float(self.latest_imu.linear_acceleration.z if self.latest_imu else 0.0),
                'imu_gyro_x': float(self.latest_imu.angular_velocity.x if self.latest_imu else 0.0),
                'imu_gyro_y': float(self.latest_imu.angular_velocity.y if self.latest_imu else 0.0),
                'imu_gyro_z': float(self.latest_imu.angular_velocity.z if self.latest_imu else 0.0),
                'imu_orient_x': float(self.latest_imu.orientation.x if self.latest_imu else 0.0),
                'imu_orient_y': float(self.latest_imu.orientation.y if self.latest_imu else 0.0),
                'imu_orient_z': float(self.latest_imu.orientation.z if self.latest_imu else 0.0),
                'imu_orient_w': float(self.latest_imu.orientation.w if self.latest_imu else 1.0)
            })

            # Odom
            row.update({
                'odom_pose_x': float(self.latest_odom.pose.pose.position.x if self.latest_odom else 0.0),
                'odom_pose_y': float(self.latest_odom.pose.pose.position.y if self.latest_odom else 0.0),
                'odom_linear_x': float(self.latest_odom.twist.twist.linear.x if self.latest_odom else 0.0),
                'odom_angular_z': float(self.latest_odom.twist.twist.angular.z if self.latest_odom else 0.0)
            })

            # Cmd vel
            row.update({
                'cmd_linear_x': float(self.latest_cmd_vel.linear.x if self.latest_cmd_vel else 0.0),
                'cmd_angular_z': float(self.latest_cmd_vel.angular.z if self.latest_cmd_vel else 0.0)
            })

            # TF pose + yaw
            row.update({
                'tf_pose_x': float(tf_pose['x']),
                'tf_pose_y': float(tf_pose['y']),
                'tf_pose_z': float(tf_pose['z']),
                'tf_pose_qx': float(tf_pose['qx']),
                'tf_pose_qy': float(tf_pose['qy']),
                'tf_pose_qz': float(tf_pose['qz']),
                'tf_pose_qw': float(tf_pose['qw']),
                'tf_pose_yaw': self.quaternion_to_yaw(tf_pose)
            })

            # Ground truth
            if self.latest_ground_truth:
                q = self.latest_ground_truth.pose.pose.orientation
                row.update({
                    'gt_pose_x': float(self.latest_ground_truth.pose.pose.position.x),
                    'gt_pose_y': float(self.latest_ground_truth.pose.pose.position.y),
                    'gt_pose_yaw': euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
                })
            else:
                row.update({'gt_pose_x': 0.0, 'gt_pose_y': 0.0, 'gt_pose_yaw': 0.0})

            # AMCL pose + cov trace
            if self.latest_amcl_pose:
                p = self.latest_amcl_pose.pose.pose
                row.update({
                    'amcl_pose_x': float(p.position.x),
                    'amcl_pose_y': float(p.position.y),
                    'amcl_pose_yaw': euler_from_quaternion([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])[2],
                    'amcl_cov_trace': float(np.trace(np.array(self.latest_amcl_pose.pose.covariance).reshape(6,6)))
                })
            else:
                row.update({'amcl_pose_x': 0.0, 'amcl_pose_y': 0.0, 'amcl_pose_yaw': 0.0, 'amcl_cov_trace': 0.0})

            # Particle stats
            row.update(self.compute_particle_stats())

            # Fault labels
            row.update(self.current_faults.copy())

            active_faults = [k for k, v in self.current_faults.items() if v == 1]
            if len(active_faults) > 1:
                rospy.logwarn_throttle(30, f"Dropping row due to multiple active faults: {active_faults}")
                return

            row['has_fault'] = self.fault_to_id.get(active_faults[0] if active_faults else None, 0)

            self.dataset.append(row)

            # Progress logging
            if len(self.dataset) % 50 == 0 or (current_time - self.last_progress_log > 30.0):
                rospy.loginfo(f"Collected {len(self.dataset)} rows")
                self.last_progress_log = current_time

    def intermediate_save(self):
        current_time = time.time()
        if current_time - self.last_intermediate_save > 60.0:
            rospy.loginfo("Performing intermediate save")
            self.save_data(temp=True)
            self.last_intermediate_save = current_time

    def save_data(self, temp=False):
        with self.data_lock:
            if not self.dataset:
                rospy.logwarn("No data collected - saving empty placeholder file")
                df = pd.DataFrame(columns=['timestamp_ros'])
            else:
                df = pd.DataFrame(self.dataset)
                df = df.fillna(0.0)

            suffix = "_temp" if temp else ""

            filename = f"dataset_{self.scenario_name}_run{self.run_id}_{self.timestamp}{suffix}.csv"
            path = os.path.join(self.save_dir, filename)
            df.to_csv(path, index=False)
            rospy.loginfo(f"{'INTERMEDIATE' if temp else 'FINAL'} SAVE: {len(df)} rows → {path}")

    def shutdown_handler(self):
        elapsed = time.time() - (self.start_time or time.time())
        rospy.loginfo(f"Shutdown: {len(self.dataset)} rows collected in {elapsed:.1f}s")
        self.save_data()

    def run(self):
        rate = rospy.Rate(self.logging_frequency)
        self.start_time = time.time()
        self.last_intermediate_save = self.start_time
        rospy.loginfo(f"Starting collection at {self.logging_frequency} Hz for {self.experiment_duration}s")

        while not rospy.is_shutdown() and (time.time() - self.start_time) < self.experiment_duration:
            self.log_data()
            self.intermediate_save()
            rate.sleep()

        self.save_data()
        rospy.loginfo("Collection complete - final dataset saved")

if __name__ == '__main__':
    try:
        collector = UnifiedFaultCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")

