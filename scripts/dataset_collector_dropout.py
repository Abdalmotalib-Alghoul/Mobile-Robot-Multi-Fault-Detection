#!/usr/bin/env python3
"""
ENHANCED Dataset Collector for LiDAR Dropout Detection Training
Updated for dropout injector compatibility - RAW-ONLY VERSION
Node Name: dataset_collector_dropout
"""

import rospy
import numpy as np
import pandas as pd
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, String
import os
from datetime import datetime
from threading import Lock
import time

class EnhancedDatasetCollector:
    def __init__(self):
        rospy.init_node('dataset_collector_dropout', anonymous=True)
        
        # Parameters - KEEP SIMPLE
        self.experiment_duration = float(rospy.get_param('~experiment_duration', 60.0))
        self.scenario_name = rospy.get_param('~scenario_name', 'dropout_injection')
        self.run_id = rospy.get_param('~run_id', '1')
        self.logging_frequency = float(rospy.get_param('~logging_frequency', 10.0))
        
        # Data collection
        self.save_directory = rospy.get_param('~injector_save_dir', "/home/talib/catkin_ws/Plot/dropout_analysis")
        os.makedirs(self.save_directory, exist_ok=True)

        self.timestamp = datetime.now().strftime("%d-%b-%Y_%H_%M_%S") 
        
        # State
        self.data_lock = Lock()
        self.dataset = []
        self.start_time = None
        self.experiment_active = True
        
        # Data storage
        self.latest_imu = {'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0, 
                          'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0}
        self.latest_odom = {'pose_x': 0.0, 'pose_y': 0.0}
        self.latest_cmd_vel = {'linear_x': 0.0, 'angular_z': 0.0}
        self.latest_anomalous_scan = None
        
        # Timestamps for sync
        self.last_imu_stamp = rospy.Time(0)
        self.last_odom_stamp = rospy.Time(0)
        self.last_scan_stamp = rospy.Time(0)
        
        # Fault labels storage
        self.current_faults = {
            'has_outlier': 0, 'has_dropout': 0, 'has_noise': 0, 'has_drift': 0,
            'has_offset': 0, 'has_acc_offset': 0, 'has_acc_drift': 0,
            'has_ang_offset': 0, 'has_ang_drift': 0, 'has_crosstalk': 0
        }
        
        # Setup subscribers
        self.setup_subscribers()
        rospy.on_shutdown(self.shutdown_handler)
        
        rospy.loginfo(f"ENHANCED Collector - Dropout Detection Dataset (Raw-Only)")
        rospy.loginfo(f"   Duration: {self.experiment_duration}s, Freq: {self.logging_frequency}Hz")

    def setup_subscribers(self):
        """Subscribe to unified fault labels and core topics"""
        try:
            self.anomalous_scan_sub = rospy.Subscriber('/scan_anomalous', LaserScan, self.anomalous_scan_callback)
            self.fault_labels_sub = rospy.Subscriber('/fault_labels', String, self.fault_labels_callback)
            self.imu_sub = rospy.Subscriber('/imu/data', Imu, self.imu_callback)
            self.odom_sub = rospy.Subscriber('/stretch_diff_drive_controller/odom', Odometry, self.odom_callback)
            self.cmd_vel_sub = rospy.Subscriber('/stretch_diff_drive_controller/cmd_vel', Twist, self.cmd_vel_callback)
            
            rospy.loginfo("Enhanced subscribers setup - Dropout compatible")
        except Exception as e:
            rospy.logerr(f"Subscriber setup failed: {e}")

    def fault_labels_callback(self, msg):
        """Parse unified fault labels string"""
        with self.data_lock:  # Add lock for thread safety
            try:
                parts = msg.data.split(',')
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        key = key.strip()
                        if key in self.current_faults:
                            self.current_faults[key] = int(value) if key.startswith('has_') else float(value)
            except Exception as e:
                rospy.logwarn(f"Failed to parse fault labels: {msg.data} - {e}")

    def imu_callback(self, msg):
        with self.data_lock:
            self.latest_imu = {'accel_x': msg.linear_acceleration.x, 'accel_y': msg.linear_acceleration.y, 'accel_z': msg.linear_acceleration.z,
                              'gyro_x': msg.angular_velocity.x, 'gyro_y': msg.angular_velocity.y, 'gyro_z': msg.angular_velocity.z}
            self.last_imu_stamp = msg.header.stamp

    def odom_callback(self, msg):
        with self.data_lock:
            self.latest_odom = {'pose_x': msg.pose.pose.position.x, 'pose_y': msg.pose.pose.position.y}
            self.last_odom_stamp = msg.header.stamp

    def cmd_vel_callback(self, msg):
        with self.data_lock:
            self.latest_cmd_vel = {'linear_x': msg.linear.x, 'angular_z': msg.angular.z}

    def anomalous_scan_callback(self, msg):
        with self.data_lock:
            self.latest_anomalous_scan = msg
            self.last_scan_stamp = msg.header.stamp

    def log_data(self):
        with self.data_lock:
            if self.start_time is None:
                self.start_time = time.time()
            
            elapsed_time = time.time() - self.start_time
            
            # Skip if experiment over
            if elapsed_time > self.experiment_duration:
                return
            
            # Skip if sensors not fresh
            now = rospy.Time.now()
            if ((now - self.last_imu_stamp).to_sec() > 0.2 or 
                (now - self.last_odom_stamp).to_sec() > 0.2 or 
                (now - self.last_scan_stamp).to_sec() > 0.2):
                return
            
            if self.latest_anomalous_scan is None:
                return
            
            row = {}
            
            # ==================== FEATURES SECTION ====================
            # Timestamps 
            row['timestamp'] = now.to_sec()
            wall_now = datetime.now()
            row['wall_time'] = wall_now.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # Lidar beams (50): Evenly subsample from full ranges
            ranges = self.latest_anomalous_scan.ranges
            num_full_beams = len(ranges)
            num_save_beams = 50
            selected_indices = [int(j * num_full_beams / num_save_beams) for j in range(num_save_beams)]
            for j, idx in enumerate(selected_indices):
                row[f'lidar_beam_{j}'] = float(ranges[idx])
            
            # IMU (6)
            row['imu_accel_x'] = float(self.latest_imu['accel_x'])
            row['imu_accel_y'] = float(self.latest_imu['accel_y'])
            row['imu_accel_z'] = float(self.latest_imu['accel_z'])
            row['imu_gyro_x'] = float(self.latest_imu['gyro_x'])
            row['imu_gyro_y'] = float(self.latest_imu['gyro_y'])
            row['imu_gyro_z'] = float(self.latest_imu['gyro_z'])
            
            # Odom (4)
            row['odom_pose_x'] = float(self.latest_odom['pose_x'])
            row['odom_pose_y'] = float(self.latest_odom['pose_y'])
            row['odom_linear_x'] = float(self.latest_cmd_vel['linear_x'])
            row['odom_angular_z'] = float(self.latest_cmd_vel['angular_z'])
            
            # ==================== LABELS SECTION (AT THE END) ====================
            # Assign all current faults as columns (dynamic, no override)
            for key, value in self.current_faults.items():
                row[key] = value
            
            self.dataset.append(row)
        
        # Log dropout rate every 50 rows
        if len(self.dataset) % 50 == 0 and len(self.dataset) > 0:
            dropout_rate = sum(d['has_dropout'] for d in self.dataset) / len(self.dataset)
            rospy.loginfo(f"Rows: {len(self.dataset)}, Dropout rate: {dropout_rate:.2%}")

    def save_data(self):
        """Save raw dataset with dropout labels"""
        try:
            if not self.dataset:
                rospy.logwarn("No data to save!")
                return
            
            df = pd.DataFrame(self.dataset)
            
            # Data validation
         
            
            if df.isnull().any().any():
                rospy.logwarn("Dataset contains NaN values - filling with zeros")
                df = df.fillna(0)
            
            filename = f"dropout_dataset_{self.scenario_name}_run{self.run_id}_{self.timestamp}.csv"
            path = os.path.join(self.save_directory, filename)
            
            df.to_csv(path, index=False)
            file_size = os.path.getsize(path)
            
            # Calculate statistics
            dropout_rows = df[df['has_dropout'] == 1]
            normal_rows = df[df['has_dropout'] == 0]
            has_cols = [col for col in df.columns if col.startswith('has_')]
            total_faulty = df[has_cols].sum(axis=1).gt(0).sum()
            column_list = list(df.columns)
            label_columns = [col for col in column_list if col.startswith('has_')]
            
            rospy.loginfo(f"✅ SAVED DROPOUT DATASET: {filename}")
            rospy.loginfo(f"   Total Rows: {len(df)}")
            rospy.loginfo(f"   Dropout Samples: {len(dropout_rows)} ({len(dropout_rows)/len(df)*100:.1f}%)")
            rospy.loginfo(f"   Normal Samples: {len(normal_rows)} ({len(normal_rows)/len(df)*100:.1f}%)")
            rospy.loginfo(f"   Total Faulty Samples: {total_faulty} ({total_faulty/len(df)*100:.1f}%)")
            rospy.loginfo(f"   Columns: {len(df.columns)}, Size: {file_size} bytes")
            rospy.loginfo(f"   Label columns (at end): {len(label_columns)}")
            
            # Verify column structure
            expected_min_columns = 50 + 6 + 4 + 2 + 10  # beams + IMU + odom + timestamps + labels
            last_columns = column_list[-len(label_columns):]
            if sorted(last_columns) == sorted(label_columns):
                rospy.loginfo(f"   ✅ Labels correctly placed at the end")
            else:
                rospy.logwarn(f"   ⚠️  Labels not at the end - check column order")
            if len(df.columns) >= expected_min_columns:
                rospy.loginfo(f"   ✅ Column count sufficient: {len(df.columns)}")
            else:
                rospy.logwarn(f"   ⚠️  Column count low: expected {expected_min_columns}, got {len(df.columns)}")
                
        except Exception as e:
            rospy.logerr(f"Save error: {e}")
            import traceback
            traceback.print_exc()

    def shutdown_handler(self):
        """Clean shutdown"""
        elapsed_time = time.time() - (self.start_time or time.time())
        dropout_count = sum(d['has_dropout'] for d in self.dataset)
        rospy.loginfo(f"Shutdown - Collected {len(self.dataset)} rows, {dropout_count} dropout samples in {elapsed_time:.1f}s")
        self.save_data()

    def run(self):
        """Main loop with experiment duration check"""
        rate = rospy.Rate(self.logging_frequency)
        self.start_time = time.time()
        rospy.loginfo(f"Starting dropout data collection at {self.logging_frequency} Hz")
        rospy.loginfo(f"Experiment duration: {self.experiment_duration} seconds")
        
        while (not rospy.is_shutdown() and 
               self.experiment_active and 
               (time.time() - self.start_time) < self.experiment_duration):
            self.log_data()
            rate.sleep()
        
        rospy.loginfo(f"Experiment duration ({self.experiment_duration}s) reached - saving data...")
        self.save_data()
        rospy.loginfo("Data saved successfully, shutting down.")

if __name__ == '__main__':
    try:
        collector = EnhancedDatasetCollector()
        collector.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Failed to start: {e}")
