#!/usr/bin/env python3

import rospy
import numpy as np
import tf.transformations
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, String
import time

class TFDriftInjector:
    def __init__(self):
        rospy.init_node('tf_drift_injector', anonymous=True)

        # Parameters
        self.pose_x_offset = float(rospy.get_param('~pose_x_offset', 0.0))
        self.pose_ang_z_offset = float(rospy.get_param('~pose_ang_z_offset', 0.0)) * np.pi / 180.0
        self.experiment_duration = float(rospy.get_param('~experiment_duration', 300.0))
        self.acceleration_x_rate = float(rospy.get_param('~acceleration_x_rate', 0.0))
        self.angular_velocity_z_rate = float(rospy.get_param('~angular_velocity_z_rate', 0.0))
        self.warmup_duration = float(rospy.get_param('~warmup_duration', 2.0))

        self.scenario_name = rospy.get_param('~scenario_name', 'normal_operation')
        self.run_id = rospy.get_param('~run_id', '1')

        # Statistics
        self.total_transforms_received = 0
        self.total_transforms_processed = 0
        self.total_drifts_injected = 0
        self.experiment_start_time = time.time()

        # State
        self.start_timestamp_wall = None
        self.start_timestamp_ros = None  # Set to first real odom->base_link stamp
        self.experiment_active = True
        self.warmup_complete = False

        # Drift accumulators
        self.current_position_drift_x = np.float64(self.pose_x_offset)  # forward drift (from accel bias + offset)

        self.current_yaw_drift = np.float64(self.pose_ang_z_offset)





        # Subscribers
        self.tf_subscriber = rospy.Subscriber('/tf', TFMessage, self.tf_callback, queue_size=2000)
        self.tf_static_subscriber = rospy.Subscriber('/tf_static', TFMessage, self.tf_static_callback, queue_size=100)
        self.shutdown_sub = rospy.Subscriber('/shutdown_signal', Bool, self.on_external_shutdown, queue_size=1)

        # Publishers
        self.tf_publisher = rospy.Publisher('/tf_anomalous', TFMessage, queue_size=2000)
        self.tf_diagnose_publisher = rospy.Publisher('/tf_anomalous_diagnose', TFMessage, queue_size=2000)
        self.tf_static_publisher = rospy.Publisher('/tf_static_anomalous', TFMessage, queue_size=100)
        self.fault_labels_pub = rospy.Publisher("/fault_labels", String, queue_size=10)

        rospy.on_shutdown(self.shutdown_handler)

        rospy.loginfo("TF Drift Injector Started (OPTIMIZED FOR 76 Hz TF)")
        rospy.loginfo(f" Accel X rate: {self.acceleration_x_rate} m/s²")
        rospy.loginfo(f" Ang Z rate: {self.angular_velocity_z_rate} rad/s")


        rospy.loginfo(f" Scenario: {self.scenario_name}, run {self.run_id}")

    def on_external_shutdown(self, msg):
        if msg.data:
            rospy.loginfo("Received external shutdown signal")
            rospy.signal_shutdown("External shutdown")

    def build_fault_labels(self):
        labels = {
            'has_outlier': 0, 'has_dropout': 0, 'has_noise': 0, 'has_drift': 0,
            'has_offset': 0, 'has_pose_x_offset': 1 if self.pose_x_offset != 0.0 else 0,
            'has_acc_drift': 1 if self.acceleration_x_rate != 0.0 else 0,
            'has_pose_ang_z_offset': 1 if self.pose_ang_z_offset != 0.0 else 0,
            'has_ang_drift': 1 if self.angular_velocity_z_rate != 0.0 else 0,
            'has_crosstalk': 0
        }
        return ','.join([f"{k}:{v}" for k, v in labels.items()])

    def publish_simultaneously(self, msg):
        self.tf_publisher.publish(msg)
        self.tf_diagnose_publisher.publish(msg)

    def tf_static_callback(self, msg):
        self.tf_static_publisher.publish(msg)

    def tf_callback(self, msg):
        if not self.experiment_active:
            return

        self.total_transforms_received += 1
        current_time_wall = time.time()

        # Extract REAL timestamp from odom->base_link (if present)
        odom_stamp = None
        odom_transform_idx = None
        for i, transform in enumerate(msg.transforms):
            if transform.header.frame_id == 'odom' and transform.child_frame_id == 'base_link':
                odom_stamp = transform.header.stamp.to_sec()
                odom_transform_idx = i
                break

        if odom_stamp is None:
            # No odom->base_link in this bundle → republish unchanged
            self.publish_simultaneously(msg)
            return

        current_time_ros = odom_stamp

        # First odom->base_link → zero the timer exactly here
        if self.start_timestamp_ros is None:
            self.start_timestamp_wall = current_time_wall
            self.start_timestamp_ros = current_time_ros
            rospy.loginfo("Drift timer ZEROED to first odom->base_link stamp")
            rospy.loginfo(f" t=0 at ROS time: {current_time_ros:.3f}")

        elapsed_time_wall = current_time_wall - self.start_timestamp_wall
        elapsed_time_ros = max(0.0, current_time_ros - self.start_timestamp_ros)

        # Warmup period (no drift)
        if not self.warmup_complete and elapsed_time_ros < self.warmup_duration:
            self.publish_simultaneously(msg)
            return
        self.warmup_complete = True

        # Experiment duration check
        if self.experiment_duration > 0 and elapsed_time_wall > self.experiment_duration:
            rospy.loginfo("Experiment duration reached. Shutting down.")
            rospy.signal_shutdown("Experiment completed")
            return

        # === Process the bundle with drift injection ===
        drift_msg = TFMessage()
        drift_count = 0


        for i, transform in enumerate(msg.transforms):
            self.total_transforms_processed += 1

            if i == odom_transform_idx:  # The critical odom->base_link transform
                # Extract original (true) pose
                orig_x = np.float64(transform.transform.translation.x)
                orig_y = np.float64(transform.transform.translation.y)
                quat = np.array([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ], dtype=np.float64)
                euler = tf.transformations.euler_from_quaternion(quat)
                original_yaw = np.float64(euler[2])





                # === Compute current drifts ===
                self.current_position_drift_x = np.float64(self.pose_x_offset) + 0.5 * np.float64(self.acceleration_x_rate) * (elapsed_time_ros ** 2)
                self.current_yaw_drift = np.float64(self.pose_ang_z_offset) + np.float64(self.angular_velocity_z_rate) * elapsed_time_ros


        

                cos_y = np.cos(original_yaw)
                sin_y = np.sin(original_yaw)
                transform.transform.translation.x += self.current_position_drift_x * cos_y
                transform.transform.translation.y += self.current_position_drift_x * sin_y  

                # Apply yaw drift
                if abs(self.current_yaw_drift) > 1e-9:
                    new_yaw = original_yaw + self.current_yaw_drift
                    new_quat = tf.transformations.quaternion_from_euler(euler[0], euler[1], new_yaw)
                    norm = np.linalg.norm(new_quat)
                    if norm > 0:
                        new_quat /= norm
                    transform.transform.rotation.x = new_quat[0]
                    transform.transform.rotation.y = new_quat[1]
                    transform.transform.rotation.z = new_quat[2]
                    transform.transform.rotation.w = new_quat[3]

                # Count if significant drift injected
                if (abs(self.current_position_drift_x) > 1e-6 or

                    abs(self.current_yaw_drift) > 1e-6):
                    drift_count += 1
                    self.total_drifts_injected += 1

                drift_msg.transforms.append(transform)

            else:
                # All other transforms unchanged
                drift_msg.transforms.append(transform)

        self.publish_simultaneously(drift_msg)

        # Publish fault labels
        labels_msg = String(data=self.build_fault_labels())
        self.fault_labels_pub.publish(labels_msg)

        # Periodic logging
        if self.total_transforms_received % 100 == 0:
            rospy.loginfo(f"t={elapsed_time_ros:.1f}s | "
                          f"X drift={self.current_position_drift_x:+.3f}m | "

                          f"Yaw drift={self.current_yaw_drift:+.4f} rad | "

                          f"injected={drift_count}")

    def shutdown_handler(self):
        rospy.loginfo(f"Injector shutdown. "
                      f"Final X drift: {self.current_position_drift_x:+.3f} m | "

                      f"Final yaw drift: {self.current_yaw_drift:+.4f} rad")
        rospy.loginfo("Injector node terminated.")

if __name__ == "__main__":
    try:
        injector = TFDriftInjector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
