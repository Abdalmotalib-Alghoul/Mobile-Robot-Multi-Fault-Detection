#!/usr/bin/env python3
"""
Robust Navigation Goal Node - Fast Transition Version
====================================================

Navigates the robot to randomized fixed goals with faster goal transitions.

Optimizations:
- Immediate goal cancellation upon success detection
- Reduced waiting time between goals
- More aggressive goal completion checking

Author: Optimized Navigation System
"""

import rospy
import actionlib
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseResult
from std_msgs.msg import Int32, String, Header, Bool
from tf.transformations import euler_from_quaternion
import random
import json

class RobustNavGoal:
    def __init__(self):
        rospy.init_node("robust_nav_goal", anonymous=True)
        
        # Parameters
        self.num_goals = 5
        self.min_distance = 1.0
        self.map_yaml = rospy.get_param("~map_yaml", "/home/talib/catkin_ws/maps/map.yaml")
        self.map_timeout = 10.0
        self.loop_duration = 36000.0
        
        # OPTIMIZED: Faster goal timeout parameters
        self.goal_timeout = 60.0  # Reduced from 120 to 60 seconds per goal
        self.stuck_time_threshold = 5.0  # Reduced from 10 to 5 seconds without movement
        self.success_wait_time = 0.5  # Wait only 0.5 seconds after success confirmation
        
        # Map variables
        self.map_data = None
        self.map_resolution = None
        self.map_origin = None
        self.map_width = None
        self.map_height = None
        self.goals = []
        self.current_goal_index = -1
        
        # Path metrics
        self.current_robot_pose = None
        self.last_robot_pose = None
        self.last_movement_time = None
        self.actual_path_length = 0.0
        self.deviation_from_path = 0.0
        self.recovery_behavior_triggered = False
        
        # Publishers
        self.goal_id_pub = rospy.Publisher("/current_goal_id", Int32, queue_size=1)
        self.navigation_feedback_pub = rospy.Publisher("/navigation_feedback", String, queue_size=10)
        
        # Subscribers
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        self.amcl_pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.amcl_pose_callback, queue_size=10)
        
        # In __init__ (for all nodes):
        self.shutdown_sub = rospy.Subscriber('/shutdown_signal', Bool, self.on_external_shutdown, queue_size=1)
        
        # Move base client with faster timeout
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        if not self.move_base_client.wait_for_server(rospy.Duration(5)):  # Reduced from 10 to 5 seconds
            rospy.logerr("Move_base action server not available")
            rospy.signal_shutdown("Move_base not available")
            return
        
        # Wait for map
        self.wait_for_map()
        if self.map_data is None:
            rospy.logerr("Failed to receive map within timeout")
            rospy.signal_shutdown("No map received")
            return
        
        # Generate and navigate goals
        self.generate_goals()
        self.navigate_goals()
        
        rospy.on_shutdown(self.shutdown)

    def on_external_shutdown(self, msg):
        if msg.data:
            rospy.loginfo(f"{rospy.get_name()} received shutdown signal - cleaning up.")
            # Node-specific: e.g., in collector: self.save_data()
            rospy.signal_shutdown("External graceful shutdown")

    def wait_for_map(self):
        start_time = rospy.get_time()
        while self.map_data is None and not rospy.is_shutdown():
            if rospy.get_time() - start_time > self.map_timeout:
                break
            rospy.sleep(0.1)

    def map_callback(self, msg):
        try:
            self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
            self.map_resolution = msg.info.resolution
            self.map_origin = msg.info.origin
            self.map_width = msg.info.width
            self.map_height = msg.info.height
        except Exception as e:
            rospy.logerr(f"Error processing map: {e}")

    def amcl_pose_callback(self, msg):
        self.current_robot_pose = msg.pose.pose
        
        # Check for movement
        current_time = rospy.get_time()
        if self.last_robot_pose is not None:
            dist = np.sqrt((self.current_robot_pose.position.x - self.last_robot_pose.position.x)**2 +
                           (self.current_robot_pose.position.y - self.last_robot_pose.position.y)**2)
            self.actual_path_length += dist
            
            # Update last movement time if robot moved significantly
            if dist > 0.01:  # 1cm movement threshold
                self.last_movement_time = current_time
                
        self.last_robot_pose = self.current_robot_pose
        
        # Initialize last movement time
        if self.last_movement_time is None:
            self.last_movement_time = current_time

    def generate_goals(self):
        fixed_goals = [
            {
                "position": {"x": -5.5, "y": 2.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.6460599874514892, "w": 0.7632866385665229}
            },
            {
                "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.9982646721519439, "w": 0.0588867076119223}
            },
            {
                "position": {"x": -5.5, "y": -2.5, "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "z": -0.0711579491866045, "w": -0.9974650601738171}
            }
        ]
        
        random.shuffle(fixed_goals)
        
        self.goals = []
        for goal in fixed_goals:
            x = goal["position"]["x"]
            y = goal["position"]["y"]
            orientation = goal["orientation"]
            _, _, yaw = euler_from_quaternion([orientation["x"], orientation["y"], orientation["z"], orientation["w"]])
            self.goals.append((x, y, yaw))
        
        rospy.loginfo(f"Loaded {len(self.goals)} randomized goals")

    def create_pose_stamped(self, x, y, yaw=0.0):
        pose = PoseStamped()
        pose.header = Header(frame_id="map", stamp=rospy.Time.now())
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = np.sin(yaw / 2.0)
        pose.pose.orientation.w = np.cos(yaw / 2.0)
        return pose

    def is_robot_stuck(self):
        """Check if robot hasn't moved for stuck_time_threshold"""
        if self.last_movement_time is None:
            return False
        return (rospy.get_time() - self.last_movement_time) > self.stuck_time_threshold

    def is_goal_reached(self, goal_x, goal_y, tolerance=0.3):
        """Check if robot is close enough to goal to consider it reached"""
        if self.current_robot_pose is None:
            return False
        
        distance = np.sqrt((self.current_robot_pose.position.x - goal_x)**2 +
                          (self.current_robot_pose.position.y - goal_y)**2)
        return distance <= tolerance

    def publish_navigation_feedback(self, goal_id, status, actual_path_length, deviation_from_path, recovery_behavior_triggered):
        feedback_msg = {
            "goal_id": goal_id,
            "status": status,
            "actual_path_length": actual_path_length,
            "deviation_from_path": deviation_from_path,
            "recovery_behavior_triggered": recovery_behavior_triggered
        }
        self.navigation_feedback_pub.publish(String(json.dumps(feedback_msg)))

    def navigate_goals(self):
        start_time = rospy.get_time()
        cycle_count = 0
        rate = rospy.Rate(20)  # Increased from 10 to 20 Hz for faster checking
        goal_id = 1

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < self.loop_duration:
            cycle_count += 1
            rospy.loginfo(f"Starting navigation cycle {cycle_count}")
            
            for i, (x, y, yaw) in enumerate(self.goals):
                if rospy.is_shutdown() or (rospy.get_time() - start_time) >= self.loop_duration:
                    break
                    
                rospy.loginfo(f"Navigating to goal {goal_id}: ({x:.2f}, {y:.2f}, yaw: {yaw:.2f})")
                
                self.current_goal_index = i
                self.actual_path_length = 0.0
                self.recovery_behavior_triggered = False
                self.last_movement_time = rospy.get_time()

                self.goal_id_pub.publish(Int32(data=goal_id))
                
                goal = MoveBaseGoal()
                goal.target_pose = self.create_pose_stamped(x, y, yaw=yaw)
                self.move_base_client.send_goal(goal)
                
                goal_start_time = rospy.get_time()
                goal_active = True
                goal_success = False
                
                while not rospy.is_shutdown() and (rospy.get_time() - start_time) < self.loop_duration and goal_active:
                    current_time = rospy.get_time()
                    
                    # Check for timeout
                    if current_time - goal_start_time > self.goal_timeout:
                        rospy.logwarn(f"Goal {goal_id} timeout after {self.goal_timeout} seconds")
                        self.move_base_client.cancel_goal()
                        goal_active = False
                        break
                    
                    # Check if robot is stuck
                    if self.is_robot_stuck():
                        rospy.logwarn(f"Robot appears stuck at goal {goal_id}, moving to next goal")
                        self.move_base_client.cancel_goal()
                        goal_active = False
                        self.recovery_behavior_triggered = True
                        break
                    
                    # OPTIMIZED: Check goal status more frequently
                    state = self.move_base_client.get_state()
                    
                    # If goal is succeeded, break immediately
                    if state == actionlib.GoalStatus.SUCCEEDED:
                        rospy.loginfo(f"Goal {goal_id} SUCCEEDED - moving to next goal")
                        goal_success = True
                        goal_active = False
                        break
                    
                    # Check if we're close enough to consider goal reached
                    if self.is_goal_reached(x, y) and state == actionlib.GoalStatus.ACTIVE:
                        rospy.loginfo(f"Goal {goal_id} reached proximity - canceling goal")
                        self.move_base_client.cancel_goal()
                        goal_active = False
                        goal_success = True
                        break
                    
                    # Check if goal completed via wait_for_result (non-blocking check)
                    if self.move_base_client.wait_for_result(rospy.Duration(0.05)):  # Reduced from 0.1 to 0.05
                        result = self.move_base_client.get_result()
                        if result:
                            rospy.loginfo(f"Goal {goal_id} completed with result")
                            goal_active = False
                            goal_success = True
                            break
                    
                    # Update deviation
                    if self.current_robot_pose:
                        self.deviation_from_path = np.sqrt((self.current_robot_pose.position.x - x)**2 + 
                                                          (self.current_robot_pose.position.y - y)**2)
                    
                    # Throttled feedback publishing (less frequent to reduce overhead)
                    if rospy.get_time() % 3.0 < 0.1:  # Publish every 3 seconds
                        self.publish_navigation_feedback(goal_id, "ACTIVE", self.actual_path_length, 
                                                        self.deviation_from_path, self.recovery_behavior_triggered)
                    
                    rate.sleep()
                
                # OPTIMIZED: Immediate transition to next goal
                status_text = "SUCCEEDED" if goal_success else "FAILED"
                rospy.loginfo(f"Goal {goal_id} {status_text} - transitioning to next goal")
                
                self.publish_navigation_feedback(goal_id, status_text, self.actual_path_length, 
                                                self.deviation_from_path, self.recovery_behavior_triggered)

                goal_id += 1
                # OPTIMIZED: Reduced pause between goals
                rospy.sleep(0.2)  # Reduced from 1.0 to 0.2 seconds

        rospy.loginfo(f"Completed navigation after {cycle_count} cycles or duration limit")

    def shutdown(self):
        rospy.loginfo("Shutting down robust_nav_goal node")
        if hasattr(self, 'move_base_client'):
            self.move_base_client.cancel_all_goals()
        self.map_sub.unregister()
        self.amcl_pose_sub.unregister()
        self.goal_id_pub.unregister()
        self.navigation_feedback_pub.unregister()

if __name__ == "__main__":
    try:
        RobustNavGoal()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation node terminated")
