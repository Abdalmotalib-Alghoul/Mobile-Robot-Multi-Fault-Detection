#!/usr/bin/env python3
import rospy
import time
import os
import signal
from std_msgs.msg import Bool

class LaunchTerminator:
    def __init__(self):
        rospy.init_node('launch_terminator', anonymous=True)
        self.duration = rospy.get_param('~duration', 60) # Default to 60 seconds if not provided
        # In __init__ (for all nodes):
        self.shutdown_sub = rospy.Subscriber('/shutdown_signal', Bool, self.on_external_shutdown, queue_size=1)
        
        rospy.loginfo(f"Launch terminator started. Will shut down in {self.duration} seconds.")
        time.sleep(self.duration)
        rospy.loginfo("Launch terminator shutting down ROS.")
        # Attempt to kill the entire process group of the roslaunch process
        # This assumes the launch_terminator node is part of the main roslaunch process group
        # and killing it will cause roslaunch to exit.
        try:
            pgid = os.getpgid(os.getppid()) # Get process group ID of parent (roslaunch)
            os.killpg(pgid, signal.SIGTERM)
        except Exception as e:
            rospy.logwarn(f"Could not kill process group: {e}. Falling back to rospy.signal_shutdown.")
            rospy.signal_shutdown("Experiment duration reached.")

    def on_external_shutdown(self, msg):
        if msg.data:
            rospy.loginfo(f"{rospy.get_name()} received shutdown signal - cleaning up.")
            # Node-specific: e.g., in collector: self.save_data()
            rospy.signal_shutdown("External graceful shutdown")

if __name__ == '__main__':
    try:
        terminator = LaunchTerminator()
    except rospy.ROSInterruptException:
        pass
