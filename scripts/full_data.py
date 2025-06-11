#!/usr/bin/env python3
import rospy
from datetime import datetime
from std_msgs.msg import Float64 ,Float32 # Changed to Float32
from perception.msg import comparizon , depth_multiposition

# Global variables
latest_sonar = None
latest_estimated = None
sonar_received = False
estimated_received = False
csv_file = None
csv_writer = None



def sonar_callback(msg):
    """Callback for /sonar/depth (faster topic)"""
    global latest_sonar, sonar_received
    latest_sonar = msg.data  # Store just the float value
    sonar_received = True

def estimated_callback(msg):
    """Callback for /depth_estimated (slower topic)"""
    global latest_estimated, estimated_received
    latest_estimated = msg.center  # Store just the float value
    estimated_received = True
    
    # Only save if we've received sonar data
    if sonar_received:
        publish_new_topics()
    else:
        rospy.logwarn_once("Received estimated depth but no sonar data yet")

def publish_new_topics():
    """Publish synchronized depth data to new topics"""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Truncate to milliseconds
        
        # Get the depth values (0.0 if None)
        sonar_depth = latest_sonar if latest_sonar is not None else 0.0
        estimated_depth = latest_estimated if latest_estimated is not None else 0.0
        
        message = comparizon()
        message.sonar = sonar_depth/10
        message.corrected_depth = estimated_depth/10
        message.corrected_depth_error= abs(sonar_depth - estimated_depth)/10
        msg_pub.publish(message)
        
        
    except Exception as e:
        rospy.logerr(f"error: {e}")

def shutdown_hook():
    """Cleanup when node is shutdown"""
    if csv_file:
        csv_file.close()
        rospy.loginfo("CSV file closed")

if __name__ == '__main__':
    rospy.init_node('depth_data_logger')
    
    # Setup CSV - modify path as needed
    
    # Setup subscribers with Float32 type
    rospy.Subscriber('/sonar/depth', Float64, sonar_callback)
    rospy.Subscriber('/depth_estimated', depth_multiposition, estimated_callback)
    msg_pub = rospy.Publisher('/depth_comparison', comparizon, queue_size=10)
    # Register shutdown hook
    rospy.on_shutdown(shutdown_hook)
    
    rospy.loginfo("Depth data logger started. Waiting for data...")
    rospy.spin()