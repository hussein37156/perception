#!/usr/bin/env python
import rospy
import csv
from os.path import expanduser
from datetime import datetime
from std_msgs.msg import Float64 ,Float32 # Changed to Float32

# Global variables
latest_sonar = None
latest_estimated = None
sonar_received = False
estimated_received = False
csv_file = None
csv_writer = None

def setup_csv(output_path):
    """Initialize the CSV file with headers"""
    global csv_file, csv_writer
    try:
        csv_file = open(expanduser(output_path), 'w')
        csv_writer = csv.writer(csv_file)
        # Simplified header row for Float32 data
        csv_writer.writerow([
            'timestamp',
            'sonar_depth',
            'estimated_depth'
        ])
        csv_file.flush()
        rospy.loginfo(f"CSV logging started at {output_path}")
    except IOError as e:
        rospy.logerr(f"Failed to open CSV file: {e}")
        rospy.signal_shutdown("CSV file error")

def sonar_callback(msg):
    """Callback for /sonar/depth (faster topic)"""
    global latest_sonar, sonar_received
    latest_sonar = msg.data  # Store just the float value
    sonar_received = True

def estimated_callback(msg):
    """Callback for /depth_estimated (slower topic)"""
    global latest_estimated, estimated_received
    latest_estimated = msg.data  # Store just the float value
    estimated_received = True
    
    # Only save if we've received sonar data
    if sonar_received:
        save_to_csv()
    else:
        rospy.logwarn_once("Received estimated depth but no sonar data yet")

def save_to_csv():
    """Save synchronized depth data to CSV"""
    global csv_writer
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # Truncate to milliseconds
        
        # Get the depth values (0.0 if None)
        sonar_depth = latest_sonar if latest_sonar is not None else 0.0
        estimated_depth = latest_estimated if latest_estimated is not None else 0.0
        
        # Write to CSV
        csv_writer.writerow([
            timestamp,
            sonar_depth,
            estimated_depth
        ])
        csv_file.flush()
    except Exception as e:
        rospy.logerr(f"Error writing to CSV: {e}")

def shutdown_hook():
    """Cleanup when node is shutdown"""
    if csv_file:
        csv_file.close()
        rospy.loginfo("CSV file closed")

if __name__ == '__main__':
    rospy.init_node('depth_data_logger')
    
    # Setup CSV - modify path as needed
    setup_csv('~/depth_comparison.csv')
    
    # Setup subscribers with Float32 type
    rospy.Subscriber('/sonar/depth', Float64, sonar_callback)
    rospy.Subscriber('/depth_estimated', Float32, estimated_callback)
    
    # Register shutdown hook
    rospy.on_shutdown(shutdown_hook)
    
    rospy.loginfo("Depth data logger started. Waiting for data...")
    rospy.spin()