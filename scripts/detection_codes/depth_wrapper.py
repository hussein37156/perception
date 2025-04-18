#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

bridge = CvBridge()
depth_msg = Float32()

def depth_map(data):
    try:
        depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        height, width = depth_image.shape[:2]

        # Define a 10x10 region around the center
        window_size = 10
        y1 = max(0, height // 2 - window_size//2)
        y2 = min(height, height // 2 + window_size//2)
        x1 = max(0, width // 2 - window_size//2)
        x2 = min(width, width // 2 + window_size//2)
        center_region = depth_image[y1:y2, x1:x2]

        # Average the depth in this region, ignoring NaNs
        avg_depth = np.nanmean(center_region)
        depth_msg.data = float(avg_depth)
        depth_pub.publish(depth_msg)

    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge error: {e}")
    except Exception as e:
        depth_msg.data = 0.0
        depth_pub.publish(depth_msg)

if __name__ == "__main__":
    rospy.init_node("depth_wrapper", anonymous=False)
    depth_pub = rospy.Publisher('/depth', Float32, queue_size=10)
    rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, depth_map)
    rate = rospy.Rate(15)  # 15 Hz

    while not rospy.is_shutdown():
        rate.sleep()

