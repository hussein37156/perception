#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from perception.msg import depth_multiposition
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

bridge = CvBridge()
depth_msg = Float32()


def region_average(depth_image, point, window_size):
    height, width = depth_image.shape[:2]
    y1 = max(0, point[1] - (window_size // 2))
    y2 = min(height, point[1] + (window_size // 2))
    x1 = max(0, point[0] - (window_size // 2))
    x2 = min(width, point[0] + (window_size // 2))

    region = depth_image[y1:y2, x1:x2]
    avg_depth = np.nanmean(region)  # Ignore NaNs
    return avg_depth

def depth_map(data):
    try:
        depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        height, width = depth_image.shape[:2]
        print(f"Depth image size: {width}x{height}")
        center=region_average(depth_image, (width // 2, height // 2),50)
        right = region_average(depth_image,(480, 180), 50)
        left = region_average(depth_image, (160, 180), 50)
        depth_msg= depth_multiposition()
        depth_msg.center = center
        depth_msg.right = right
        depth_msg.left = left
        depth_pub.publish(depth_msg)
    except CvBridgeError as e:
        rospy.logerr(f"CV Bridge error: {e}")

if __name__ == "__main__":
    rospy.init_node("depth_wrapper", anonymous=False)
    depth_pub = rospy.Publisher('/depth', depth_multiposition, queue_size=10)
    rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, depth_map)
    rate = rospy.Rate(15)  # 15 Hz

    while not rospy.is_shutdown():
        rate.sleep()

