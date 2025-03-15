#!/usr/bin/env python3
import rospy
import std_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def depth_map(data):
    bridge = CvBridge()
    depth_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    depth_msg.data = depth_image[960//2, 540//2]
    depth_pub.publish(depth_msg)





if __name__ == "__main__":
    rospy.init_node("depth_wrapper", anonymous=False)
    rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, depth_map)
    depth_pub = rospy.Publisher('/depth', std_msgs.msg.Float32, queue_size=10)
    depth_msg=std_msgs.msg.Float32()
    rate = rospy.Rate(15)  # 15 Hz processing loop
    while not rospy.is_shutdown():
        rate.sleep()
