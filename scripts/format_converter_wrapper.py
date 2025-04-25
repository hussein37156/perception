#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()

def convert_and_publish(msg, publisher, cam_name):
    try:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        if cv_image.shape[2] == 4:
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)

        #resize the image to 640x480
        cv_image = cv2.resize(cv_image, (1280, 720))
        new_msg = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        new_msg.header = msg.header
        publisher.publish(new_msg)

    except Exception as e:
        rospy.logerr("[%s] Image conversion failed: %s", cam_name, str(e))


def camera1_callback(msg):
    convert_and_publish(msg, pub1, "camera1")


def camera2_callback(msg):
    convert_and_publish(msg, pub2, "camera2")


rospy.init_node('format_converter_node')

sub1 = rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, camera1_callback)
sub2 = rospy.Subscriber('/zed/zed_node/right/image_rect_color', Image, camera2_callback)

pub1 = rospy.Publisher('/left_image_converted', Image, queue_size=1)
pub2 = rospy.Publisher('/right_image_converted', Image, queue_size=1)

rospy.spin()
