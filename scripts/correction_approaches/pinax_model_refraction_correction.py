#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import std_msgs

mapx = None
mapy = None
bridge = CvBridge()
left_image_pub = rospy.Publisher('/left_rect_image', Image, queue_size=10)
right_image_pub = rospy.Publisher('/right_rect_image', Image, queue_size=10)

# CLAHE object (can tune clipLimit and tileGridSize)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def init():
    mapPath = "/home/hussein/AUV_ws/src/auv_perception/scripts"
    global mapx, mapy
    height = 720 // 2
    width = 1280 // 2
    mapx = np.loadtxt(f"{mapPath}/MapX.txt", dtype=np.float32, delimiter=",")
    mapy = np.loadtxt(f"{mapPath}/MapY.txt", dtype=np.float32, delimiter=",")
    rospy.init_node('pinax_correction_node', anonymous=False)
    rospy.Subscriber("/zed/zed_node/left/image_rect_color", Image, left_image_call_back)
    rospy.Subscriber("/zed/zed_node/right/image_rect_color", Image, right_image_call_back)

def enhance_contrast(image_bgr):
    # Convert to LAB color space
    lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)

    # Apply CLAHE to the L-channel
    cl = clahe.apply(l)

    # Merge and convert back to BGR
    limg = cv.merge((cl, a, b))
    result = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return result

def left_image_call_back(data):
    global mapx, mapy, bridge
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    if len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
        left_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)
    else:
        left_image = cv_image.copy()

    correctedImg = cv.remap(left_image, mapx, mapy, cv.INTER_LINEAR)

    # Apply contrast enhancement
    enhanced = enhance_contrast(correctedImg)

    msg = bridge.cv2_to_imgmsg(enhanced, encoding="passthrough")
    left_image_pub.publish(msg)

def right_image_call_back(data):
    global mapx, mapy, bridge
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    if len(cv_image.shape) == 3 and cv_image.shape[2] == 4:
        right_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)
    else:
        right_image = cv_image.copy()

    correctedImg = cv.remap(right_image, mapx, mapy, cv.INTER_LINEAR)

    # Apply contrast enhancement
    enhanced = enhance_contrast(correctedImg)

    msg = bridge.cv2_to_imgmsg(enhanced, encoding="passthrough")
    right_image_pub.publish(msg)

if __name__ == '__main__':
    init()
    rospy.spin()
