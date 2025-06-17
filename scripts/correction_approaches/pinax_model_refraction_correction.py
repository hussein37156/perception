#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

mapx_left = None
mapy_left = None
mapx_right = None
mapy_right = None

bridge = CvBridge()
left_image_pub = rospy.Publisher('/left_rect_image', Image, queue_size=10)
right_image_pub = rospy.Publisher('/right_rect_image', Image, queue_size=10)

# CLAHE object (can tune clipLimit and tileGridSize)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def init():
    mapPath = "/home/hussein/AUV_ws/src/perception/scripts/maps"
    global mapx_left, mapy_left , mapx_right, mapy_right
    mapx_left = np.loadtxt(f"{mapPath}/MapX_left.txt", dtype=np.float32, delimiter=",")
    mapy_left = np.loadtxt(f"{mapPath}/MapY_left.txt", dtype=np.float32, delimiter=",")
    mapx_right = np.loadtxt(f"{mapPath}/MapX_right.txt", dtype=np.float32, delimiter=",")
    mapy_right = np.loadtxt(f"{mapPath}/MapY_right.txt", dtype=np.float32, delimiter=",")
    
    
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
    global mapx_left, mapy_left, bridge
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    correctedImg = cv.remap(cv_image, mapx_left, mapy_left, cv.INTER_LINEAR)

    # Apply contrast enhancement
    enhanced = enhance_contrast(correctedImg)
    
    
    #resize
    enhanced = cv.resize(enhanced, (640, 360))
    

    msg = bridge.cv2_to_imgmsg(enhanced, encoding="bgr8")
    left_image_pub.publish(msg)

def right_image_call_back(data):
    global mapx_right, mapy_right, bridge
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    correctedImg = cv.remap(cv_image, mapx_right, mapy_right, cv.INTER_LINEAR)

    # Apply contrast enhancement
    enhanced = enhance_contrast(correctedImg)
    #resize
    enhanced = cv.resize(enhanced, (640, 360))
    

    msg = bridge.cv2_to_imgmsg(enhanced, encoding="bgr8")
    right_image_pub.publish(msg)

if __name__ == '__main__':
    init()
    rospy.spin()
