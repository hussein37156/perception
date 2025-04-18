#!/usr/bin/env python3
import rospy
import std_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

# Camera parameters for HD720
fx = 700.515
fy = 700.515
cx = 662.935
cy = 353.9215
k1, k2, p1, p2, k3 = -0.174335, 0.0267531, 0, 0, 0

h, w = 720, 1280

# Intrinsic and distortion matrices
K1 = np.array([[fx, 0, cx], 
               [0, fy, cy], 
               [0, 0, 1.0]])

K2 = K1.copy()  # Assume both cameras have same intrinsic parameters



# ROS bridge
bridge = CvBridge()
left_image = None
right_image = None


def left_image_call_back(data):
    global left_image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")  # Keep original format

    if len(cv_image.shape) == 3 and cv_image.shape[2] == 4:  # RGBA (8UC4) detected
        left_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)  # Convert to BGR
    else:
        left_image = cv_image.copy()  # Already in compatible format
        

def right_image_call_back(data):
    global right_image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

    if len(cv_image.shape) == 3 and cv_image.shape[2] == 4:  # RGBA (8UC4) detected
        right_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)  # Convert to BGR
    else:
        right_image = cv_image.copy()


def correct_underwater_image(image, K, n_air, n_water):
    h, w = image.shape[:2]
    new_image = np.zeros_like(image)
    n = n_air / n_water  # Refractive index ratio

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Normalize pixel coordinates
    x_norm = (x - K[0, 2]) / K[0, 0]
    y_norm = (y - K[1, 2]) / K[1, 1]

    # Compute refraction angles using Snell's Law
    q_air = np.arctan(np.sqrt(x_norm**2 + y_norm**2)/fx)
    # Correction term
    correction_factor = np.sqrt((1 - (n * np.sin(q_air))**2) / (1 - np.sin(q_air)**2)) / n
    # Compute new pixel locations
    x_new = correction_factor * (x - K[0, 2]) + K[0, 2]
    y_new = correction_factor * (y - K[1, 2]) + K[1, 2]

    # Ensure valid indices
    valid_mask = (x_new >= 0) & (x_new < w) & (y_new >= 0) & (y_new < h)
    # Apply transformation using bilinear interpolation
    new_image[valid_mask] = cv.remap(image, x_new.astype(np.float32), y_new.astype(np.float32), cv.INTER_LINEAR)[valid_mask]

    return new_image





def main_function():

    left_image_corrected = correct_underwater_image(left_image, K1,1,1.333)
    right_image_corrected = correct_underwater_image(right_image, K2,1,1.333)
    left_corrected_msg = bridge.cv2_to_imgmsg(left_image_corrected, encoding="passthrough")
    right_corrected_msg = bridge.cv2_to_imgmsg(right_image_corrected, encoding="passthrough")
    left_corrected_pub.publish(left_corrected_msg)
    right_corrected_pub.publish(right_corrected_msg)
    
    

    


    

if __name__ == "__main__":
    rospy.init_node("image_correction", anonymous=False)
    rospy.Subscriber("/left_image", Image, left_image_call_back)
    rospy.Subscriber("/right_image", Image, right_image_call_back)
    left_corrected_pub = rospy.Publisher('/left_corrected', Image, queue_size=10)
    right_corrected_pub = rospy.Publisher('/right_corrected', Image, queue_size=10)
    left_corrected_msg = Image()
    right_corrected_msg = Image()
    rate = rospy.Rate(15)  # 10 Hz processing loop
    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            main_function()
        else:
            rospy.logwarn("No images received")
        rate.sleep()

