#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from perception.msg import depth_multiposition

# ====== Camera Parameters ======
fx = 350
baseline = 0.12  # meters
MIN_DEPTH = 0.3
MAX_DEPTH = 10.0
h, w = 360, 640  # Image dimensions

# ====== ROS Globals ======
bridge = CvBridge()
left_image = None
right_image = None

# ====== Stereo Matchers ======
window_size = 5
min_disp = 0
num_disp = 16 * 6  # Must be divisible by 16

left_matcher = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8 * 3 * window_size**2,
    P2=32 * 3 * window_size**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv.ximgproc.createRightMatcher(left_matcher)

wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(8000)
wls_filter.setSigmaColor(1.5)

# ====== Callback Functions ======
def left_image_call_back(data):
    global left_image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    left_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)

def right_image_call_back(data):
    global right_image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    right_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)

# ====== Utility Functions ======
def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return gray

def postprocess_disparity(disp):
    disp = disp.astype(np.float32) / 16.0
    disp = cv.medianBlur(disp, 5)
    kernel = np.ones((5, 5), np.uint8)
    disp = cv.morphologyEx(disp, cv.MORPH_CLOSE, kernel)
    disp[disp <= 0] = 0
    return disp



# ====== Main Processing Loop ======
def main_function():
    global left_image, right_image

    if left_image is None or right_image is None:
        return

    left_gray = preprocess_image(left_image)
    right_gray = preprocess_image(right_image)

    disp_left = left_matcher.compute(left_gray, right_gray)
    disp_right = right_matcher.compute(right_gray, left_gray)

    filtered_disp = wls_filter.filter(disp_left, left_image, disparity_map_right=disp_right)
    disparity = postprocess_disparity(filtered_disp)



    # Convert disparity to full depth map (in meters)
    with np.errstate(divide='ignore'):
        depth_image = (fx * baseline) / (disparity + 1e-6)
    depth_image[disparity <= 0] = 0

    # Publish full depth map as ROS Image
    depth_msg_img = bridge.cv2_to_imgmsg(depth_image.astype(np.float32), encoding="32FC1")
    depth_msg_img.header.stamp = rospy.Time.now()
    depth_msg_img.header.frame_id = "depth_frame"
    depth_map_pub.publish(depth_msg_img)


    valid_disp = disparity[disparity > 0]
    if len(valid_disp) > 0:
        rospy.loginfo(f"Disparity range: {valid_disp.min():.2f}-{valid_disp.max():.2f}")

# ====== Main Entry ======
if __name__ == "__main__":
    rospy.init_node("depth_estimation_node")
    rospy.Subscriber("/left_rect_image", Image, left_image_call_back)
    rospy.Subscriber("/right_rect_image", Image, right_image_call_back)
    depth_map_pub = rospy.Publisher("/depth_image", Image, queue_size=1)

    rate = rospy.Rate(18)
    while not rospy.is_shutdown():
        try:
            main_function()
        except Exception as e:
            rospy.logerr(f"Error: {str(e)}")
        rate.sleep()

    cv.destroyAllWindows()