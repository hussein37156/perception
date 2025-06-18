#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
import threading
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# ====== Camera Parameters ======
baseline = 0.12  # meters
MIN_DEPTH = 0.3
MAX_DEPTH = 10.0
h, w = 360, 640

# ====== ROS Globals ======
bridge = CvBridge()
left_image = None
right_image = None
lock = threading.Lock()

# ====== Stereo Matchers ======
window_size = 5
min_disp = 0
num_disp = 16 * 6

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

# ====== Stereo Rectification with Real Parameters ======
def stereo_rectify():
    image_size = (w, h)

    K1 = np.array([[349.815, 0, 322.1375],
                   [0, 349.815, 166.01825],
                   [0, 0, 1]])
    D1 = np.array([-0.174432, 0.0266348, 0.0, 0.0])

    K2 = np.array([[350.2575, 0, 331.4675],
                   [0, 350.2575, 176.96075],
                   [0, 0, 1]])
    D2 = np.array([-0.174432, 0.0266348, 0.0, 0.0])

    R = np.eye(3)  # Replace with actual R from stereo calibration if available
    T = np.array([-baseline, 0, 0])

    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

    map1x, map1y = cv.initUndistortRectifyMap(K1, D1, R1, P1, image_size, cv.CV_16SC2)
    map2x, map2y = cv.initUndistortRectifyMap(K2, D2, R2, P2, image_size, cv.CV_16SC2)

    return map1x, map1y, map2x, map2y, Q, P1

map1x, map1y, map2x, map2y, Q, P1 = stereo_rectify()
fx = P1[0, 0]  # Use rectified projection fx from left camera

# ====== Callbacks ======
def left_image_call_back(data):
    global left_image
    try:
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        img_bgr = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)
        rectified = cv.remap(img_bgr, map1x, map1y, cv.INTER_LINEAR)
        with lock:
            left_image = rectified
    except Exception as e:
        rospy.logerr(f"Left image callback error: {e}")

def right_image_call_back(data):
    global right_image
    try:
        cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
        img_bgr = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)
        rectified = cv.remap(img_bgr, map2x, map2y, cv.INTER_LINEAR)
        with lock:
            right_image = rectified
    except Exception as e:
        rospy.logerr(f"Right image callback error: {e}")

# ====== Utility Functions ======
def preprocess_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    return gray

def postprocess_disparity(disp):
    disp = disp.astype(np.float32) / 16.0
    disp = cv.medianBlur(disp, 5)
    kernel = np.ones((5, 5), np.uint8)
    disp = cv.morphologyEx(disp, cv.MORPH_CLOSE, kernel)
    disp[disp <= 0] = 0
    return disp

# ====== Main Processing Thread ======
def processing_loop():
    global left_image, right_image

    rate = rospy.Rate(15)
    while not rospy.is_shutdown():
        with lock:
            l_img = left_image.copy() if left_image is not None else None
            r_img = right_image.copy() if right_image is not None else None

        if l_img is None or r_img is None:
            rate.sleep()
            continue

        try:
            left_gray = preprocess_image(l_img)
            right_gray = preprocess_image(r_img)

            disp_left = left_matcher.compute(left_gray, right_gray)
            disp_right = right_matcher.compute(right_gray, left_gray)

            filtered_disp = wls_filter.filter(disp_left, l_img, disparity_map_right=disp_right)
            disparity = postprocess_disparity(filtered_disp)

            with np.errstate(divide='ignore'):
                depth_image = (fx * baseline) / (disparity + 1e-6)
            depth_image[disparity <= 0] = 0
            depth_image[depth_image > MAX_DEPTH] = 0

            depth_msg = bridge.cv2_to_imgmsg(depth_image.astype(np.float32), encoding="32FC1")
            depth_msg.header.stamp = rospy.Time.now()
            depth_msg.header.frame_id = "depth_frame"
            depth_map_pub.publish(depth_msg)

        except Exception as e:
            rospy.logerr(f"Processing error: {e}")
        
        rate.sleep()

# ====== Main Entry ======
if __name__ == "__main__":
    rospy.init_node("depth_estimation_node")
    rospy.Subscriber("/left_rect_image", Image, left_image_call_back)
    rospy.Subscriber("/right_rect_image", Image, right_image_call_back)
    depth_map_pub = rospy.Publisher("/depth_image", Image, queue_size=1)

    processing_thread = threading.Thread(target=processing_loop)
    processing_thread.start()

    rospy.spin()
