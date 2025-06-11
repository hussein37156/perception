#!/usr/bin/env python3
import rospy
import std_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
from perception.msg import depth_multiposition

# Camera parameters (verified)
fx = 350
fy = 350


baseline = 0.12  # 12cm baseline

# Image dimensions
h, w = 360, 640

# Depth range limits (in meters)
MIN_DEPTH = 0.3  # 30cm
MAX_DEPTH = 10.0  # 10m



# Stereo matcher configuration
window_size = 2
min_disp = 0
num_disp = 18*10  # Increased number of disparities


# ROS setup
bridge = CvBridge()
left_image = None
right_image = None


stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=window_size,
    P1=8*3*window_size**2,
    P2=32*3*window_size**2,
    disp12MaxDiff=5,  # Increased tolerance
    uniquenessRatio=10,
    speckleWindowSize=200,  # Larger speckle window
    speckleRange=3,
    preFilterCap=63,
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
)

def left_image_call_back(data):
    global left_image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")  # Keep original format
    left_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)  # Convert to BGR


def right_image_call_back(data):
    global right_image
    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
    right_image = cv.cvtColor(cv_image, cv.COLOR_RGBA2BGR)  # Convert to BGR

def calculate_depth(disparity_map, point, window_size=50):
    """Calculate robust depth at (u,v) with validity checks"""
    half = window_size // 2
    u, v = int(point[0]), int(point[1])

    # Get valid window bounds
    x_min = max(0, u - half)
    x_max = min(disparity_map.shape[1], u + half)
    y_min = max(0, v - half)
    y_max = min(disparity_map.shape[0], v + half)
    
    # Extract disparities in window
    disp_window = disparity_map[y_min:y_max, x_min:x_max]
    
    # Filter invalid disparities (values <= 0)
    valid_disp = disp_window[disp_window > 0]
    
    if len(valid_disp) == 0:
        return 0.0
    
    # Calculate median disparity (more robust than mean)
    median_disp = np.median(valid_disp)
    
    # Calculate depth with physical limits
    depth = (fx * baseline) / median_disp if median_disp > 0 else 0.0
    
    # Apply depth range limits
    depth = np.clip(depth, MIN_DEPTH, MAX_DEPTH)
    
    return depth

def preprocess_image(img):
    """Enhanced preprocessing pipeline"""
    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply CLAHE for adaptive histogram equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Apply bilateral filter
    gray = cv.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    return gray

def postprocess_disparity(disp):
    """Clean up disparity map"""
    # Convert to float32 and scale
    disp = disp.astype(np.float32) / 16.0
    
    # Apply median filter to reduce noise
    disp = cv.medianBlur(disp, 5)
    
    # Apply morphological closing
    kernel = np.ones((5,5), np.uint8)
    disp = cv.morphologyEx(disp, cv.MORPH_CLOSE, kernel)
    
    # Set invalid disparities to 0
    disp[disp <= 0] = 0
    
    return disp

def main_function():
    global left_image, right_image
    
    if left_image is None or right_image is None:
        return
    
    
    # Compute disparity
    raw_disp = stereo.compute(left_image, right_image)
    disparity = postprocess_disparity(raw_disp)
    
    # Sample multiple points around center
    center = calculate_depth(disparity, (w*0.5, h*0.5), 50)*1000
    right = calculate_depth(disparity, (w*0.75, h*0.5), 50)
    left = calculate_depth(disparity, (w*0.35, h*0.5), 50)


    depth_msg = depth_multiposition()
    depth_msg.center = center
    depth_msg.right = right
    depth_msg.left = left

    depth_pub.publish(depth_msg)

    # Visualization
    vis_disp = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    #vis_disp = cv.applyColorMap(vis_disp, cv.COLORMAP_JET)
    
    #draw circles on the depth points
    cv.circle(left_image, (int(w*0.5), int(h*0.5)), 10, (0, 255, 0), -1)
    cv.circle(left_image, (int(w*0.65), int(h*0.5)), 10, (255, 0, 0), -1)
    cv.circle(left_image, (int(w*0.35), int(h*0.5)), 10, (0, 0, 255), -1)


    
    # Show images
    #invert color order for display
    cv.imshow("Left", left_image)
    cv.imshow("Right", right_image)
    cv.imshow("Disparity", vis_disp)
    cv.waitKey(1)
    
    # Debug output
    valid_disp = disparity[disparity > 0]
    if len(valid_disp) > 0:
        rospy.loginfo(f"Disparity range: {valid_disp.min():.2f}-{valid_disp.max():.2f}")
    
if __name__ == "__main__":
    rospy.init_node("depth_estimation_node")
    
    # Publishers and subscribers
    rospy.Subscriber("/left_rect_image", Image, left_image_call_back)
    rospy.Subscriber("/right_rect_image", Image, right_image_call_back)
    depth_pub = rospy.Publisher('/depth_estimated', depth_multiposition, queue_size=10)
    
    rate = rospy.Rate(18)
    
    while not rospy.is_shutdown():
        try:
            main_function()
        except Exception as e:
            rospy.logerr(f"Error: {str(e)}")
        rate.sleep()
    
    cv.destroyAllWindows()
