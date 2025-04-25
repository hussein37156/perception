#!/usr/bin/env python3
import rospy
import std_msgs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

# Camera parameters (verified)
fx_left = 473.847
fy_left = 456.285
cx_left = 368.059
cy_left = 177.863

fx_right = 479.941
fy_right = 463.461
cx_right = 361.966
cy_right = 168.336

baseline = 0.12  # 12cm baseline

# Image dimensions
h, w = 360, 640

# Depth range limits (in meters)
MIN_DEPTH = 0.3  # 30cm
MAX_DEPTH = 10.0  # 10m

# Intrinsic matrix
K_left = np.array([[fx_left, 0, cx_left],
              [0, fy_left, cy_left],
              [0, 0, 1]])



K_right = np.array([[fx_right, 0, cx_right],
              [0, fy_right, cy_right],
              [0, 0, 1]])


D_left = np.zeros(5)  # Replace with real distortion if available
D_right = np.zeros(5)


R = np.eye(3)      # Replace with real rotation matrix
T = np.array([[baseline], [0], [0]])  # Translation along x-axis

R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(K_left, D_left, K_right, D_right, (w, h), R, T, alpha=0)

left_map1, left_map2 = cv.initUndistortRectifyMap(K_left, D_left, R1, P1, (w, h), cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(K_right, D_right, R2, P2, (w, h), cv.CV_16SC2)

# Stereo matcher configuration
window_size = 1
min_disp = 0
num_disp = 16*10  # Increased number of disparities

# Create stereo matcher
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

# ROS setup
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


def calculate_depth(disparity_map, u, v, window_size=50):
    """Calculate robust depth at (u,v) with validity checks"""
    half = window_size // 2
    u, v = int(u), int(v)
    
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
    depth = (fx_left * baseline) / median_disp if median_disp > 0 else 0.0
    
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
    
    # Resize and preprocess
    left_img =left_image #cv.resize(left_image, (w, h))
    right_img =right_image #cv.resize(right_image, (w, h))
    
    
    #left_img = cv.remap(left_image, left_map1, left_map2, cv.INTER_LINEAR)
    #right_img = cv.remap(right_image, right_map1, right_map2, cv.INTER_LINEAR)
    
    gray_left = preprocess_image(left_img)
    gray_right = preprocess_image(right_img)
    
    # Compute disparity
    raw_disp = stereo.compute(gray_left, gray_right)
    disparity = postprocess_disparity(raw_disp)
    
    # Calculate depth at multiple points for robustness
    center_u = int(w *0.5)
    center_v = int(h*0.65)
    
    # Sample multiple points around center
    points = [
        (center_u, center_v),
        (center_u + 50, center_v),
        (center_u - 50, center_v),
        (center_u, center_v + 50),
        (center_u, center_v - 50)
    ]
    
    depths = []
    for u, v in points:
        depth = calculate_depth(disparity, u, v)
        if MIN_DEPTH <= depth <= MAX_DEPTH:
            depths.append(depth)
    
    # Use median of valid depth measurements
    final_depth = np.median(depths) if len(depths) > 0 else 0.0
    
    # Publish depth
    depth_msg = std_msgs.msg.Float32()
    depth_msg.data = final_depth
    depth_pub.publish(depth_msg)
    
    # Visualization
    vis_disp = cv.normalize(disparity, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    #vis_disp = cv.applyColorMap(vis_disp, cv.COLORMAP_JET)
    
    # Mark center point
    cv.circle(left_img, (center_u, center_v), 5, (0, 255, 0), 2)
    cv.putText(left_img, f"Depth: {final_depth:.2f}m", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show images
    cv.imshow("Left", left_img)
    cv.imshow("Disparity", vis_disp)
    cv.waitKey(1)
    
    # Debug output
    valid_disp = disparity[disparity > 0]
    if len(valid_disp) > 0:
        rospy.loginfo(f"Disparity range: {valid_disp.min():.2f}-{valid_disp.max():.2f}")
    rospy.loginfo(f"Depth: {final_depth:.2f}m")
    
if __name__ == "__main__":
    rospy.init_node("depth_estimation_node")
    
    # Publishers and subscribers
    rospy.Subscriber("/left_rect_image", Image, left_image_call_back)
    rospy.Subscriber("/right_rect_image", Image, right_image_call_back)
    depth_pub = rospy.Publisher('/depth_estimated', std_msgs.msg.Float32, queue_size=10)
    
    rate = rospy.Rate(20)
    
    while not rospy.is_shutdown():
        try:
            main_function()
        except Exception as e:
            rospy.logerr(f"Error: {str(e)}")
        rate.sleep()
    
    cv.destroyAllWindows()