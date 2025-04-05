#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import std_msgs


# =================== CAMERA PARAMETERS ===================
fx = 700.515
fy = 700.515
cx = 662.935
cy = 353.9215
k1, k2, p1, p2, k3 = -0.174335, 0.0267531, 0, 0, 0

h, w = 720, 1280  # Image resolution

# Intrinsic matrices (Physical Camera)
K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
K2 = K1.copy()
D = np.array([k1, k2, p1, p2, k3])
R = np.eye(3)
T = np.array([[12.0], [0.0], [0.0]])

# Pinax Model (Virtual Camera Parameters)
n_air, n_glass, n_water = 1.0, 1.5, 1.333
K1_virtual = np.array([[fx * (n_glass / n_water), 0, cx], [0, fy * (n_glass / n_water), cy], [0, 0, 1]])
K2_virtual = K1_virtual.copy()

# Stereo Rectification with Refraction Compensation
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K1_virtual, D, K2_virtual, D, (w, h), R, T)

stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=9,
    P1=8 * 3 * 9**2,
    P2=32 * 3 * 9**2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# =================== ROS SETUP ===================
bridge = CvBridge()
left_image, right_image = None, None
depth_pub = rospy.Publisher('/depth_estimated', std_msgs.msg.Float32, queue_size=10)
depth_msg = std_msgs.msg.Float32()

# =================== ROS CALLBACK ===================
def left_image_call_back(data):
    global left_image
    left_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

def right_image_call_back(data):
    global right_image
    right_image = bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")

# =================== REFRACTION CORRECTION FUNCTION ===================
def snell_law(theta_incident, n1, n2):
    """Applies Snell's Law safely."""
    sin_theta_refracted = np.clip((n1 / n2) * np.sin(theta_incident), -1, 1)
    return np.arcsin(sin_theta_refracted)

def correct_underwater_image(image, K, n_air=1.0, n_glass=1.5, n_water=1.333):
    """Applies refraction correction using Pinax model."""
    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    x_norm = (x - K[0, 2]) / K[0, 0]
    y_norm = (y - K[1, 2]) / K[1, 1]

    theta_air = np.arctan(np.sqrt(x_norm**2 + y_norm**2))
    theta_glass = snell_law(theta_air, n_air, n_glass)
    theta_water = snell_law(theta_glass, n_glass, n_water)

    x_new = K[0, 0] * np.tan(theta_water) * np.cos(np.arctan2(y_norm, x_norm)) + K[0, 2]
    y_new = K[1, 1] * np.tan(theta_water) * np.sin(np.arctan2(y_norm, x_norm)) + K[1, 2]

    return cv.remap(image, x_new.astype(np.float32), y_new.astype(np.float32), interpolation=cv.INTER_LINEAR)

# =================== DEPTH ESTIMATION FUNCTIONS ===================
def location(points_3D, u, v, region_size=50):
    """Extracts depth from a local region safely."""
    half_size = region_size // 2
    h, w, _ = points_3D.shape

    v_start, v_end = max(0, v - half_size), min(h, v + half_size)
    u_start, u_end = max(0, u - half_size), min(w, u + half_size)

    region = points_3D[v_start:v_end, u_start:u_end, 2]
    region = np.nan_to_num(region, nan=0.0, posinf=0.0, neginf=0.0)
    region = region[region != 0]

    return np.mean(region) if len(region) > 0 else np.nan

# =================== IMAGE PROCESSING FUNCTION ===================
def main_function():
    """Processes stereo images, applies refraction correction, and estimates depth."""
    global left_image, right_image

    if left_image is None or right_image is None:
        rospy.logwarn("No images received yet.")
        return

    left_corrected = correct_underwater_image(left_image, K1_virtual)
    right_corrected = correct_underwater_image(right_image, K2_virtual)

    disparity_map = stereo.compute(left_corrected, right_corrected).astype(np.float32) / 16.0
    disparity_map[disparity_map < 1] = np.nan

    mask = np.isnan(disparity_map).astype(np.uint8) * 255
    disparity_map_filled = cv.inpaint(np.nan_to_num(disparity_map, nan=0), mask, 5, cv.INPAINT_TELEA)

    B = 0.12
    f = K1_virtual[0, 0]
    depth_map = (B * f) / (disparity_map_filled + 1e-6)

    disparity_map_3d = np.expand_dims(disparity_map_filled, axis=-1)
    points_3D = cv.reprojectImageTo3D(disparity_map_3d, Q)

    if points_3D.ndim != 3 or points_3D.shape[2] != 3:
        rospy.logerr("Unexpected shape for points_3D: {}".format(points_3D.shape))
        return

    center_y = int(depth_map.shape[0] * 0.65)
    center_x = depth_map.shape[1] // 2
    depth = location(points_3D, center_x, center_y, region_size=50)

    if np.isnan(depth):
        rospy.logwarn("Depth estimation failed. Skipping depth publishing.")
    else:
        depth_msg.data = depth
        depth_pub.publish(depth_msg)

    depth_vis = cv.normalize(depth_map, None, 0, 255, cv.NORM_MINMAX)
    depth_vis = np.nan_to_num(depth_vis, nan=0, posinf=255, neginf=0)
    depth_vis = np.uint8(depth_vis)
    depth_vis_color = cv.applyColorMap(depth_vis, cv.COLORMAP_JET)

    cv.circle(depth_vis_color, (center_x, center_y), 10, (0, 255, 0), 2)
    cv.imshow("Corrected Left Image", left_corrected)
    cv.imshow("Disparity Map", depth_vis_color)
    cv.waitKey(1)

# =================== ROS MAIN LOOP ===================
if __name__ == "__main__":
    rospy.init_node("depth_estimation", anonymous=False)
    rospy.Subscriber("/left_image", Image, left_image_call_back)
    rospy.Subscriber("/right_image", Image, right_image_call_back)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        main_function()
        rate.sleep()
