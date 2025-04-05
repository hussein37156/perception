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

D = np.array([k1, k2, p1, p2, k3])

# Stereo parameters
R = np.eye(3)  #Rotation 
T = np.array([[12.0], [0.0], [0.0]])  # Translation

# ROS bridge
bridge = CvBridge()
left_image = None
right_image = None

# Stereo rectification
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K1, D, K2, D, (w, h), R, T)
print(R1)
print(R2)
print(P1)
print(P2)
print(Q)
print(roi1)
print(roi2)


# Stereo matcher
stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,  # Increased range
    blockSize=5,  # Smaller block size for finer details
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    disp12MaxDiff=1,
    uniquenessRatio=15,  # Increased for better uniqueness
    speckleWindowSize=50,  # Reduced to capture more details
    speckleRange=16,  # Lowered to preserve more valid disparities
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY  # Using 3-way SGBM for better accuracy
)


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




def location(points_3D, u, v, region_size=50):         
    half_size = region_size // 2

    # Extract Z-channel (depth values)
    region = points_3D[v - half_size:v + half_size, u - half_size:u + half_size, 2]

    # Replace NaN and infinite values with 0
    region = np.nan_to_num(region, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove zero values
    region = region[region != 0]

    # Compute mean depth, handle empty case
    z = np.mean(region) 
    return z



def main_function():

    #left_image_corrected = correct_underwater_image(left_image, K1,1,1.333)
    #right_image_corrected = correct_underwater_image(right_image, K2,1,1.333)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    left_image_corrected = clahe.apply(cv.cvtColor(left_image, cv.COLOR_BGR2GRAY))
    right_image_corrected = clahe.apply(cv.cvtColor(right_image, cv.COLOR_BGR2GRAY))
    # Compute disparity map (ensure correct data type)
    disparity_map = stereo.compute(left_image,right_image).astype(np.float32) / 16.0
    # Mask invalid disparity values
    mask = disparity_map > 0  # Valid disparities only

    # Compute 3D points
    points_3D = cv.reprojectImageTo3D(disparity_map, Q)
    
    #points_3D[~mask] = np.nan  # Remove invalid points

    # Get the central pixel
    center_y = int(disparity_map.shape[0] * 0.65)  # Middle height
    center_x = disparity_map.shape[1] // 2  # Middle width
    
    depth=location(points_3D, center_x, center_y, region_size=50)

    depth_msg.data=depth*-10
    depth_pub.publish(depth_msg)

        
    # Display results
    #cv.imshow("Left image", left_image)
    #cv.imshow("right image resized", left_image_resized)
    # Draw a circle at the center of the corrected left image
    
    center_coordinates = (left_image_corrected.shape[1] // 2, int(left_image_corrected.shape[0]*0.65))
    radius = 10
    color = (0, 255, 0)  # Green color in BGR
    thickness = 2
    cv.circle(left_image_corrected, center_coordinates, radius, color, thickness)
    cv.imshow("left image corrected", left_image_corrected)
    cv.imshow("Disparity Map", cv.normalize(disparity_map, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U))
    cv.waitKey(1)
    


    

if __name__ == "__main__":
    rospy.init_node("depth_estimation", anonymous=False)
    rospy.Subscriber("/left_image", Image, left_image_call_back)
    rospy.Subscriber("/right_image", Image, right_image_call_back)
    depth_pub = rospy.Publisher('/depth_estimated', std_msgs.msg.Float32, queue_size=10)
    depth_msg=std_msgs.msg.Float32()
    rate = rospy.Rate(20)  # 10 Hz processing loop
    while not rospy.is_shutdown():
        if left_image is not None and right_image is not None:
            main_function()
        else:
            rospy.logwarn("No images received")
        rate.sleep()

