#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from perception.msg import Multi_instance, Landmark
from cv_bridge import CvBridge
import pyzed.sl as sl
import numpy as np
import cv2  
import torch
from ultralytics import YOLO

# Initialize the node
rospy.init_node('camera_node')

# Initialize YOLO model on CPU
device = torch.device('cpu')
model_weights_path = "/home/hussein/AUV_ws/src/auv_perception/best.pt"
model = YOLO(model_weights_path)
model.to(device=device)

# Create ROS publishers
left_image_pub = rospy.Publisher('/zed/zed_node/left/image_rect_color', Image, queue_size=10)
right_image_pub = rospy.Publisher('/zed/zed_node/right/image_rect_color', Image, queue_size=10)
landmark_pub = rospy.Publisher('/landmarks', Multi_instance, queue_size=10)

# Initialize ZED camera
zed = sl.Camera()
rate = rospy.Rate(15)  # 15 Hz
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.ULTRA
init_params.coordinate_units = sl.UNIT.METER
init_params.depth_minimum_distance = 0.8
init_params.depth_maximum_distance = 10.0
init_params.camera_fps = 15

# Open the camera and check for errors
runtime_params = sl.RuntimeParameters()
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    print('Camera initialization failed')
    exit(-1)

# Create image and point cloud objects
depth_map = sl.Mat()
bridge = CvBridge()
left_image = sl.Mat()
right_image = sl.Mat()

# Get camera intrinsics
intrinsics = zed.get_camera_information().camera_configuration.calibration_parameters
f_x = intrinsics.left_cam.fx
f_y = intrinsics.left_cam.fy
c_x = intrinsics.left_cam.cx
c_y = intrinsics.left_cam.cy

def extract_region_and_average(depth_map, center_x, center_y, region_size=50):
    """ Extract a depth region and calculate the average depth. """
    u, v = int(center_x), int(center_y)
    half_size = region_size // 2
    region = depth_map[v - half_size:v + half_size, u - half_size:u + half_size]

    # Replace NaN and infinite values with 0
    region = np.nan_to_num(region, nan=0, posinf=5, neginf=0.8)
    region = region[region != 0]
    z = np.mean(region) if region.size > 0 else 0

    # Convert to real-world coordinates
    x = (u - c_x) * z / f_x
    y = (v - c_y) * z / f_y
    return x, y, z

def getInstanceName(instance_number):
    """ Return instance name from label index. """
    labels = ['gate', 'path marker', 'badge', 'gun', 'box', 'hand', 'barrel', 'note', 'phone', 'bottle', 
              'gman', 'bootlegger', 'axe', 'dollar', 'beer']
    return labels[instance_number] if instance_number < len(labels) else "unknown"


def yolo_model(img):
    """ Run YOLO inference, extract depth, and publish results. """
    print("Running YOLO detection...")
    try:
        if img is None or img.size == 0:
            rospy.logwarn("Empty image received.")
            return

        # Run YOLO inference with visualization enabled
        result = model.predict(source=img, show=False, conf=0.70)
        num_of_instances = result[0].boxes.data.size()[0]

        if num_of_instances == 0:
            return  # No detections, exit early

        # Retrieve depth map
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        depth_data = depth_map.get_data()
        
        landmarks_msg = Multi_instance()
        landmarks_msg.data = []  # Clear the message before filling it

        for i in range(num_of_instances):
            zed_x_top_left = int(result[0].boxes.data[i][0].item())
            zed_y_top_left = int(result[0].boxes.data[i][1].item())
            zed_x_bottom_right = int(result[0].boxes.data[i][2].item())
            zed_y_bottom_right = int(result[0].boxes.data[i][3].item())
            zed_u_mid = int((zed_x_top_left + zed_x_bottom_right) / 2)
            zed_v_mid = int((zed_y_top_left + zed_y_bottom_right) / 2)

            instance_type = getInstanceName(int(result[0].boxes.data[i][5].item()))
            confidence_level = result[0].boxes.data[i][4].item()

            # Extract depth
            l = Landmark()
            l.ID = instance_type
            l.x, l.y, l.z = extract_region_and_average(depth_data, zed_u_mid, zed_v_mid)

            landmarks_msg.data.append(l)

        landmarks_msg.header.stamp = rospy.Time.now()
        landmark_pub.publish(landmarks_msg)

    except Exception as e:
        rospy.logerr(f"Error in YOLO model: {e}")


print("Camera Node Running...")

while not rospy.is_shutdown():
    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        try:
            # Retrieve the left image
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            left_img_np = left_image.get_data()

            # Convert to RGB and run YOLO
            left_image_np_rgb = cv2.cvtColor(left_img_np, cv2.COLOR_BGRA2BGR)
            yolo_model(left_image_np_rgb)

            # Close the window when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

        except Exception as e:
            rospy.logerr(f"Error: {e}")

    rate.sleep()
