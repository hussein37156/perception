#!/usr/bin/env python3
import pyzed.sl as sl
import rospy
import std_msgs
from sensor_msgs.msg import Image
import std_msgs.msg
from cv_bridge import CvBridge

rospy.init_node('camera_node')

left_image_pub = rospy.Publisher('/left_image', Image, queue_size=10)
right_image_pub = rospy.Publisher('/right_image', Image, queue_size=10)
depth_map_pub = rospy.Publisher('/depth_map', Image, queue_size=10)
depth_pub = rospy.Publisher('/depth', std_msgs.msg.Float32, queue_size=10)

zed = sl.Camera()
init_parameters = sl.InitParameters(
                                camera_resolution=sl.RESOLUTION.HD720, 
                                depth_mode=sl.DEPTH_MODE.ULTRA,
                                coordinate_units=sl.UNIT.METER, 
                                camera_fps=15,
                                depth_minimum_distance = 0.25,
                                depth_maximum_distance = 10.0
                                )



runtime_parameters =sl.RuntimeParameters()


err = zed.open(init_parameters)
if err != sl.ERROR_CODE.SUCCESS:
    print('Camera initialization failed')
    exit(-1)

left_image = sl.Mat()
right_image = sl.Mat()
depth_map = sl.Mat()

bridge = CvBridge()

print("Camera Node Running...")

# Inside your loop where you grab data from the ZED camera
while not rospy.is_shutdown():
    if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
        # Retrieve the image and point cloud
        zed.retrieve_image(left_image, sl.VIEW.LEFT)
        zed.retrieve_image(right_image, sl.VIEW.RIGHT)
        zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH)
        
        # Convert to ROS messages
        left_image_msg = bridge.cv2_to_imgmsg(left_image.get_data(), encoding="passthrough")
        right_image_msg = bridge.cv2_to_imgmsg(right_image.get_data(), encoding="passthrough")
        depth_map_msg = bridge.cv2_to_imgmsg(depth_map.get_data(), encoding="passthrough")
        depth_msg=std_msgs.msg.Float32()
        
        x = int(left_image.get_width() / 2)
        y = int(left_image.get_height() / 2)
        _,depth_msg.data = depth_map.get_value(x, y)
        
        # Publish the image
        left_image_pub.publish(left_image_msg)
        right_image_pub.publish(right_image_msg)
        depth_map_pub.publish(depth_map_msg)
        depth_pub.publish(depth_msg)
    
    rospy.sleep(0.05)
