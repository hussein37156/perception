#!/usr/bin/env python3
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from auv_perception.msg import Multi_instance, Landmark
import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detection', anonymous=False)

        # Class variables
        self.landmarks_msg = Multi_instance()
        self.bridge = CvBridge()
        self.depth_map = None
        self.fx = 349.815
        self.fy = 349.815
        self.cx = 337.6375
        self.cy = 173.51825

        # Create Ros publishers
        self.landmark_pub = rospy.Publisher('/landmarks', Multi_instance,queue_size=10)

        # Create the subscribers
        rospy.Subscriber('/zed/zed_node/left/image_rect_color', Image, self.left_image_callback, queue_size=1)
        rospy.Subscriber('/zed/zed_node/depth/depth_registered', Image, self.depthmap_callback, queue_size=10)


        # Initialize the YOLO model
        #torch.cuda.set_device(0)
        self.device = torch.device('cpu')
        model_weights_path = "/home/hussein/AUV_ws/src/best.pt"
        self.model = YOLO(model_weights_path)
        self.model.to(device=self.device)
    
    def depthmap_callback(self, msg: Image):
        """ Callback for depthmap messages. """
        try:
            # Convert ROS Image to OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.depth_map = depth_image
        except Exception as e:
            rospy.logerr("Error processing depthmap: %s", e)

    def getInstanceName(self, instance_number):
        labels = ['gate', 'path marker', 'badge', 'gun', 'box', 'hand', 'barrel', 'note', 'phone', 'bottle', 'gman', 'bootlegger', 'axe', 'dollar', 'beer']
        return labels[instance_number]
    
    def location(self, u, v, region_size=50):
        """
        Extract a region from the depth map,
        remove NaN and infinite values,
        and calculate the average depth.
        """                
        half_size = region_size // 2
        region = self.depth_map[v - half_size:v + half_size, u - half_size:u + half_size]

        # Replace NaN and infinite values with 0
        region = np.nan_to_num(region, nan=0.0, posinf=0.0, neginf=0.0)
        region = region[region != 0]
        z = np.mean(region)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        return x,y,z

    def left_image_callback(self, image: Image):
        """ Callback for left image messages. """
        print("Running YOLO detection...")
        try:
            img = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
            #print(img.shape)
            if len(img) == 0:
                rospy.logwarn("Image not available.")
                return

            result = self.model.predict(source=img, show=False, conf=0.85)
    
            num_of_instances = result[0].boxes.data.size()[0]

            if num_of_instances == 0:
                return

            self.landmarks_msg.data = []
            for i in range(num_of_instances):
                x_top_left = int(result[0].boxes.data[i][0].item())
                y_top_left = int(result[0].boxes.data[i][1].item())
                x_bottom_right = int(result[0].boxes.data[i][2].item())
                y_bottom_right = int(result[0].boxes.data[i][3].item())
                u_mid = int((x_top_left + x_bottom_right) / 2)
                v_mid = int((y_top_left + y_bottom_right) / 2)
                instance_type = self.getInstanceName(int(result[0].boxes.data[i][5].item()))
                # confidence_level = result[0].boxes.data[i][4].item()
                
                l=Landmark()
                l.ID= instance_type
                l.x,l.y,l.z = self.location(u_mid, v_mid)
                
                self.landmarks_msg.data.append(l)
            
            self.landmarks_msg.header.stamp = rospy.Time.now()
            self.landmark_pub.publish(self.landmarks_msg)
            
        except Exception as e:
            rospy.logerr(f"Error in YOLO : {e}")


if __name__ == "__main__":
    collector = ObjectDetector()
    rospy.spin()
