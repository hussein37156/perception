#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSquareOverlay:
    def __init__(self):
        rospy.init_node('image_square_overlay', anonymous=True)
        
        # Parameters
        self.input_topic = rospy.get_param("~input_topic", "/zed/zed_node/left/image_rect_color")
        self.output_topic = rospy.get_param("~output_topic", "/image_with_square")
        self.publish_modified = rospy.get_param("~publish_modified", False)  # Set to False for visualization only
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.input_topic, Image, self.image_callback)
        if self.publish_modified:
            self.image_pub = rospy.Publisher(self.output_topic, Image, queue_size=1)
        
        rospy.loginfo("Subscribed to %s", self.input_topic)
        if self.publish_modified:
            rospy.loginfo("Publishing modified images to %s", self.output_topic)
        else:
            rospy.loginfo("Visualizing image only, not publishing")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width, _ = cv_image.shape
        
        # Define red square in center
        square_size = 20
        top_left = (width // 2 - square_size // 2, height // 2 - square_size // 2)
        bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
        
        # Draw red square
        cv2.rectangle(cv_image, top_left, bottom_right, (0, 0, 255), -1)
        
        if self.publish_modified:
            image_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.image_pub.publish(image_msg)
        else:
            cv2.imshow("Live with Red Square", cv_image)
            cv2.waitKey(1)

if __name__ == '__main__':
    try:
        node = ImageSquareOverlay()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
