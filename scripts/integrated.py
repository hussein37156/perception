#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer

class StereoDepthNode:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('stereo_depth_node', anonymous=True)
        
        # Paths - make sure these are correct
        mapPath = rospy.get_param("~map_path", "/home/hussein/AUV_ws/src/auv_perception/scripts")
        
        try:
            # Load calibration maps with error handling
            self.mapx_left = np.loadtxt(f"{mapPath}/MapX_left.txt", dtype=np.float32, delimiter=",")
            self.mapy_left = np.loadtxt(f"{mapPath}/MapY_left.txt", dtype=np.float32, delimiter=",")
            self.mapx_right = np.loadtxt(f"{mapPath}/MapX_right.txt", dtype=np.float32, delimiter=",")
            self.mapy_right = np.loadtxt(f"{mapPath}/MapY_right.txt", dtype=np.float32, delimiter=",")
        except Exception as e:
            rospy.logerr(f"Failed to load calibration maps: {e}")
            rospy.signal_shutdown("Calibration maps missing")
            return

        # Camera parameters
        self.fx = 473.847
        self.fy = 456.285
        self.cx = 368.059
        self.cy = 177.863
        self.baseline = 0.12  # 12 cm
        self.MIN_DEPTH = 0.3
        self.MAX_DEPTH = 10.0
        self.h, self.w = 360, 640

        # Bridge & Publishers
        self.bridge = CvBridge()
        self.depth_pub = rospy.Publisher("/depth_estimated", Float32, queue_size=1)
        
        # Track processing time for debugging
        self.last_time = rospy.Time.now()
        
        # CLAHE with lower memory usage
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # Reduced grid size

        # Stereo Matcher with less aggressive settings
        window_size = 5
        self.stereo = cv.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 6,  # Reduced from 10 to 6
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,  # Reduced from 5
            uniquenessRatio=15,
            speckleWindowSize=100,  # Reduced from 200
            speckleRange=2,  # Reduced from 3
            preFilterCap=31,  # Reduced from 63
            mode=cv.STEREO_SGBM_MODE_SGBM
        )

        # Subscribers with synchronization and smaller queue
        left_sub = Subscriber("/zed/zed_node/left/image_rect_color", Image, queue_size=1)
        right_sub = Subscriber("/zed/zed_node/right/image_rect_color", Image, queue_size=1)
        
        # More conservative synchronizer settings
        self.ats = ApproximateTimeSynchronizer(
            [left_sub, right_sub], 
            queue_size=5,  # Reduced from 10
            slop=0.05  # Tighter synchronization (reduced from 0.1)
        )
        self.ats.registerCallback(self.stereo_callback)
        
        # Track callback rate
        self.callback_count = 0
        self.last_log_time = rospy.Time.now()

    def enhance_contrast(self, img):
        try:
            lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
            l, a, b = cv.split(lab)
            l2 = self.clahe.apply(l)
            merged = cv.merge((l2, a, b))
            return cv.cvtColor(merged, cv.COLOR_LAB2BGR)
        except Exception as e:
            rospy.logwarn(f"Contrast enhancement failed: {e}")
            return img

    def calculate_depth(self, disparity_map, u, v, window_size=30):  # Reduced window size
        try:
            half = window_size // 2
            u, v = int(u), int(v)
            x_min = max(0, u - half)
            x_max = min(disparity_map.shape[1], u + half)
            y_min = max(0, v - half)
            y_max = min(disparity_map.shape[0], v + half)
            
            window = disparity_map[y_min:y_max, x_min:x_max]
            valid_disp = window[window > 0]
            
            if len(valid_disp) == 0:
                return 0.0
                
            median_disp = np.median(valid_disp)
            depth = (self.fx * self.baseline) / median_disp if median_disp > 0 else 0.0
            return np.clip(depth, self.MIN_DEPTH, self.MAX_DEPTH)
        except Exception as e:
            rospy.logwarn(f"Depth calculation failed: {e}")
            return 0.0

    def postprocess_disparity(self, disp):
        try:
            disp = disp.astype(np.float32) / 16.0
            disp = cv.medianBlur(disp, 3)  # Reduced kernel size from 5
            kernel = np.ones((3, 3), np.uint8)  # Reduced kernel size
            disp = cv.morphologyEx(disp, cv.MORPH_CLOSE, kernel)
            disp[disp <= 0] = 0
            return disp
        except Exception as e:
            rospy.logwarn(f"Disparity postprocessing failed: {e}")
            return np.zeros((self.h, self.w), dtype=np.float32)

    def stereo_callback(self, left_msg, right_msg):
        try:
            current_time = rospy.Time.now()
            
            # Rate limiting (optional)
            if (current_time - self.last_time).to_sec() < 0.05:  # ~20Hz max
                return
            self.last_time = current_time
            
            # Convert images with error handling
            try:
                left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
                right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
            except CvBridgeError as e:
                rospy.logwarn(f"Image conversion failed: {e}")
                return

            # Rectify images
            try:
                left = cv.remap(left, self.mapx_left, self.mapy_left, cv.INTER_LINEAR)
                right = cv.remap(right, self.mapx_right, self.mapy_right, cv.INTER_LINEAR)
            except Exception as e:
                rospy.logwarn(f"Image rectification failed: {e}")
                return

            # Resize
            left = cv.resize(left, (self.w, self.h))
            right = cv.resize(right, (self.w, self.h))

            # Contrast enhancement
            left = self.enhance_contrast(left)
            right = self.enhance_contrast(right)

            # Compute disparity
            try:
                disp_raw = self.stereo.compute(cv.cvtColor(left, cv.COLOR_BGR2GRAY),
                                          cv.cvtColor(right, cv.COLOR_BGR2GRAY))
                disp = self.postprocess_disparity(disp_raw)
            except Exception as e:
                rospy.logwarn(f"Disparity computation failed: {e}")
                return

            # Calculate depth at center and surrounding points
            cu, cv_center = int(self.w / 2), int(self.h / 2)
            points = [(cu, cv_center)]  # Just center point for simplicity
            
            depths = []
            for u, v in points:
                depth = self.calculate_depth(disp, u, v)
                if self.MIN_DEPTH <= depth <= self.MAX_DEPTH:
                    depths.append(depth)
            
            final_depth = 1000*np.median(depths) if depths else 0.0
            self.depth_pub.publish(Float32(int(final_depth)))
            
            self.callback_count += 1
            
            # Log processing rate occasionally
            if (current_time - self.last_log_time).to_sec() > 5.0:
                rate = self.callback_count / 5.0
                rospy.loginfo(f"Processing rate: {rate:.1f} Hz")
                self.callback_count = 0
                self.last_log_time = current_time
                
        except Exception as e:
            rospy.logerr(f"Error in stereo callback: {e}")

if __name__ == "__main__":
    try:
        node = StereoDepthNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass