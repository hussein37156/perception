#!/usr/bin/env python3
import rospy
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from cv_bridge import CvBridge, CvBridgeError
from message_filters import Subscriber, ApproximateTimeSynchronizer
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import atexit

class StereoDepthNode:
    def __init__(self):
        rospy.init_node('stereo_depth_node', anonymous=True)

        mapPath = rospy.get_param("~map_path", "/home/hussein/AUV_ws/src/perception/scripts")
        
        try:
            self.mapx_left = np.loadtxt(f"{mapPath}/MapX_left.txt", dtype=np.float32, delimiter=",")
            self.mapy_left = np.loadtxt(f"{mapPath}/MapY_left.txt", dtype=np.float32, delimiter=",")
            self.mapx_right = np.loadtxt(f"{mapPath}/MapX_right.txt", dtype=np.float32, delimiter=",")
            self.mapy_right = np.loadtxt(f"{mapPath}/MapY_right.txt", dtype=np.float32, delimiter=",")
        except Exception as e:
            rospy.logerr(f"Failed to load calibration maps: {e}")
            rospy.signal_shutdown("Calibration maps missing")
            return

        self.fx = 473.847
        self.fy = 456.285
        self.cx = 368.059
        self.cy = 177.863
        self.baseline = 0.12
        self.MIN_DEPTH = 0.3
        self.MAX_DEPTH = 10.0
        self.h, self.w = 360, 640

        self.bridge = CvBridge()
        self.depth_pub = rospy.Publisher("/depth_estimated", Float32, queue_size=1)

        self.last_time = rospy.Time.now()
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        window_size = 7
        self.stereo = cv.StereoSGBM_create(
            minDisparity=0,
            numDisparities=40,
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=31,
            mode=cv.STEREO_SGBM_MODE_SGBM
        )

        left_sub = Subscriber("/zed/zed_node/left/image_rect_color", Image, queue_size=1)
        right_sub = Subscriber("/zed/zed_node/right/image_rect_color", Image, queue_size=1)

        self.ats = ApproximateTimeSynchronizer(
            [left_sub, right_sub],
            queue_size=5,
            slop=0.05
        )
        self.ats.registerCallback(self.stereo_callback)

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

    def calculate_depth(self, disparity_map, u, v, window_size=30):
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
            disp = cv.medianBlur(disp, 3)
            kernel = np.ones((3, 3), np.uint8)
            disp = cv.morphologyEx(disp, cv.MORPH_CLOSE, kernel)
            disp[disp <= 0] = 0
            return disp
        except Exception as e:
            rospy.logwarn(f"Disparity postprocessing failed: {e}")
            return np.zeros((self.h, self.w), dtype=np.float32)

    def plot_depth_profile(self, disparity_map):
        try:
            # Zero out disparity in the specified horizontal ranges
            disparity_map[:, 0:150] = 0
            disparity_map[:, 450:640] = 0

            mid_v = self.h // 2
            step = 10
            u_coords = list(range(0, 639, step))
            depths = []

            for u in u_coords:
                depth = self.calculate_depth(disparity_map, u, mid_v, window_size=10)
                depths.append(depth)

            # Convert to numpy for filtering
            depths = np.array(depths)

            # Remove outliers using Z-score
            mean_depth = np.mean(depths)
            std_depth = np.std(depths)
            z_scores = (depths - mean_depth) / (std_depth + 1e-6)
            depths[np.abs(z_scores) > 2] = 0.0  # Zero out outliers beyond Z-score of 2

            # Plot
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(u_coords, depths, label="Depth (m)")
            ax.set_xlabel("Horizontal Pixel (u)")
            ax.set_ylabel("Depth (m)")
            ax.set_title("Depth vs Horizontal Pixel at Midline (Outliers Removed)")
            ax.grid(True)
            ax.legend()
            fig.tight_layout()

            # Convert plot to OpenCV image
            canvas = FigureCanvas(fig)
            canvas.draw()
            img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            cv.imshow("Depth Profile", img)
            cv.waitKey(1)
            plt.close(fig)

        except Exception as e:
            rospy.logwarn(f"Depth profile plotting failed: {e}")




    def stereo_callback(self, left_msg, right_msg):
        try:
            current_time = rospy.Time.now()
            if (current_time - self.last_time).to_sec() < 0.05:
                return
            self.last_time = current_time

            try:
                left = self.bridge.imgmsg_to_cv2(left_msg, "bgr8")
                right = self.bridge.imgmsg_to_cv2(right_msg, "bgr8")
            except CvBridgeError as e:
                rospy.logwarn(f"Image conversion failed: {e}")
                return

            try:
                left = cv.remap(left, self.mapx_left, self.mapy_left, cv.INTER_LINEAR)
                right = cv.remap(right, self.mapx_right, self.mapy_right, cv.INTER_LINEAR)
            except Exception as e:
                rospy.logwarn(f"Image rectification failed: {e}")
                return

            left = cv.resize(left, (self.w, self.h))
            right = cv.resize(right, (self.w, self.h))

            left = self.enhance_contrast(left)
            right = self.enhance_contrast(right)

            try:
                disp_raw = self.stereo.compute(cv.cvtColor(left, cv.COLOR_BGR2GRAY),
                                               cv.cvtColor(right, cv.COLOR_BGR2GRAY))
                disp = self.postprocess_disparity(disp_raw)
            except Exception as e:
                rospy.logwarn(f"Disparity computation failed: {e}")
                return

            cu, cv_center = int(self.w * 0.5), int(self.h * 0.5)
            depth = self.calculate_depth(disp, cu, cv_center)
            final_depth = 1000 * depth if self.MIN_DEPTH <= depth <= self.MAX_DEPTH else 0.0
            self.depth_pub.publish(Float32(int(final_depth)))

            self.plot_depth_profile(disp)

            self.callback_count += 1
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
        atexit.register(cv.destroyAllWindows)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
