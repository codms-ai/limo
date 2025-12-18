#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineTracerWithObstacleAvoidance:
    def __init__(self):
        rospy.init_node("line_tracer_with_obstacle_avoidance")
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        self.scan_ranges = []
        self.front = 999.0

        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start = rospy.Time.now().to_sec()

        self.left_escape_count = 0
        self.force_right_escape = 0

        self.robot_width = 0.14

        # ===== 라인트레이싱 파라미터 =====
        self.base_gain = 1.0 / 220.0
        self.corner_scale = 140.0
        self.max_steer = 0.85

        # ===== 좌회전 지연 =====
        self.left_delay_start = None
        self.left_delay_time = 0.4

        # ===== 라인 최소 면적 (이보다 작으면 "안 보임") =====
        self.min_line_area = 300   # 핵심 파라미터

    # ============================================================
    # LIDAR
    # ============================================================
    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

    # ============================================================
    # CAMERA
    # ============================================================
    def camera_cb(self, msg):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if self.state == "ESCAPE":
            self.escape_control()
            return

        if self.state == "BACK":
            self.back_control()
            return

        if self.state == "LANE":

            if self.front < 0.36:
                self.state = "BACK"
                self.state_start = now
                self.left_delay_start = None
                return

            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            roi_near = frame[int(h * 0.55):h, :]
            hsv_near = cv2.cvtColor(roi_near, cv2.COLOR_BGR2HSV)

            # ==========================
            # 흰색 차선
            # ==========================
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])
            mask_near = cv2.inRange(hsv_near, lower_white, upper_white)

            contours_near, _ = cv2.findContours(
                mask_near, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            # ==========================
            # 라바콘 (빨간색)
            # ==========================
            lower_r1 = np.array([0, 120, 80])
            upper_r1 = np.array([10, 255, 255])
            lower_r2 = np.array([170, 120, 80])
            upper_r2 = np.array([180, 255, 255])

            mask_r1 = cv2.inRange(hsv_near, lower_r1, upper_r1)
            mask_r2 = cv2.inRange(hsv_near, lower_r2, upper_r2)
            red_mask = cv2.bitwise_or(mask_r1, mask_r2)

            red_contours, _ = cv2.findContours(
                red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            centers = []
            for cnt in red_contours:
                if cv2.contourArea(cnt) < 200:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                centers.append(int(M["m10"] / M["m00"]))

            # ==========================
            # 라바콘 주행
            # ==========================
            if len(centers) >= 1:
                if len(centers) >= 2:
                    centers.sort()
                    mid = (centers[0] + centers[-1]) // 2
                else:
                    mid = centers[0]

                error = mid - (w // 2)
                twist.linear.x = 0.21
                twist.angular.z = error / 180.0
                self.left_delay_start = None
                self.pub.publish(twist)
                return

            # =====================================================
            # 라인 안 보이면 무조건 직진
            # =====================================================
            if len(contours_near) == 0:
                twist.linear.x = 0.12
                twist.angular.z = 0.0
                self.left_delay_start = None
                self.pub.publish(twist)
                return

            c = max(contours_near, key=cv2.contourArea)

            if cv2.contourArea(c) < self.min_line_area:
                twist.linear.x = 0.12
                twist.angular.z = 0.0
                self.left_delay_start = None
                self.pub.publish(twist)
                return

            # ==========================
            # 라인트레이싱
            # ==========================
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - (w // 2)

                # 좌회전 지연
                if error > 0:
                    if self.left_delay_start is None:
                        self.left_delay_start = now

                    if now - self.left_delay_start < self.left_delay_time:
                        twist.linear.x = 0.22
                        twist.angular.z = 0.0
                        self.pub.publish(twist)
                        return
                else:
                    self.left_delay_start = None

                gain = self.base_gain * (1.0 + abs(error) / self.corner_scale)
                twist.linear.x = 0.22
                twist.angular.z = gain * error

                twist.angular.z = max(
                    min(twist.angular.z, self.max_steer),
                    -self.max_steer
                )

                self.pub.publish(twist)
                return

    # ============================================================
    # BACK / ESCAPE / GAP (변경 없음)
    # ============================================================
    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.4:
            twist.linear.x = -0.24
            twist.angular.z = 0.0
            self.pub.publish(twist)
        else:
            angle = self.find_gap_max()
            angle = self.apply_escape_direction_logic(angle)
            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            twist.linear.x = 0.19
            twist.angular.z = self.escape_angle * 1.3
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.9

        if angle < 0:
            self.left_escape_count += 1
            if self.left_escape_count >= 4:
                self.force_right_escape = 2
                self.left_escape_count = 0
        else:
            self.left_escape_count = 0

        return angle

    def find_gap_max(self):
        if len(self.scan_ranges) == 0:
            return 0.0

        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        max_dist = ranges[idx]

        if max_dist < (self.robot_width + 0.10):
            return 0.0

        angle_deg = idx - 60
        return angle_deg * np.pi / 180


if __name__ == "__main__":
    LineTracerWithObstacleAvoidance()
    rospy.spin()
