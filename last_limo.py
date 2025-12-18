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

        # ===== 🔥 라인트레이싱 파라미터 수정 (속도 하향 및 부드러운 회전) =====
        self.speed_lane = 0.12     # 일반 주행 속도 (기존 0.22 -> 0.12)
        self.speed_cone = 0.12     # 라바콘 주행 속도 (기존 0.21 -> 0.12)
        
        # 분모를 키울수록 회전이 부드러워집니다 (220.0 -> 400.0)
        self.base_gain = 1.0 / 400.0 
        self.corner_scale = 160.0  # 코너 감도 조절
        self.max_steer = 0.60      # 최대 회전각 제한 (기존 0.85 -> 0.60)

        self.left_delay_start = None
        self.left_delay_time = 0.4
        self.min_line_area = 300

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        front_zone = np.concatenate([raw[:10], raw[-10:]])
        cleaned = [d for d in front_zone if d > 0.20 and not np.isnan(d)]
        self.front = np.median(cleaned) if cleaned else 999.0

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
            # ROI를 조금 더 높게 잡아 미리 대응 (0.55 -> 0.60)
            roi_near = frame[int(h * 0.60):h, :]
            hsv_near = cv2.cvtColor(roi_near, cv2.COLOR_BGR2HSV)

            # 1. 빨간색 라바콘 검출
            lower_r1, upper_r1 = np.array([0, 120, 80]), np.array([10, 255, 255])
            lower_r2, upper_r2 = np.array([170, 120, 80]), np.array([180, 255, 255])
            red_mask = cv2.bitwise_or(cv2.inRange(hsv_near, lower_r1, upper_r1), cv2.inRange(hsv_near, lower_r2, upper_r2))
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            centers = []
            for cnt in red_contours:
                if cv2.contourArea(cnt) < 200: continue
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    centers.append(int(M["m10"] / M["m00"]))

            if centers:
                mid = (sorted(centers)[0] + sorted(centers)[-1]) // 2 if len(centers) >= 2 else centers[0]
                error = mid - (w // 2)
                twist.linear.x = self.speed_cone
                # 🔥 라바콘 회전도 더 부드럽게 수정 (180.0 -> 350.0)
                twist.angular.z = error / 350.0 
                self.left_delay_start = None
                self.pub.publish(twist)
                return

            # 2. 흰색 차선 검출
            lower_white, upper_white = np.array([0, 0, 180]), np.array([180, 40, 255])
            mask_near = cv2.inRange(hsv_near, lower_white, upper_white)
            contours_near, _ = cv2.findContours(mask_near, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if not contours_near:
                twist.linear.x = 0.08
                twist.angular.z = 0.0
                self.pub.publish(twist)
                return

            c = max(contours_near, key=cv2.contourArea)
            if cv2.contourArea(c) < self.min_line_area:
                twist.linear.x = 0.08
                twist.angular.z = 0.0
                self.pub.publish(twist)
                return

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - (w // 2)

                # 좌회전 지연 로직
                if error > 0:
                    if self.left_delay_start is None: self.left_delay_start = now
                    if now - self.left_delay_start < self.left_delay_time:
                        twist.linear.x = self.speed_lane
                        twist.angular.z = 0.0
                        self.pub.publish(twist)
                        return
                else:
                    self.left_delay_start = None

                # 🔥 부드러운 가변 Gain 적용
                gain = self.base_gain * (1.0 + abs(error) / self.corner_scale)
                twist.linear.x = self.speed_lane
                twist.angular.z = gain * error

                # 회전 제한
                twist.angular.z = max(min(twist.angular.z, self.max_steer), -self.max_steer)
                self.pub.publish(twist)

    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        if now - self.state_start < 1.0: # 후진 시간 단축
            twist.linear.x = -0.15
            self.pub.publish(twist)
        else:
            angle = self.find_gap_max()
            self.escape_angle = self.apply_escape_direction_logic(angle)
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        if now - self.state_start < 0.8:
            twist.linear.x = 0.12
            twist.angular.z = self.escape_angle * 1.0
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    # apply_escape_direction_logic 및 find_gap_max는 기존 코드 유지
    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.6
        if angle < 0:
            self.left_escape_count += 1
            if self.left_escape_count >= 4:
                self.force_right_escape = 2
                self.left_escape_count = 0
        else: self.left_escape_count = 0
        return angle

    def find_gap_max(self):
        if not len(self.scan_ranges): return 0.0
        raw = np.array(self.scan_ranges)
        ranges = np.concatenate([raw[-60:], raw[:60]])
        ranges = np.where((ranges < 0.20) | np.isnan(ranges), 0.0, ranges)
        idx = np.argmax(ranges)
        if ranges[idx] < (self.robot_width + 0.10): return 0.0
        return (idx - 60) * np.pi / 180

if __name__ == "__main__":
    LineTracerWithObstacleAvoidance()
    rospy.spin()
