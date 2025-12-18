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

        # 라인트레이싱 속도
        self.speed = 0.15

        # 라이다 정보
        self.scan_ranges = []
        self.front = 999.0

        # 상태
        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start = rospy.Time.now().to_sec()

        # ESCAPE 방향 조정 변수
        self.left_escape_count = 0
        self.force_right_escape = 0

        # 차폭 (13cm)
        self.robot_width = 0.13   

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
    # CAMERA (핵심: 라바콘 우선)
    # ============================================================
    def camera_cb(self, msg):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        # ESCAPE 모드
        if self.state == "ESCAPE":
            self.escape_control()
            return

        # BACK 모드
        if self.state == "BACK":
            self.back_control()
            return

        # ------------------------ LANE ------------------------
        if self.state == "LANE":

            # 장애물 → BACK
            if self.front < 0.45:
                self.state = "BACK"
                self.state_start = now
                return

            # 카메라 이미지 읽기
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            roi = frame[int(h*0.55):h, :]   # 아래 45%만 사용 (라바콘 + 라인)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # ================================================
            # 🔥 1) 라바콘(빨간색) 검출
            # ================================================
            lower_r1 = np.array([0, 120, 80])
            upper_r1 = np.array([10, 255, 255])
            lower_r2 = np.array([170, 120, 80])
            upper_r2 = np.array([180, 255, 255])

            mask_r1 = cv2.inRange(hsv, lower_r1, upper_r1)
            mask_r2 = cv2.inRange(hsv, lower_r2, upper_r2)
            red_mask = cv2.bitwise_or(mask_r1, mask_r2)

            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # ================================================
            # 🔥 라바콘 보이면 → 라인트레이싱 무조건 OFF
            # ================================================
            if len(red_contours) >= 1:
                # 라바콘 중심점 구하기
                centers = []
                for cnt in red_contours:
                    area = cv2.contourArea(cnt)
                    if area < 200:  # 너무 작은건 노이즈
                        continue
                    M = cv2.moments(cnt)
                    if M["m00"] == 0:
                        continue
                    cx = int(M["m10"] / M["m00"])
                    centers.append(cx)

                if len(centers) == 0:
                    return

                # 🔥 라바콘이 2개 이상 → 두 중심의 평균으로 주행
                if len(centers) >= 2:
                    centers = sorted(centers)
                    mid = (centers[0] + centers[-1]) // 2
                else:
                    # 🔥 라바콘 1개 → 화면 중앙 기준으로 보정
                    mid = int(centers[0])

                error = mid - (w // 2)

                twist.linear.x = 0.13
                twist.angular.z = error / 180.0

                self.pub.publish(twist)
                return

            # ================================================
            # 🔥 2) 라바콘 없는 경우 → 기존 라인트레이싱
            # ================================================

            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 40, 255])
            mask_line = cv2.inRange(hsv, lower_white, upper_white)

            contours, _ = cv2.findContours(mask_line, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                twist.linear.x = 0.06
                twist.angular.z = 0.4
                self.pub.publish(twist)
                return

            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] == 0:
                return

            cx = int(M["m10"] / M["m00"])
            error = cx - w//2

            twist.linear.x = 0.14
            twist.angular.z = error / 200.0
            self.pub.publish(twist)
            return

    # ============================================================
    # BACK MODE
    # ============================================================
    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.2:
            twist.linear.x = -0.15
            twist.angular.z = 0.0
            self.pub.publish(twist)
        else:
            # ESCAPE 각도 계산
            angle = self.find_gap_max()
            angle = self.apply_escape_direction_logic(angle)

            self.escape_angle = angle
            self.state = "ESCAPE"
            self.state_start = now

    # ============================================================
    # ESCAPE MODE
    # ============================================================
    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()

        if now - self.state_start < 1.0:
            twist.linear.x = 0.12
            twist.angular.z = self.escape_angle * 1.3
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    # ============================================================
    # ESCAPE 방향 로직
    # ============================================================
    def apply_escape_direction_logic(self, angle):

        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.7

        if angle < 0:
            self.left_escape_count += 1

            if self.left_escape_count >= 4:
                self.force_right_escape = 2
                self.left_escape_count = 0
        else:
            self.left_escape_count = 0

        return angle

    # ============================================================
    # 가장 뚫린 방향 찾기
    # ============================================================
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
        angle_rad = angle_deg * np.pi / 180

        return angle_rad


if __name__ == "__main__":
    LineTracerWithObstacleAvoidance()
    rospy.spin()
