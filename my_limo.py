#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineTracerWithObstacleAvoidance:
    def __init__(self): # __init__ 오타 수정됨
        rospy.init_node("line_tracer_with_obstacle_avoidance")
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)

        self.bridge = CvBridge()

        # 🔥 주행 속도 하향 (기존 0.15 -> 0.10)
        self.speed = 0.10

        self.scan_ranges = []
        self.front = 999.0

        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start = rospy.Time.now().to_sec()

        self.left_escape_count = 0
        self.force_right_escape = 0

        self.robot_width = 0.13

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
            if self.front < 0.40: # 장애물 감지 거리 소폭 단축
                self.state = "BACK"
                self.state_start = now
                return

            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            # 🔥 약간 더 먼 곳을 보도록 ROI 수정 (0.55 -> 0.6) : 미리 대응하여 부드러운 회전 유도
            roi = frame[int(h*0.6):h, :]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 1) 라바콘(빨간색) 검출
            lower_r1, upper_r1 = np.array([0, 120, 80]), np.array([10, 255, 255])
            lower_r2, upper_r2 = np.array([170, 120, 80]), np.array([180, 255, 255])
            red_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_r1, upper_r1), cv2.inRange(hsv, lower_r2, upper_r2))
            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(red_contours) >= 1:
                centers = []
                for cnt in red_contours:
                    if cv2.contourArea(cnt) < 200: continue
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        centers.append(int(M["m10"] / M["m00"]))

                if centers:
                    mid = (sorted(centers)[0] + sorted(centers)[-1]) // 2 if len(centers) >= 2 else centers[0]
                    error = mid - (w // 2)

                    # 🔥 라바콘 모드 속도 하향 및 회전 부드럽게 (180.0 -> 300.0)
                    twist.linear.x = 0.08
                    twist.angular.z = -error / 300.0 # 부호 수정: 좌우 반전 시 - 제거
                    self.pub.publish(twist)
                    return

            # 2) 흰색 라인 트레이싱
            lower_white, upper_white = np.array([0, 0, 180]), np.array([180, 40, 255])
            mask_line = cv2.inRange(hsv, lower_white, upper_white)
            contours, _ = cv2.findContours(mask_line, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                twist.linear.x = 0.05
                twist.angular.z = 0.2 # 회전량 반감
                self.pub.publish(twist)
                return

            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - w//2

                # 🔥 라인트레이싱 속도 하향 및 회전 부드럽게 (200.0 -> 400.0)
                twist.linear.x = 0.10
                twist.angular.z = -error / 400.0 
                self.pub.publish(twist)
            return

    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        if now - self.state_start < 1.0:
            twist.linear.x = -0.12 # 후진 속도 소폭 감소
            self.pub.publish(twist)
        else:
            angle = self.find_gap_max()
            self.escape_angle = self.apply_escape_direction_logic(angle)
            self.state = "ESCAPE"
            self.state_start = now

    def escape_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        if now - self.state_start < 0.8: # 탈출 시간 단축
            twist.linear.x = 0.10
            twist.angular.z = self.escape_angle * 1.0 # 회전 배수 하향
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    def apply_escape_direction_logic(self, angle):
        if self.force_right_escape > 0:
            self.force_right_escape -= 1
            return 0.5 # 우회전 각도 축소
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
    try:
        node = LineTracerWithObstacleAvoidance()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
