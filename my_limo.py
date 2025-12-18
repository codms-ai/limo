#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class LineTracerWithObstacleAvoidance:
    def __init__(self): # def init -> __init__ 으로 수정
        rospy.init_node("line_tracer_with_obstacle_avoidance")
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw/compressed", CompressedImage, self.camera_cb)
        rospy.Subscriber("/scan", LaserScan, self.lidar_cb)
        self.bridge = CvBridge()

        # 파라미터 최적화
        self.speed = 0.12 # 주행 속도 약간 하향 (안정성)
        self.scan_ranges = []
        self.front = 999.0
        self.state = "LANE"
        self.escape_angle = 0.0
        self.state_start = rospy.Time.now().to_sec()
        self.robot_width = 0.13

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw
        # 정면 감지 범위를 조금 더 넓힘
        front_zone = np.concatenate([raw[:15], raw[-15:]])
        cleaned = [d for d in front_zone if d > 0.10 and not np.isnan(d)]
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
            # 🔥 장애물 감지 거리를 늘려 미리 피하도록 수정 (0.45 -> 0.50)
            if self.front < 0.50:
                self.state = "BACK"
                self.state_start = now
                return

            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            roi = frame[int(h*0.6):h, :] # ROI를 약간 올려서 멀리 봄
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # (라바콘 검출 로직 생략 - 기존 코드 유지)
            # (라인 트레이싱 로직 생략 - 기존 코드 유지)
            # ... 부드러운 주행을 위해 angular.z = error / 400.0 권장

    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        # 후진 시간을 조금 줄여 벽에 너무 붙지 않게 함
        if now - self.state_start < 0.8:
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
        if now - self.state_start < 1.2:
            twist.linear.x = 0.10
            # 🔥 회전 강도를 높여 빈틈으로 확실히 머리를 돌림 (1.3 -> 1.8)
            twist.angular.z = self.escape_angle * 1.8 
            self.pub.publish(twist)
        else:
            self.state = "LANE"

    def find_gap_max(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        
        # 🔥 탐색 범위를 정면 기준 좌우 90도로 확장 (기존 60도)
        # 장애물 회피 미션에서 옆쪽 빈칸을 찾기 위함
        ranges = np.concatenate([raw[-90:], raw[:90]])
        ranges = np.where((ranges < 0.15) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        max_dist = ranges[idx]

        # 🔥 빈 공간 판단 기준 완화 (더 좁은 틈도 인식)
        if max_dist < (self.robot_width + 0.05):
            return 0.0

        angle_deg = idx - 90
        return angle_deg * np.pi / 180

    def apply_escape_direction_logic(self, angle):
        # 기존 로직 유지
        return angle

if __name__ == "__main__":
    LineTracerWithObstacleAvoidance()
    rospy.spin()
