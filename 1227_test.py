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

        # 🔥 라바콘 구간 종료 판단 변수
        self.seen_lavacon = False       
        self.passed_lavacon_section = False 
        self.last_lavacon_time = rospy.Time.now().to_sec()

    def lidar_cb(self, scan):
        raw = np.array(scan.ranges)
        self.scan_ranges = raw

        # 정면 감지 (미션 #3 충돌 방지용)
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
            # 🔥 [미션 #3] 장애물 감지 거리 (0.50m로 넉넉하게)
            if self.front < 0.50:
                self.state = "BACK"
                self.state_start = now
                return

            frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            h, w = frame.shape[:2]
            roi = frame[int(h*0.55):h, :]   
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # --- 라바콘(빨간색) 검출 ---
            lower_r1 = np.array([0, 120, 80])
            upper_r1 = np.array([10, 255, 255])
            lower_r2 = np.array([170, 120, 80])
            upper_r2 = np.array([180, 255, 255])
            mask_r1 = cv2.inRange(hsv, lower_r1, upper_r1)
            mask_r2 = cv2.inRange(hsv, lower_r2, upper_r2)
            red_mask = cv2.bitwise_or(mask_r1, mask_r2)

            red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 라바콘이 보일 때
            if len(red_contours) >= 1:
                self.seen_lavacon = True            
                self.last_lavacon_time = now        
                
                centers = []
                for cnt in red_contours:
                    if cv2.contourArea(cnt) < 200: continue
                    M = cv2.moments(cnt)
                    if M["m00"] == 0: continue
                    cx = int(M["m10"] / M["m00"])
                    centers.append(cx)

                if len(centers) == 0: return

                if len(centers) >= 2:
                    centers = sorted(centers)
                    mid = (centers[0] + centers[-1]) // 2
                else:
                    mid = int(centers[0])

                error = mid - (w // 2)
                twist.linear.x = 0.13
                twist.angular.z = error / 180.0
                self.pub.publish(twist)
                return

            # --- 라바콘 없는 경우 (라인 주행) ---
            
            # 🔥 [미션 #4] 라바콘 안 본지 5초 지났으면 검은 선 모드
            if self.seen_lavacon and (now - self.last_lavacon_time > 5.0):
                self.passed_lavacon_section = True

            if self.passed_lavacon_section:
                # [미션 #5] 검은색 라인 (바닥)
                lower_line = np.array([0, 0, 0])
                upper_line = np.array([180, 255, 60]) 
            else:
                # [기본] 흰색 라인
                lower_line = np.array([0, 0, 180])
                upper_line = np.array([180, 40, 255])

            mask_line = cv2.inRange(hsv, lower_line, upper_line)
            contours, _ = cv2.findContours(mask_line, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                twist.linear.x = 0.06
                twist.angular.z = 0.4
                self.pub.publish(twist)
                return

            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] == 0: return
            cx = int(M["m10"] / M["m00"])
            error = cx - w//2

            twist.linear.x = 0.14
            twist.angular.z = error / 200.0
            self.pub.publish(twist)
            return

    def back_control(self):
        twist = Twist()
        now = rospy.Time.now().to_sec()
        if now - self.state_start < 0.8: # 후진 시간 약간 단축
            twist.linear.x = -0.15
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
        if now - self.state_start < 1.2:
            twist.linear.x = 0.10
            # 🔥 [미션 #3] 회전력 1.8배 (확실하게 꺾기)
            twist.angular.z = self.escape_angle * 1.8
            self.pub.publish(twist)
        else:
            self.state = "LANE"

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

    def find_gap_max(self):
        if len(self.scan_ranges) == 0: return 0.0
        raw = np.array(self.scan_ranges)
        
        # 🔥 [미션 #3 핵심 수정] 탐색 범위를 다시 90도로 복구! (이게 문제였음)
        ranges = np.concatenate([raw[-90:], raw[:90]])
        ranges = np.where((ranges < 0.15) | np.isnan(ranges), 0.0, ranges)

        idx = np.argmax(ranges)
        max_dist = ranges[idx]

        if max_dist < (self.robot_width + 0.05):
            return 0.0

        # 각도 계산도 90도 기준으로 수정
        angle_deg = idx - 90
        return angle_deg * np.pi / 180

if __name__ == "__main__":
    LineTracerWithObstacleAvoidance()
    rospy.spin()
