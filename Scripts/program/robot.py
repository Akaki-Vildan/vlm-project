import math
import sys
import time
import atexit
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from pulseapi import RobotPulse, position, MT_LINEAR, MT_JOINT, tool_info
from utils import DetectedObject

TABLE_Z = 0.03
TARGET_OFFSET_Z = 0.01
H_tcp_to_cam_file = np.array([
    [1.0, 0.0, 0.0, -0.014],
    [0.0, 1.0, 0.0,  0.05853],
    [0.0, 0.0, 1.0,  0.068],
    [0.0, 0.0, 0.0,  1.0],
])
H_cam_to_tcp = np.linalg.inv(H_tcp_to_cam_file)
HOME_POS = position([-0.4, 0.0, 0.5], [math.pi, 0, -math.pi / 2])

class RobotController:
    def __init__(self, host="http://10.10.10.20:8081"):
        self.host = host
        self.robot = None
        self.move_velocity = 0.1

    def connect(self):
        """Turns on the robot and moves to home"""
        print("Connecting to robot...")
        try:
            self.robot = RobotPulse(self.host)
            gripper = tool_info(position([0, 0, 0.1], [0, 0, math.pi / 4]))
            self.robot.change_tool_info(gripper)
            print(f"Connected to robot at {self.host}")
            
            self.go_home()
            time.sleep(0.5)
            self.robot.set_digital_output_low(1)
            
            atexit.register(self.freeze)
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            sys.exit(1)

    def go_home(self):
        self.robot.set_position(HOME_POS, tcp_max_velocity=self.move_velocity, motion_type=MT_LINEAR)
        self.robot.await_stop()

    def freeze(self):
        if self.robot: self.robot.freeze()

    def get_target_positions(self, target_obj: DetectedObject, image):
        """Calculates base coordinates from the object's camera coordinates"""
        try:
            raw = self.robot.get_position()
            current_pos = [raw.point.x, raw.point.y, raw.point.z]
            roll, pitch, yaw = raw.rotation.roll, raw.rotation.pitch, raw.rotation.yaw
        except Exception as e:
            print(f"[ROBOT] ERROR: Could not get current TCP pose: {e}")
            return None

        # 1. Camera to TCP
        point_cam_h = np.array([target_obj.camera_point.x, target_obj.camera_point.y, target_obj.camera_point.z, 1.0])
        point_tcp_xyz = (H_cam_to_tcp @ point_cam_h)[:3]

        # 2. TCP to Base
        rot_for_matrix = [roll, pitch, -yaw]
        R_base_tcp = R_scipy.from_euler('xyz', rot_for_matrix, degrees=False).as_matrix()
        point_base_offset = R_base_tcp @ point_tcp_xyz
        point_base_xyz = np.array(current_pos) + point_base_offset

        # 3. Target Z
        target_z = point_base_xyz[2] + TARGET_OFFSET_Z
        target_pos_base = [point_base_xyz[0], point_base_xyz[1], target_z]
        target_pos_base_5 = [point_base_xyz[0], point_base_xyz[1], target_z + 0.05]
        target_pos_base_1 = [point_base_xyz[0], point_base_xyz[1], target_z - 0.02]

        # 4. Angle calculation
        if target_obj.angle > 0:
            target_yaw_rad = target_obj.angle + math.pi / 2
        else:
            target_yaw_rad = target_obj.angle - math.pi / 2
            
        output_rot = [roll, pitch, target_yaw_rad]

        # 5. UI Verification (No more pixel_center list bug!)
        if image is not None:
            disp = image.copy()
            cx, cy = int(target_obj.image_point.x), int(target_obj.image_point.y)
            cv2.circle(disp, (cx, cy), 5, (0, 255, 0), -1)
            
            line_length = 50
            end_x = int(cx + line_length * math.cos(target_yaw_rad))
            end_y = int(cy - line_length * math.sin(target_yaw_rad))
            cv2.line(disp, (cx, cy), (end_x, end_y), (0, 255, 0), 2)

            cv2.putText(disp, "Press 'C' to continue, 'Q' to quit", (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Target Verification", disp)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key not in (ord('c'), ord('C')):
                print("[ROBOT] User quit.")
                return None

        return [
            position(target_pos_base, output_rot), 
            position(target_pos_base_5, output_rot), 
            position(target_pos_base_1, output_rot)
        ]

    def execute_move(self, targets):
        """Your sequential movement logic"""
        if not targets: return
        
        steps = [
            ("Move to approach 1", targets[1]),
            ("Move to target", targets[0]),
            ("Move to grip", targets[2])
        ]
        
        for name, pos in steps:
            if int(input(f'0 - leave, 1 - {name}: ')) != 1:
                break
            self.robot.set_position(pos, tcp_max_velocity=self.move_velocity, motion_type=MT_LINEAR)
            self.robot.await_stop()
            time.sleep(0.5)
            
        # Grip and go home
        self.robot.set_digital_output_high(1) 
        self.robot.await_stop()
        time.sleep(0.5)
        self.go_home()
        self.robot.set_digital_output_low(1)