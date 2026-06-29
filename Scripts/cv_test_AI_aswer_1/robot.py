import math
import sys
import time
import atexit

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R_scipy
from pulseapi import (
    RobotPulse,
    position,
    MT_LINEAR,
    MT_JOINT,
    PulseApiException,
    tool_info,
)

# --- Настройки ---
ROBOT_HOST = "http://10.10.10.20:8081"  # ваш IP

# Высота рабочего стола над базой робота (м).
# Глубина с depth-камеры годится для X/Y, но для абсолютного Z она шумная,
# поэтому по умолчанию даём опцию зафиксировать Z на плоскость стола.
TABLE_Z = 0.03
TARGET_OFFSET_Z = 0.01  # смещение по Z над обнаруженной точкой

# --- Hand-eye калибровка (поза камеры во фрейме фланца/TCP) ---
# Сдвиг камеры относительно фланца. Поворот единичный — ровно как в старом
# (рабочем) YOLO-пайплайне на этом же оборудовании.
H_tcp_to_cam_file = np.array([
    [1.0, 0.0, 0.0, -0.014],
    [0.0, 1.0, 0.0,  0.05853],
    [0.0, 0.0, 1.0,  0.068],
    [0.0, 0.0, 0.0,  1.0],
])
# Точку из камеры в TCP переводим обратной матрицей (как в старом коде).
H_cam_to_tcp = np.linalg.inv(H_tcp_to_cam_file)

# --- Параметры робота ---
HOME_POS   = position([-0.4, 0.0, 0.5], [math.pi, 0, -math.pi / 2])
HOME_POS_1 = position([-0.4, 0.1, 0.3], [math.pi, 0, -math.pi / 2])
MOVE_SPEED = 25       # град/с для MT_JOINT
MOVE_VELOCITY = 0.1   # м/с для MT_LINEAR

# --- Подключение к роботу ---
print("Connecting to robot...")
try:
    robot = RobotPulse(ROBOT_HOST)
    gripper = tool_info(position([0, 0, 0.1], [0, 0, math.pi / 4]))
    robot.change_tool_info(gripper)
    print(f"Connected to robot at {ROBOT_HOST}")
except Exception as e:
    print(f"Failed to connect to robot: {e}")
    sys.exit(1)


@atexit.register
def robot_stop():
    robot.freeze()


in_free_drive = False

robot.set_position(HOME_POS, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
robot.await_stop()
time.sleep(0.5)
robot.set_digital_output_low(1)


def get_pos_from_cord(coord_camera, image, clamp_to_table=False):
    """
    Преобразует точку из оптического фрейма камеры [X_cam, Y_cam, Z_cam]
    в целевую позицию робота (base), показывает превью и ждёт C/Q.

    :param coord_camera: [X_cam, Y_cam, Z_cam] в метрах
    :param image: кадр для отображения
    :param clamp_to_table: True -> Z фиксируется на TABLE_Z (надёжнее),
                           False -> Z из глубины + TARGET_OFFSET_Z (как в старом коде)
    :return: объект position, либо None при отмене
    """
    # 1. Текущая поза TCP
    try:
        raw = robot.get_position()
        current_pos = [raw.point.x, raw.point.y, raw.point.z]
        roll = raw.rotation.roll
        pitch = raw.rotation.pitch
        yaw = raw.rotation.yaw
        print(f"[ROBOT] Current position: {current_pos}, [{roll:.4f}, {pitch:.4f}, {yaw:.4f}]")
    except Exception as e:
        print(f"[ROBOT] ERROR: Could not get current TCP pose: {e}")
        return None

    # 2. cam -> TCP (hand-eye)
    point_cam_h = np.array([coord_camera[0], coord_camera[1], coord_camera[2], 1.0])
    point_tcp_xyz = (H_cam_to_tcp @ point_cam_h)[:3]

    # 3. TCP -> base.
    # КЛЮЧЕВОЙ МОМЕНТ: yaw инвертируется ТОЛЬКО для построения матрицы поворота.
    # Конвенция yaw у Pulse противоположна scipy 'xyz' (extrinsic RPY).
    # Именно эту инверсию потерял VLM-код относительно рабочего YOLO-пайплайна.
    rot_for_matrix = [roll, pitch, -yaw]
    R_base_tcp = R_scipy.from_euler('xyz', rot_for_matrix, degrees=False).as_matrix()

    point_base_offset = R_base_tcp @ point_tcp_xyz
    point_base_xyz = np.array(current_pos) + point_base_offset
    print(f"[ROBOT] Offset in base: ({point_base_offset[0]:.3f}, "
          f"{point_base_offset[1]:.3f}, {point_base_offset[2]:.3f})")
    print(f"[ROBOT] Cube in base:   ({point_base_xyz[0]:.3f}, "
          f"{point_base_xyz[1]:.3f}, {point_base_xyz[2]:.3f})")

    # 4. Целевая позиция
    if clamp_to_table:
        target_z = TABLE_Z
    else:
        target_z = point_base_xyz[2] + TARGET_OFFSET_Z
    target_pos_base = [point_base_xyz[0], point_base_xyz[1], target_z]

    # Ориентацию роботу командуем РЕАЛЬНУЮ текущую (yaw НЕ инвертируем).
    output_rot = [roll, pitch, yaw]

    # 5. Визуализация и подтверждение
    if image is not None:
        disp = image.copy()
        lines = [
            f"Target Base Coord: ({target_pos_base[0]:.3f}, "
            f"{target_pos_base[1]:.3f}, {target_pos_base[2]:.3f})",
            f"Target Rotation (RPY): ({output_rot[0]:.2f}, "
            f"{output_rot[1]:.2f}, {output_rot[2]:.2f})",
        ]
        y_off = disp.shape[0] - 60
        for text in reversed(lines):
            cv2.putText(disp, text, (10, y_off),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            y_off -= 20
        cv2.putText(disp, "Press 'C' to continue, 'Q' to quit",
                    (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        print("[ROBOT] Displaying target coordinates. Waiting for input (C/Q)...")
        cv2.imshow("Target Verification", disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
            print("[ROBOT] User chose to quit. Aborting movement.")
            return None
        if key not in (ord('c'), ord('C')):
            print("[ROBOT] Unexpected key press. Aborting movement.")
            return None
        print("[ROBOT] User confirmed (C). Proceeding.")

    print(f"[ROBOT] Final target position of the cube: {target_pos_base}, "
          f"rotation: {output_rot}")
    return position(target_pos_base, output_rot)


def move_robot_to_pos(target):
    """Перемещение робота к целевой позиции."""
    print(f"[ROBOT] robot next position: {target}")
    x = int(input('0 - leave, 1 - continue: '))
    if x == 1:
        robot.set_position(target, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
        robot.await_stop()
        time.sleep(0.5)
