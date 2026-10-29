import atexit
import math
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from pulseapi import RobotPulse, position, MT_LINEAR, MT_JOINT, tool_info
from utils import DetectedObject, Point3D


# ---------------------------------------------------------------------------
# Константы конфигурации робота
# ---------------------------------------------------------------------------
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
    """Управление роботом Pulse: подключение, расчёт целевых позиций,
    последовательное движение к захвату/освобождению.
    """

    # Действия, известные execute_move / move_to_object.
    ACTION_PICK = "pick"
    ACTION_PUT = "put"

    # Высота (z в базовой системе координат, метры) для прохождения
    # ПРОМЕЖУТОЧНЫХ waypoint-ов траектории. Подъём (lift) и опускание
    # (placement) используют ТУ ЖЕ логику подхода, что и pick/put:
    # +0.05 -> +0.02 -> -0.013.
    MIDDLE_WAYPOINT_Z = 0.4

    def __init__(self, host="http://10.10.10.20:8081"):
        self.host = host
        self.robot = None
        self.move_velocity = 0.1

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def connect(self):
        """Подключается к роботу, активирует gripper, уводит в HOME."""
        print("Connecting to robot...")
        try:
            self.robot = RobotPulse(self.host)
            gripper = tool_info(position([0, 0, 0.1], [0, 0, math.pi / 4]))
            # self.robot.change_tool_info(gripper)
            print(f"Connected to robot at {self.host}")

            self.go_home()
            time.sleep(0.5)
            self.robot.set_digital_output_low(1)

            atexit.register(self.freeze)
        except Exception as e:
            print(f"Failed to connect to robot: {e}")
            sys.exit(1)

    def go_home(self):
        self.robot.set_position(HOME_POS, tcp_max_velocity=self.move_velocity,
                                motion_type=MT_LINEAR)
        self.robot.await_stop()

    def freeze(self):
        if self.robot:
            self.robot.freeze()

    def toggle_free_drive(self):
        self.robot.zg_on()
        while True:
            continue

    # ------------------------------------------------------------------
    # Расчёт целевых позиций
    # ------------------------------------------------------------------
    def get_target_positions(self, target_obj: DetectedObject, image):
        """Переводит camera_point объекта в систему базы робота и возвращает
        список из трёх позиций: grip, approach-5cm, approach-2cm.

        Возвращает None, если пользователь отказался (клавиша 'Q' в окне
        верификации), либо при ошибке чтения позиции TCP.
        """
        try:
            raw = self.robot.get_position()
            current_pos = [raw.point.x, raw.point.y, raw.point.z]
            roll, pitch, yaw = raw.rotation.roll, raw.rotation.pitch, raw.rotation.yaw
        except Exception as e:
            print(f"[ROBOT] ERROR: Could not get current TCP pose: {e}")
            return None

        # 1. Camera -> TCP
        point_cam_h = np.array([
            target_obj.camera_point.x,
            target_obj.camera_point.y,
            target_obj.camera_point.z,
            1.0,
        ])
        point_tcp_xyz = (H_cam_to_tcp @ point_cam_h)[:3]

        # 2. TCP -> Base
        rot_for_matrix = [roll, pitch, -yaw]
        R_base_tcp = R_scipy.from_euler('xyz', rot_for_matrix, degrees=False).as_matrix()
        point_base_offset = R_base_tcp @ point_tcp_xyz
        point_base_xyz = np.array(current_pos) + point_base_offset

        # 3. Целевое Z (подход, недоход, перехват).
        target_z = point_base_xyz[2] + TARGET_OFFSET_Z
        target_pos_base = [point_base_xyz[0], point_base_xyz[1], target_z - 0.013]
        target_pos_base_5 = [point_base_xyz[0], point_base_xyz[1], target_z + 0.05]
        target_pos_base_1 = [point_base_xyz[0], point_base_xyz[1], target_z + 0.02]

        # 4. Угол объекта -> yaw робота.
        target_yaw_rad = math.radians(target_obj.angle)
        output_rot = [roll, pitch, target_yaw_rad]

        # 5. UI-верификация (только если передали кадр).
        if image is not None and not self._verify_target(image, target_obj, target_yaw_rad):
            print("[ROBOT] User quit.")
            return None

        return [
            position(target_pos_base, output_rot),
            position(target_pos_base_5, output_rot),
            position(target_pos_base_1, output_rot),
        ]

    def _verify_target(self, image, target_obj, target_yaw_rad) -> bool:
        """Рисует целевую точку + ось угла на кадре и спрашивает подтверждение.
        True = пользователь нажал 'C' (continue), False = 'Q' (quit).
        """
        disp = image.copy()
        cx, cy = int(target_obj.image_point.x), int(target_obj.image_point.y)
        cv2.circle(disp, (cx, cy), 5, (0, 255, 0), -1)

        line_length = 50
        end_x = int(cx + line_length * math.cos(target_yaw_rad))
        end_y = int(cy - line_length * math.sin(target_yaw_rad))
        cv2.line(disp, (cx, cy), (end_x, end_y), (0, 255, 0), 2)

        cv2.putText(disp, "Press 'C' to continue, 'Q' to quit",
                    (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Target Verification", disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        return key in (ord('c'), ord('C'))

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------
    def _camera_point_to_base(self, cam_point: Point3D, yaw_rad: float = 0.0):
        """Конвертирует одну 3D-точку камеры в базовую систему робота.

        Возвращает кортеж (target_pos_base, output_rot, target_z):
          - target_pos_base = [x, y, z_grip]  — позиция на глубине захвата
            (z = camera_z + TARGET_OFFSET_Z - 0.013), как в get_target_positions
          - output_rot = [roll, pitch, yaw_rad]
          - target_z   = camera_z + TARGET_OFFSET_Z (без -0.013 — для расчёта
            подходов +0.05 / +0.02)

        Возвращает None, если не удалось прочитать текущую позу TCP.
        """
        try:
            raw = self.robot.get_position()
            current_pos = [raw.point.x, raw.point.y, raw.point.z]
            roll, pitch, yaw = raw.rotation.roll, raw.rotation.pitch, raw.rotation.yaw
        except Exception as e:
            print(f"[ROBOT] ERROR: Could not get current TCP pose: {e}")
            return None

        # Camera -> TCP
        point_cam_h = np.array([cam_point.x, cam_point.y, cam_point.z, 1.0])
        point_tcp_xyz = (H_cam_to_tcp @ point_cam_h)[:3]

        # TCP -> Base
        rot_for_matrix = [roll, pitch, -yaw]
        R_base_tcp = R_scipy.from_euler('xyz', rot_for_matrix, degrees=False).as_matrix()
        point_base_offset = R_base_tcp @ point_tcp_xyz
        point_base_xyz = np.array(current_pos) + point_base_offset

        target_z = point_base_xyz[2] + TARGET_OFFSET_Z
        target_pos_base = [point_base_xyz[0], point_base_xyz[1], target_z - 0.013]
        output_rot = [roll, pitch, yaw_rad]

        # Debug: показываем всю цепочку camera → tcp → base.
        print(f"[ROBOT] cam3D=({cam_point.x:.3f}, {cam_point.y:.3f}, {cam_point.z:.3f}) "
              f"-> tcp=({point_tcp_xyz[0]:.3f}, {point_tcp_xyz[1]:.3f}, {point_tcp_xyz[2]:.3f}) "
              f"-> base_xyz=({point_base_xyz[0]:.3f}, {point_base_xyz[1]:.3f}, {point_base_xyz[2]:.3f}) "
              f"| target_z={target_z:.3f} grip_z={target_pos_base[2]:.3f} "
              f"yaw={yaw_rad:.4f}")

        return (target_pos_base, output_rot, target_z)

    def _move_to_z_offset(self, base_info, z_offset):
        """Двигаться в (xy, target_z + z_offset) с текущей ротацией.

        base_info: кортеж из _camera_point_to_base.
        z_offset: +0.05 (approach-5cm), +0.02 (approach-2cm), -0.013 (grip).

        Возвращает [x, y, z] целевой позиции в базовой системе.
        """
        target_pos_base, output_rot, target_z = base_info
        final_xyz = [target_pos_base[0], target_pos_base[1], target_z + z_offset]
        pos = position(final_xyz, output_rot)
        # Печатаем то, что УХОДИТ в robot.set_position — для отладки.
        print(f"[ROBOT] -> set_position: xyz=({final_xyz[0]:.3f}, {final_xyz[1]:.3f}, {final_xyz[2]:.3f}) "
              f"rot=(r={output_rot[0]:.3f}, p={output_rot[1]:.3f}, y={output_rot[2]:.3f})")
        self.robot.set_position(pos, tcp_max_velocity=self.move_velocity,
                                motion_type=MT_LINEAR)
        self.robot.await_stop()
        time.sleep(0.5)
        return final_xyz

    @staticmethod
    def _compute_target_xyz(base_info, z_offset):
        """Вычисляет целевой [x, y, z] в базовой системе БЕЗ движения.
        Нужно для отображения в окне подтверждения ДО того, как робот поедет.
        """
        target_pos_base, _output_rot, target_z = base_info
        return [target_pos_base[0], target_pos_base[1], target_z + z_offset]

    def _confirm_trajectory_start(self, image) -> bool:
        """Показывает кадр и ждёт 'C' (start) или 'Q' (abort)."""
        if image is None:
            return True
        disp = image.copy()
        cv2.putText(disp, "Trajectory ready. Press 'C' to start, 'Q' to abort",
                    (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Trajectory Start", disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        return key in (ord('c'), ord('C'))

    def _confirm_step(self, name: str, image=None,
                      target_pos=None, reference_pixels=None,
                      lift_angle_deg=None) -> bool:
        """Ждёт подтверждения оператора перед ОДНОЙ командой движения.

        Args:
            name:             название шага ("LIFT approach +5cm" и т.п.).
            image:            исходный кадр (для отрисовки reference_pixels).
            target_pos:        [x, y, z] — куда робот собирается ехать
                              в БАЗОВОЙ системе координат.
            reference_pixels:  список кортежей (Point2D, label, BGR-color).
                              Например:
                              [(lift_pixel, "LIFT", (0,0,255)),
                               (placement_pixel, "PLACE", (255,0,0))]
                              Эти точки рисуются на кадре, чтобы оператор
                              видел, ОТКУДА была получена target_pos.
            lift_angle_deg:    угол lift (для отрисовки оси на lift-точке).

        Возвращает True если 'C' (continue), False если 'Q' (abort).
        """
        # Если изображения нет — рисуем пустое окно с текстом.
        if image is None:
            disp = np.zeros((200, 700, 3), np.uint8)
        else:
            disp = image.copy()

        # 1. Reference pixels (рисуем ПЕРВЫМИ, чтобы текст был сверху).
        if reference_pixels:
            for pix, label, color in reference_pixels:
                if pix is None:
                    continue
                cv2.circle(disp, (int(pix.x), int(pix.y)), 8, color, -1)
                cv2.putText(disp, label,
                            (int(pix.x) + 12, int(pix.y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Для LIFT рисуем ось угла.
                if label == "LIFT" and lift_angle_deg is not None:
                    end_x = int(pix.x + 40 * math.cos(math.radians(lift_angle_deg)))
                    end_y = int(pix.y - 40 * math.sin(math.radians(lift_angle_deg)))
                    cv2.line(disp, (int(pix.x), int(pix.y)),
                             (end_x, end_y), color, 2)

        # 2. Текст сверху: название шага + целевая позиция.
        cv2.putText(disp, f"Step: {name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        if target_pos is not None:
            cv2.putText(disp,
                        f"Target base XYZ: ({target_pos[0]:.3f}, "
                        f"{target_pos[1]:.3f}, {target_pos[2]:.3f})",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # 3. Текст снизу: инструкция.
        cv2.putText(disp, "Press 'C' to continue, 'Q' to abort trajectory",
                    (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow(f"Confirm: {name}", disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key in (ord('c'), ord('C')):
            print(f"[ROBOT] Confirmed: {name}")
            return True
        print(f"[ROBOT] Aborted at step: {name}")
        return False

    def move_through_trajectory(self, trajectory_camera_points, image=None,
                                lift_angle_deg=0.0,
                                lift_pixel=None, placement_pixel=None,
                                middle_pixels=None):
        """Выполняет траекторию: lift -> middle waypoints (z=0.4) -> placement.

        Перед КАЖДЫМ движением ждёт подтверждения оператора через _confirm_step.
        В окне подтверждения рисуются lift (красный) и placement (синий) пиксели
        и пишется целевая позиция робота в БАЗЕ — чтобы оператор видел
        соответствие "пиксель на кадре ↔ куда едет робот".

        Контракт по waypoint-ам:
          - trajectory_camera_points[0]  : lift  — подход + grip  (текущая логика pick).
                                            Угол объекта (lift_angle_deg) применяется
                                            ТОЛЬКО к этой позиции как yaw робота.
          - trajectory_camera_points[1:-1]: middle — прямолинейное движение на z=0.4.
          - trajectory_camera_points[-1] : placement — подход + release (текущая логика put).
                                            Угол = 0 (плейсмент не требует ориентации).

        Args:
            trajectory_camera_points: list[Point3D] в координатах камеры, len >= 2.
            image: опционально — кадр для окон подтверждения.
            lift_angle_deg: угол объекта в градусах (например, из
                            cam.find_angle_around_point). Применяется к lift-позиции.
            lift_pixel:      Point2D пиксельных координат lift (для отрисовки).
            placement_pixel: Point2D пиксельных координат placement (для отрисовки).
            middle_pixels:   list[Point2D] — пиксели middle waypoints (для отрисовки).

        Returns:
            True — траектория выполнена, False — прервана/ошибка.
        """
        if not trajectory_camera_points or len(trajectory_camera_points) < 2:
            print("[ROBOT] Trajectory must have at least 2 points (lift + placement)")
            return False

        if not self._confirm_trajectory_start(image):
            print("[ROBOT] User aborted trajectory")
            return False

        lift_cam = trajectory_camera_points[0]
        placement_cam = trajectory_camera_points[-1]
        middle_cams = trajectory_camera_points[1:-1]

        # Угол lift-позиции переводим в радианы и используем как yaw.
        # Placement — yaw = 0.
        lift_yaw_rad = math.radians(lift_angle_deg) if lift_angle_deg else 0.0
        print(f"[ROBOT] Lift angle: {lift_angle_deg}° ({lift_yaw_rad:.4f} rad)")

        # === ТРАНСФОРМАЦИЯ: 3D камеры → база робота (camera_point_to_base) ===
        lift_base = self._camera_point_to_base(lift_cam, yaw_rad=lift_yaw_rad)
        placement_base = self._camera_point_to_base(placement_cam, yaw_rad=0.0)
        if lift_base is None or placement_base is None:
            print("[ROBOT] Could not compute base positions for lift/placement")
            return False

        # Reference pixels — отрисовываются в КАЖДОМ окне подтверждения,
        # чтобы оператор видел, ОТКУДА взялась target_pos.
        ref_pixels = [
            (lift_pixel, "LIFT", (0, 0, 255)),       # красный
            (placement_pixel, "PLACE", (255, 0, 0)),# синий
        ]

        # === LIFT: те же шаги, что и в execute_move для pick ===
        print("[ROBOT] --- LIFT ---")

        # Approach +5cm
        target = self._compute_target_xyz(lift_base, +0.05)
        if not self._confirm_step("LIFT approach +5cm", image,
                                  target_pos=target,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        self._move_to_z_offset(lift_base, +0.05)

        # Approach +2cm
        target = self._compute_target_xyz(lift_base, +0.02)
        if not self._confirm_step("LIFT approach +2cm", image,
                                  target_pos=target,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        self._move_to_z_offset(lift_base, +0.02)

        # Grip depth
        target = self._compute_target_xyz(lift_base, -0.013)
        if not self._confirm_step("LIFT descend to grip (-1.3cm)", image,
                                  target_pos=target,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        self._move_to_z_offset(lift_base, -0.013)

        # Close gripper
        if not self._confirm_step("Close gripper (pick)", image,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        time.sleep(0.5)
        self.robot.set_digital_output_high(1)      # close gripper
        self.robot.await_stop()
        time.sleep(0.5)

        # === MIDDLE WAYPOINTS: z = 0.4 ===
        for i, mid_cam in enumerate(middle_cams):
            # Трансформация middle: 3D камеры → база (yaw=0)
            mid_base = self._camera_point_to_base(mid_cam)
            if mid_base is None:
                print(f"[ROBOT] Skipping middle waypoint {i+1} (TCP pose read failed)")
                continue
            # Переписываем z на безопасную высоту 0.4 (xy берём из расчёта).
            target_pos_base, output_rot, _ = mid_base
            mid_xyz = [target_pos_base[0], target_pos_base[1], self.MIDDLE_WAYPOINT_Z]
            mid_pos = position(mid_xyz, output_rot)

            # Если есть пиксель для этого middle, добавим в reference_pixels.
            mid_ref = list(ref_pixels)
            if middle_pixels and i < len(middle_pixels):
                mid_ref.append((middle_pixels[i], f"M{i+1}", (0, 255, 0)))

            if not self._confirm_step(
                f"MIDDLE {i+1}/{len(middle_cams)} -> z={self.MIDDLE_WAYPOINT_Z}",
                image,
                target_pos=mid_xyz,
                reference_pixels=mid_ref,
                lift_angle_deg=lift_angle_deg,
            ):
                return False
            print(f"[ROBOT] --- MIDDLE {i+1}/{len(middle_cams)} -> z={self.MIDDLE_WAYPOINT_Z} ---")
            print(f"[ROBOT] -> set_position: xyz=({mid_xyz[0]:.3f}, {mid_xyz[1]:.3f}, {mid_xyz[2]:.3f}) "
                  f"rot=(r={output_rot[0]:.3f}, p={output_rot[1]:.3f}, y={output_rot[2]:.3f})")
            self.robot.set_position(mid_pos, tcp_max_velocity=self.move_velocity,
                                    motion_type=MT_LINEAR)
            self.robot.await_stop()
            time.sleep(0.5)

        # === PLACEMENT: те же шаги, что и в execute_move для put ===
        print("[ROBOT] --- PLACEMENT ---")

        # Approach +5cm
        target = self._compute_target_xyz(placement_base, +0.05)
        if not self._confirm_step("PLACEMENT approach +5cm", image,
                                  target_pos=target,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        self._move_to_z_offset(placement_base, +0.05)

        # Approach +2cm
        target = self._compute_target_xyz(placement_base, +0.02)
        if not self._confirm_step("PLACEMENT approach +2cm", image,
                                  target_pos=target,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        self._move_to_z_offset(placement_base, +0.02)

        # Descend to placement
        target = self._compute_target_xyz(placement_base, -0.013)
        if not self._confirm_step("PLACEMENT descend (-1.3cm)", image,
                                  target_pos=target,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        self._move_to_z_offset(placement_base, -0.013)

        # Open gripper
        if not self._confirm_step("Open gripper (release)", image,
                                  reference_pixels=ref_pixels,
                                  lift_angle_deg=lift_angle_deg):
            return False
        time.sleep(0.5)
        self.robot.set_digital_output_low(1)            # open gripper
        self.robot.await_stop()
        time.sleep(0.5)

        # Return home
        if not self._confirm_step("Return to HOME", image,
                                  reference_pixels=ref_pixels):
            return False
        self.go_home()
        return True

    # ------------------------------------------------------------------
    # Движение (одиночная цель — pick/put)
    # ------------------------------------------------------------------
    def execute_move(self, targets, type='pick'):
        """Последовательное движение по позициям targets (grip, +5cm, +2cm)
        с пошаговым подтверждением пользователя, затем включение/выключение
        gripper и возврат в HOME.
        """
        if not targets:
            return

        steps = [
            ("Move to approach 1", targets[1]),
            ("Move to target",     targets[2]),
            ("Move to grip",       targets[0]),
        ]

        for name, pos in steps:
            if int(input(f'0 - leave, 1 - {name}: ')) != 1:
                break
            self.robot.set_position(pos, tcp_max_velocity=self.move_velocity,
                                    motion_type=MT_LINEAR)
            self.robot.await_stop()
            time.sleep(0.5)

        # Grip and go home
        time.sleep(4)
        if type == self.ACTION_PICK:
            self.robot.set_digital_output_high(1)
        elif type == self.ACTION_PUT:
            self.robot.set_digital_output_low(1)
        self.robot.await_stop()
        time.sleep(0.5)
        self.go_home()

    def move_to_object(self, target, image, action="pick"):
        """Высокоуровневая обёртка: считает позиции и сразу выполнит движение.

        action: 'pick' или 'put'.
        Возвращает True, если движение выполнено, False — если цели нет
        или пользователь прервал верификацию.
        """
        targets = self.get_target_positions(target, image)
        if not targets:
            print("[ROBOT] No targets calculated, movement aborted")
            return False
        print(f"[ROBOT] Targets: {targets}")
        self.execute_move(targets, type=action)
        return True
