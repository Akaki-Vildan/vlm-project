import numpy as np
import math
import time
import cv2
import pyrealsense2 as rs
# from Progs.pulse_key import HOME_POSITION
# from requests.packages import target
from ultralytics import YOLO
from pulseapi import (
    RobotPulse,
    position,
    MT_LINEAR,
    MT_JOINT,
    PulseApiException,
    tool_info  # Импортируем tool_info
)
import keyboard
import sys

# --- Настройки ---
ROBOT_HOST = "http://10.10.10.20:8081"  # Убедитесь, что это ваш IP

# Параметры для камеры и YOLO
YOLO_MODEL_PATH = 'train_model/models/cube_detection_v3.pt'  # Путь к модели, совместимой с новым кодом
YOLO_CONFIDENCE = 0.4  # Уменьшено, как в новом коде
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
# Z координата объекта на столе (в метрах) - будет использоваться как приближение для глубины
# или Z_cam при вычислении 3D координат, если глубина недоступна.
OBJECT_HEIGHT_ON_TABLE = 0.05  # Пример высоты объекта над уровнем стола

# --- Калибровочные данные (ПРЕДПОЛАГАЕТСЯ: H_tcp_to_cam_file - это матрица из TCP в камеру) ---
# Пожалуйста, подставьте свои значения!
H_tcp_to_cam_file = np.array([[1, 0, 0, -0.014888],  # Пример: смещение X
                              [0, 1, 0, 0.052431],  # Пример: смещение Y
                              [0, 0, 1, 0.070306],  # Пример: смещение Z
                              [0, 0, 0, 1]])  # Пример: ориентация (обычно единичная матрица 3x3 в R, t внизу)
# print('H_tcp_to_cam_file (from TCP to camera):', H_tcp_to_cam_file)

# Для преобразования из камеры в TCP, нужно взять обратную матрицу от H_tcp_to_cam_file
H_cam_to_tcp = np.linalg.inv(H_tcp_to_cam_file)  # Это матрица из камеры в TCP
# print('Calculated H_cam_to_tcp (from camera to TCP):', H_cam_to_tcp)

# --- Параметры для робота ---
# Домашняя позиция
HOME_POSE = position([-0.4, 0, 0.5], [math.pi, 0, -math.pi / 2])  # Пример из предыдущего кода

MOVE_SPEED = 25  # Скорость движения (в градусах/сек для MT_JOINT)
MOVE_VELOCITY = 0.1  # Максимальная скорость TCP (м/с) для MT_LINEAR
TARGET_OFFSET_Z = 0.005  # Смещение по Z над обнаруженной точкой (2 см)

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

in_free_drive = False

robot.set_position(HOME_POSE, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
robot.await_stop()
time.sleep(0.5)
robot.set_digital_output_low(1)

class cube:
    #инициализация класса
    def __init__(self, x, y, z, angl, id):
        self.x = x
        self.y = y
        self.z = z
        self.angle_deg = angl
        self.class_id = id
        self.queue_pos = 0
        self.target_rot_base = None
        self.target_pose_air = None
        self.target_pose_air_5 = None
        self.target_pose_base = None
        self.end_position_air = None
        self.end_position_base = None

    #получаем готовые позиции из координат(закинул код из S2R сюда)
    def get_pos_from_cord(self):
        raw_pose = robot.get_position()
        print(f"[S2R] Raw pose object type: {type(raw_pose)}")

        # Извлекаем позицию и ориентацию используя прямой доступ к атрибутам
        current_pos = [raw_pose.point.x, raw_pose.point.y, raw_pose.point.z]
        current_rot = [raw_pose.rotation.roll, raw_pose.rotation.pitch, -raw_pose.rotation.yaw]
        # 2. Преобразование координат из системы камеры в систему base
        #    2.1. Точка в системе камеры (гомогенная)
        point_cam_h = np.array([self.x, self.y, self.z, 1.0])

        #    2.2. Точка в системе TCP: используем H_cam_to_tcp
        point_tcp_h = H_cam_to_tcp @ point_cam_h
        #print(f"[S2R] Point in TCP frame: ({point_tcp_h[0]:.3f}, {point_tcp_h[1]:.3f}, {point_tcp_h[2]:.3f})")

        #    2.3. Точка в системе base: используем текущую позицию TCP как базовую
        #         Найдем R_base_to_tcp из current_rot (RPY)
        from scipy.spatial.transform import Rotation as R_scipy
        R_obj = R_scipy.from_euler('xyz', current_rot, degrees=False)  # 'xyz' означает roll, pitch, yaw
        R_base_to_tcp = R_obj.as_matrix()

        #         Смещение точки объекта в системе TCP
        point_tcp_xyz = point_tcp_h[:3]  # [X_tcp, Y_tcp, Z_tcp]

        #         Смещение точки объекта в системе Base
        point_base_offset = R_base_to_tcp @ point_tcp_xyz

        #         Финальная позиция объекта в системе Base
        point_base_xyz = np.array(current_pos) + point_base_offset

        '''
        print(
            f"[S2R] Point offset in Base frame: ({point_base_offset[0]:.3f}, {point_base_offset[1]:.3f}, {point_base_offset[2]:.3f})")
        print(f"[S2R] Point in Base frame: ({point_base_xyz[0]:.3f}, {point_base_xyz[1]:.3f}, {point_base_xyz[2]:.3f})")
        '''

        # 3. Определение целевой позиции в системе base
        #    Цель: находиться над объектом на TARGET_OFFSET_Z
        target_pos_air = [point_base_xyz[0], point_base_xyz[1], 0.3]
        target_pos_air_5 = [point_base_xyz[0], point_base_xyz[1], 0.05]
        target_pos_base = [point_base_xyz[0], point_base_xyz[1], point_base_xyz[2] + TARGET_OFFSET_Z]

        if self.angle_deg > 0:
            self.angle_deg -= 90
        else:
            self.angle_deg += 90
        target_yaw_rad_raw = math.radians(self.angle_deg)  # Угол поворота вокруг Z (yaw) из камеры (angle_real), в радианах
        target_yaw_deg_raw = math.degrees(target_yaw_rad_raw)  # Для отображения (до нормализации)
        #print(f"[S2R] Raw Yaw from camera (rad): {target_yaw_rad_raw:.4f}, (deg): {target_yaw_deg_raw:.2f}")  # Отладка

        #    Нормализация yaw к диапазону [-π, π]
        target_yaw_rad = (target_yaw_rad_raw + math.pi) % (2 * math.pi) - math.pi
        target_yaw_deg = math.degrees(target_yaw_rad)  # Для отображения (после нормализации)
        #print(f"[S2R] Normalized Target Yaw (rad): {target_yaw_rad:.4f}, (deg): {target_yaw_deg:.2f}")  # Отладка

        #    Ориентация: копируем ориентацию TCP из текущего положения, но меняем yaw
        #    Предположим, что ориентация захвата должна соответствовать angle_real.
        #    roll и pitch остаются как у TCP (или как у захвата, если он смещен).
        #    Текущие ориентации: current_rot = [roll, pitch, current_yaw_tcp]
        target_rot_base = [current_rot[0], current_rot[1],
                           target_yaw_rad]  # roll, pitch, новый yaw (angle_real, нормализованный)

        self.target_rot_base = target_rot_base

        self.target_pose_air = position(target_pos_air, target_rot_base)
        self.target_pose_air_5 = position(target_pos_air_5, target_rot_base)
        self.target_pose_base = position(target_pos_base, target_rot_base)

        return [target_yaw_deg, target_pos_base]


    #позиции для классификации
    def get_position_for_clasification(self):
        '''
        Для кубика, при вызове создает position, где rotation не меняется, а координаты
        меняются в зависимости от класса:
        выставляет в ряд с одинаковой y координатой и различной x, которая меняется по след закаону:
        на 25 мм дальше предыдущего
        '''
        rot = [math.pi, 0, -math.pi / 2]
        if(self.class_id == 0):#если кубик черный
            pos_air = [-0.4 - (0.025 * self.queue_pos), 0.4, 0.25]
            pos_base = [-0.4- (0.025 * self.queue_pos), 0.4, 0.005]
        elif(self.class_id == 1):#если кубик белый
            pos_air = [-0.4 - (0.025 * self.queue_pos), -0.4, 0.25]
            pos_base = [-0.4- (0.025 * self.queue_pos), -0.4, 0.005]
        else:
            pos_air = [-0.4, 0.4, 0.25]
            pos_base = [-0.4, 0.4, 0.005]
            print("[CC] where are no this class id")
        self.end_position_air = position(pos_air , rot)
        self.end_position_air = position(pos_base, rot)


# --- Функции ---

def detect_and_display_cubes():
    """
    Захватывает один кадр с камеры Realsense, обнаруживает кубики (классы 0 и 1) с помощью YOLO,
    определяет их 3D координаты (X, Y, Z) и угол поворота (angle) в системе камеры,
    отображает кадр с детекцией и информацией, возвращает
    список словарей с информацией о кубиках и изображение.
    """
    print("[CAMERA] Detecting and displaying cubes (class 0=black, class 1=white)...")
    cubes_list = []  # Список кубиков
    image_to_show = None

    try:
        # Инициализация YOLO
        model = YOLO(YOLO_MODEL_PATH)

        # Инициализация камеры Realsense
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
        config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
        profile = pipeline.start(config)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        # Небольшая прогревка камеры
        for i in range(30):  # Прогреваем 10 кадров
            pipeline.wait_for_frames()

        # Захват кадра
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not depth_frame or not color_frame:
            print("[CAMERA] Error: Could not acquire depth or color frames")
            return cubes_list, image_to_show

        # Преобразование в numpy
        default_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # Применяем коррекцию изображения (если нужно, как в старом коде)
        color_image = cv2.convertScaleAbs(default_image, alpha=1.2, beta=20)
        image_to_show = color_image.copy()  # Для отображения

        # Конвертация глубины в метры
        depth_image_meters = depth_image * depth_scale

        # --- Обнаружение отдельных кубиков ---
        # Обнаруживаем ОБА класса (0 и 1) за один проход
        results = model(color_image, classes=[0, 1], conf=YOLO_CONFIDENCE, verbose=False)
        obbs_all = results[0].obb

        if obbs_all is not None and len(obbs_all) > 0:
            # Определяем ROI (область интереса)
            wd = CAMERA_WIDTH
            hi = CAMERA_HEIGHT
            x1_roi = int(wd * 0.2)
            y1_roi = int(hi * 0.2)
            x2_roi = int(wd * 0.8)
            y2_roi = int(hi * 0.8)
            cv2.rectangle(image_to_show, (x1_roi, y1_roi), (x2_roi, y2_roi), (30, 30, 200), 2)

            # Получаем ID классов и углы для всех OBB
            obb_classes = obbs_all.cls.cpu().numpy()
            obb_xywhrs = obbs_all.xywhr.cpu().numpy()

            for i, (xywhr, cls_id) in enumerate(zip(obb_xywhrs, obb_classes)):

                x, y, w, h, angle_rad = xywhr
                class_id = int(cls_id)
                angle_deg = np.degrees(angle_rad)

                # Преобразуем координаты в int для работы с изображением
                center_x_int = int(x)
                center_y_int = int(y)

                # Проверка нахождения в ROI
                if not (x1_roi < center_x_int < x2_roi and y1_roi < center_y_int < y2_roi):
                    continue  # Переходим к следующему объекту, если он не в ROI

                # --- Вычисление 3D координат (X, Y, Z) ---
                # 1. Получаем значение глубины в центре OBB
                y_int = int(min(max(center_y_int, 0), depth_image_meters.shape[0] - 1))
                x_int = int(min(max(center_x_int, 0), depth_image_meters.shape[1] - 1))

                # Проверяем, является ли значение достоверным
                depth_value = depth_image_meters[y_int, x_int]
                if depth_value == 0:
                    print(f"[CAMERA] Warning: Depth at cube center ({x_int}, {y_int}) is 0. Skipping this detection.")
                    continue  # Переходим к следующему обнаружению

                # 2. Вычисляем 3D координаты в системе камеры
                Z_cam = depth_value  # Глубина в метрах
                X_cam = Z_cam * (x - intr.ppx) / intr.fx
                Y_cam = Z_cam * (y - intr.ppy) / intr.fy

                # --- Определение имени класса ---
                # 0 - black 1 - white

                class_name = "cube-black" if class_id == 0 else "cube-white"

                # --- Добавление информации о кубике в список ---
                cubes_list.append(cube(
                    X_cam,
                    Y_cam,
                    Z_cam,
                    angle_deg,
                    class_id))

                # --- Отображение на изображении ---
                color = (0, 255, 0) if class_id == 1 else (0, 0, 255)  # Белый = зеленый прямоугольник, Черный = красный
                # Рисуем OBB
                rect_points = ((float(x), float(y)), (float(w), float(h)), float(angle_deg))
                box_points = cv2.boxPoints(rect_points)
                box_points = np.intp(box_points)
                cv2.polylines(image_to_show, [box_points], True, color, 2)

                # Отмечаем центр
                cv2.circle(image_to_show, (int(x), int(y)), 3, (0, 255, 255), -1)  # Желтая точка в центре

                # Выводим информацию
                cv2.putText(image_to_show, f"{class_name} ({class_id}) {i}", (int(x) - 20, int(y) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(image_to_show, f"Z: {Z_cam * 1000:.0f}mm", (int(x) - 20, int(y) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(image_to_show, f"A: {angle_deg:.1f}", (int(x) - 20, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 0, 0), 1)

                print(
                    f"[CAMERA] Detected {class_name} cube [{class_id}] at Cam coords (X,Y,Z): ({X_cam:.3f}, {Y_cam:.3f}, {Z_cam:.3f}) m, Angle: {angle_deg:.2f} deg")

        else:
            print("[CAMERA] No cubes detected by YOLO.")
            cv2.putText(image_to_show, "No cubes found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        print(f"[CAMERA] Found {len(cubes_list)} cubes in total.")
        return cubes_list, image_to_show

    except Exception as e:
        print(f"[CAMERA] Error during cube detection/display: {e}")
        import traceback
        traceback.print_exc()
        if 'image_to_show' in locals():
            cv2.putText(image_to_show, f"Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return cubes_list, image_to_show
    finally:
        try:
            pipeline.stop()
        except:
            pass  # Игнорируем ошибки остановки


def sim2real_step():
    """Выполняет один шаг Sim2Real с данными с камеры и отображением."""
    global in_free_drive
    if in_free_drive:
        print("[S2R] Cannot perform Sim2Real step while in Free Drive mode. Please press F to exit.")
        return

    in_loop = False

    print(f"[S2R] --- Starting Sim2Real with Camera Detection (Cam on TCP, API docs, angle_only logic) ---")

    # 0. Получить текущую позицию TCP в системе base
    print("[S2R] Getting current TCP position in base frame...")
    try:
        raw_pose = robot.get_position()
        print(f"[S2R] Raw pose object type: {type(raw_pose)}")

        # Извлекаем позицию и ориентацию используя прямой доступ к атрибутам
        current_pos = [raw_pose.point.x, raw_pose.point.y, raw_pose.point.z]
        current_rot = [raw_pose.rotation.roll, raw_pose.rotation.pitch, -raw_pose.rotation.yaw]

        print(f"[S2R] Current TCP Position (Base): {current_pos}")
        print(f"[S2R] Current TCP Rotation (Base) [RPY]: {current_rot}")

    except Exception as e:
        print(f"[S2R] ERROR: Could not get current TCP pose from robot: {e}")
        return

    queue = np.zeros(2)  # [0] - если черный, [1] - если белый

    # 1. Обнаружение объекта с камеры и отображение (используем обновлённую логику)
    cubes_list, image = detect_and_display_cubes()

    for cur_cube in cubes_list:

        if(cur_cube.class_id == 0):
            cur_cube.class_id = queue[0]
            queue[0] += 1
        else:
            cur_cube.class_id = queue[1]
            queue[1] += 1

        if cur_cube.x is None:
            print("[S2R] FAILED: Could not detect object pair [1] & [0] from camera within ROI. Exiting.")
            if image is not None:
                cv2.imshow("Camera View", image)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                cv2.destroyAllWindows()
            return

        target = cur_cube.get_pos_from_cord()

        target_yaw_deg = target[0]
        target_pos_base = target[1]

        # 4. Наложение координат для перемещения на изображение
        if image is not None and in_loop == False:
            in_loop = True
            # Собираем строку с координатами для перемещения
            # Используем нормализованный угол для отображения
            move_info_text = [
                f"Target Base Coord: ({target_pos_base[0]:.3f}, {target_pos_base[1]:.3f}, {target_pos_base[2]:.3f})",
                f"Target Yaw (from cam, norm): {target_yaw_deg:.2f} deg"
            ]
            y_offset = image.shape[0] - 60  # Начинаем снизу
            for text in reversed(move_info_text):
                cv2.putText(image, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                y_offset -= 20  # Поднимаемся вверх

            cv2.putText(image, "Press 'c' to continue, 'q' to quit", (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            print(f"[S2R] Displaying target coordinates on image.")
            print(move_info_text)
            cv2.imshow("Camera View", image)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('q'):
                print("[S2R] User chose to quit. Exiting without moving.")
                return
            elif key != ord('c'):
                print("[S2R] Unexpected key press. Exiting without moving.")
                return
            # Если нажата 'c', продолжаем

        # 5. Установка инструмента (гриппера) - опционально, но рекомендуется если используется
        print("[S2R] Setting tool info...")
        gripper_tool = tool_info(position([0, 0, 0.1], [0, 0, math.pi / 4]))  # Указанная вами строка
        robot.change_tool_info(gripper_tool)

        # 6. Получение текущей позиции TCP для отладки
        current_pos_after = robot.get_position()
        print(f"[S2R] Current TCP after detection: {current_pos_after}")

        # 7. Движение к целевой позиции (в системе Base)
        print(f"[S2R] Moving to target pose in Base frame...")
        robot.set_position(cur_cube.target_pose_air, tcp_max_velocity=MOVE_VELOCITY,
                           motion_type=MT_LINEAR)  # позиция над кубиком
        robot.await_stop()
        time.sleep(0.5)
        # добавить новую детекцию

        cubes_list_new, image_new = detect_and_display_cubes()#получаем новый список кубиков далее работаем с ним, чтоб не менять исходный

        if image is not None:
            cv2.imshow("Camera View", image_new)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()


        if key == ord('q'):
            print("[S2R] User chose to quit. Exiting without moving.")
            return
        elif key != ord('c'):
            print("[S2R] Unexpected key press. Exiting without moving.")
            return

        id = input("[S2R] Choose a cube")#получаем id кубика, который мы выберем, а точнее который стоит под роботом

        cur_cube_new = cubes_list_new[id]#работаем теперь с этим кубиком(тот же самый что и исходный, мы же не идиоты, чтоб другой выбирать, правильно?=) )

        try:
            target = cur_cube_new.get_pos_from_cord()#высчитываем новую позицию
        except Exception as e:
            print(f"[S2R] There are no cube with this id error:{e}")
            return

        target_yaw_deg = target[0][0]
        target_pos_base = target[0][1]

        robot.set_position(cur_cube_new.target_pose_air, tcp_max_velocity=MOVE_VELOCITY,
                           motion_type=MT_LINEAR)  # позиция над кубиком повторяем, после проверки

        robot.set_position(cur_cube_new.target_pose_air_5, tcp_max_velocity=MOVE_VELOCITY,
                           motion_type=MT_LINEAR)  # позиция над кубиком 5 см
        robot.await_stop()
        time.sleep(0.5)

        # остановить робот за 5 см до кубика и ждать нажатия кнопки с
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        if key == ord('q'):
            print("[S2R] User chose to quit. Exiting without moving.")
            return
        elif key != ord('c'):
            print("[S2R] Unexpected key press. Exiting without moving.")
            return
        # Если нажата 'c', продолжаем

        robot.set_position(cur_cube_new.target_pose_base, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)  # позиция кубика
        time.sleep(0.5)

        print(f"[S2R] Command sent to move to target pose in Base frame.")
        robot.await_stop()
        robot.set_digital_output_high(1)  # захват
        robot.set_position(cur_cube_new.target_pose_air, tcp_max_velocity=MOVE_VELOCITY,
                           motion_type=MT_LINEAR)
        robot.set_position(HOME_POSE, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
        robot.await_stop()
        time.sleep(0.5)

        cur_cube_new.get_position_for_clasification()#получаем конечные позиции

        print(cur_cube.class_id)
        print(f"[S2R] it is black cube")
        robot.set_position(cur_cube_new.end_position_air, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
        robot.await_stop()
        time.sleep(0.5)
        robot.set_position(cur_cube_new.end_position_base, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
        robot.await_stop()
        time.sleep(0.5)
        robot.set_digital_output_low(1)
        robot.set_position(cur_cube_new.end_position_air, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
        robot.await_stop()
        time.sleep(0.5)
        robot.set_position(HOME_POSE, tcp_max_velocity=MOVE_VELOCITY, motion_type=MT_LINEAR)
        robot.await_stop()
        time.sleep(0.5)

        # Опционально: Ждём завершения движения
        # robot.await_stop() # <-- МОЖНО РАСКОММЕНТИРОВАТЬ, если нужно дождаться остановки

        print(f"[S2R] --- Sim2Real Movement Command Sent ---")


def go_home():
    """Отправляет робота в домашнюю позицию."""
    global in_free_drive
    if in_free_drive:
        print("[CMD] Exiting Free Drive mode before moving...")
        try:
            robot.zg_off()
            in_free_drive = False
            time.sleep(1)
        except PulseApiException as e:
            print(f"Error exiting free drive: {e}")
            return

    print("[CMD] Moving to Home Position...")
    try:
        robot.set_position(HOME_POSE, speed=MOVE_SPEED, motion_type=MT_JOINT)
        robot.await_stop()
        print("[CMD] Home Position reached.")
    except PulseApiException as e:
        print(f"Error moving home: {e}")


def toggle_free_drive():
    """Включает или выключает режим свободного перемещения."""
    global in_free_drive
    try:
        if not in_free_drive:
            print("[CMD] Entering Free Drive (Zero Gravity) mode...")
            robot.zg_on()
            in_free_drive = True
        else:
            print("[CMD] Exiting Free Drive mode...")
            robot.zg_off()
            in_free_drive = False
    except PulseApiException as e:
        print(f"Error toggling free drive: {e}")


def print_position():
    """Выводит текущую позицию и ориентацию манипулятора."""
    try:
        current_position = robot.get_position()
        print(f"Current Position: {current_position}")
        print(f"Current Pose: {robot.get_pose()}")
    except PulseApiException as e:
        print(f"Error getting position: {e}")


def main():
    """Основной цикл программы."""
    print("\n--- Robot Control via Camera Detection (Cam on TCP, API docs, angle_only logic) ---")
    print("Controls:")
    print("  S - Execute Sim2Real step (Detect object pair [1]&[0], display info, move directly)")
    print("  H - Move to Home position")
    print("  F - Toggle Free Drive mode")
    print("  P - Print current position")
    print("  ESC - Exit program")
    print("----------------------------------------------------------\n")

    try:
        print_position()  # Вывести начальную позицию

        while True:
            if keyboard.is_pressed('s'):
                sim2real_step()
                time.sleep(0.5)  # Увеличена задержка, чтобы дать время отпустить 'S'
            elif keyboard.is_pressed('h'):
                go_home()
                time.sleep(0.3)
            elif keyboard.is_pressed('f'):
                toggle_free_drive()
                time.sleep(0.5)
            elif keyboard.is_pressed('p'):
                print_position()
                time.sleep(0.3)
            elif keyboard.is_pressed('esc'):
                print("\n[EXIT] ESC key pressed. Exiting program.")
                break

            time.sleep(0.05)  # Маленькая задержка для снижения нагрузки на CPU

    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\n[EXIT] An unexpected error occurred in main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[EXIT] Stopping robot...")
        try:
            robot.freeze()
        except:
            pass
        print("[EXIT] Program finished.")


if __name__ == "__main__":
    main()

cube_clasification_move_v2.py
Отображается файл "cube_clasification_move_v2.py"