"""PickAndPlacePipeline — высокоуровневый оркестратор.

Скрывает от внешнего кода всю последовательность:
    capture frame -> VLM grounding -> contour + angle -> VLM pickup-point
    -> deproject 3D -> robot move.

main.py становится минимальным: создаёт pipeline и вызывает .run().
"""

import cv2

from camera import RealSenseCamera
from robot import RobotController
from utils import DetectedObject, Point2D
from vlm import VLMProcessor


class PickAndPlacePipeline:
    """Оркестрирует работу камеры, VLM и робота для pick-and-place задачи."""

    def __init__(self, camera_kwargs=None, robot_host="http://10.10.10.20:8081"):
        """
        Args:
            camera_kwargs: dict с width/height/fps для RealSenseCamera (None = defaults).
            robot_host:    URL PulseAPI-сервера.
        """
        camera_kwargs = camera_kwargs or {}
        self.cam = RealSenseCamera(**camera_kwargs)
        self.vlm = VLMProcessor()
        self.robot = RobotController(host=robot_host)

        self._initialized = False

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def setup(self):
        """Инициализирует железо: камера -> прогрев, робот -> connect+home."""
        try:
            self.cam.start()
            self.robot.connect()
            self._initialized = True
        except Exception as e:
            print(f"[PIPELINE] Failed to initialize hardware: {e}")
            raise

    def cleanup(self):
        """Освобождает ресурсы (idempotent — можно вызывать多次)."""
        try:
            self.cam.stop()
        except Exception:
            pass
        try:
            self.robot.freeze()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Публичные операции
    # ------------------------------------------------------------------
    def pick(self, prompt=None):
        """Полный цикл PICK: захват объекта по prompt."""
        return self._run_action(prompt=prompt, action=RobotController.ACTION_PICK)

    def put(self, prompt=None):
        """Полный цикл PUT:放置 объекта по prompt."""
        return self._run_action(prompt=prompt, action=RobotController.ACTION_PUT)

    # ------------------------------------------------------------------
    # Trajectory mode — НЕЗАВИСИМО от pick/put.
    # VLM строит траекторию → для первой точки (lift) вычисляется угол через
    # 10×10 bbox + контуры → robot.move_through_trajectory.
    # ------------------------------------------------------------------
    def run_trajectory(self, prompt=None):
        """Полный цикл TRAJECTORY (отдельно от pick/put):

        1. Захват кадра.
        2. VLM find_trajectory (двухстадийный: Stage 1 поиск + Stage 2 рефакторинг).
           Возвращает waypoints [[x, y, d], ...] в VLM-координатах (0..1000).
        3. Для первой точки (lift): вычисляется угол объекта через
           cam.find_angle_around_point(image, lift_pixel, window_size=10).
           Создаётся 10×10 px прямоугольник вокруг точки подъёма, внутри него
           ищется контур и возвращается угол объекта.
        4. Все waypoints конвертируются из VLM → пиксели → 3D камеры.
        5. Robot.move_through_trajectory(camera_points, image, lift_angle_deg).

        Returns: True, если траектория выполнена, False иначе.
        """
        if not self._initialized:
            print("[PIPELINE] Hardware not initialized. Call setup() first.")
            return False

        # 1. Захват кадра.
        image, depth_image, intrin = self.cam.capture_frame()
        if image is None:
            print("[PIPELINE] No image captured, aborting trajectory")
            return False

        # 2. Промпт.
        if prompt is None:
            prompt = input("Get prompt for trajectory: ").strip()
            if not prompt:
                print("[PIPELINE] Empty prompt, aborting trajectory")
                return False

        # 3. VLM trajectory (двухстадийный flow внутри inference).
        waypoints, traj_img = self.vlm.find_trajectory(image, prompt, plot=True)
        if waypoints is None or len(waypoints) < 2:
            print("[PIPELINE] Trajectory acquisition failed")
            return False
        if traj_img is not None:
            self.cam.show_img(img=traj_img, title="Trajectory")

        # 4. Конвертация ВСЕХ waypoints из VLM (0..1000) → пиксели.
        #    ИСПОЛЬЗУЕМ int(round(...)) — ТОЧНО как в оригинальном
        #    DetectedObject.convert_vlm_to_image (utils.py).
        #    int(...) без round() даёт truncate и дрейф ~1 px — раньше был баг.
        def _vlm_to_pixel(vlm_x, vlm_y) -> Point2D:
            """Точная копия логики convert_vlm_to_image(points=(x,y)) из utils.py."""
            return Point2D(
                x=int(round(int(vlm_x) / 1000.0 * self.cam.width)),
                y=int(round(int(vlm_y) / 1000.0 * self.cam.height)),
            )

        lift_wp = waypoints[0]
        placement_wp = waypoints[-1]
        middle_wps = waypoints[1:-1]

        lift_pixel = _vlm_to_pixel(lift_wp[0], lift_wp[1])
        placement_pixel = _vlm_to_pixel(placement_wp[0], placement_wp[1])
        middle_pixels = [_vlm_to_pixel(wp[0], wp[1]) for wp in middle_wps]

        # Debug-принт: VLM → пиксель для каждой точки.
        print(f"[PIPELINE] ---- VLM → PIXEL (cam {self.cam.width}x{self.cam.height}) ----")
        print(f"[PIPELINE] LIFT    VLM=({lift_wp[0]}, {lift_wp[1]})  "
              f"pixel=({lift_pixel.x}, {lift_pixel.y})")
        for i, (wp, mp) in enumerate(zip(middle_wps, middle_pixels)):
            print(f"[PIPELINE] MID[{i+1}]   VLM=({wp[0]}, {wp[1]})  "
                  f"pixel=({mp.x}, {mp.y})")
        print(f"[PIPELINE] PLACE   VLM=({placement_wp[0]}, {placement_wp[1]})  "
              f"pixel=({placement_pixel.x}, {placement_pixel.y})")

        # 5. Угол для первой точки (lift = точка подъёма).
        #    Берём lift_pixel, вокруг него строим 30×30 bbox и ищем контур.
        lift_angle_deg = self.cam.find_angle_around_point(
            image, lift_pixel, window_size=30
        )
        print(f"[PIPELINE] Lift angle (30x30 contour): {lift_angle_deg}°")

        # 6. Конвертация всех пикселей → 3D координаты камеры.
        #    Используем уже вычисленные пиксели (lift_pixel, middle_pixels,
        #    placement_pixel) — НЕ пересчитываем, чтобы не было расхождений.
        all_pixels = [lift_pixel] + middle_pixels + [placement_pixel]
        camera_points = []
        print(f"[PIPELINE] ---- PIXEL → 3D CAMERA ----")
        for wp, pix in zip(waypoints, all_pixels):
            pt3d = self.cam.deproject_pixel(pix.x, pix.y, depth_image, intrin)
            if pt3d is None:
                print(f"[PIPELINE] Failed to deproject pixel ({pix.x}, {pix.y}) "
                      f"(VLM {wp[0]}, {wp[1]}) — skipping")
                continue
            print(f"[PIPELINE] VLM=({wp[0]}, {wp[1]})  pixel=({pix.x}, {pix.y})  "
                  f"3D=({pt3d.x:.3f}, {pt3d.y:.3f}, {pt3d.z:.3f}) m")
            camera_points.append(pt3d)

        if len(camera_points) < 2:
            print("[PIPELINE] Not enough valid 3D points for trajectory "
                  f"(got {len(camera_points)}, need >= 2)")
            return False

        # 7. Визуальная верификация: рисуем все точки на кадре.
        #    LIFT — красный круг, PLACE — синий, middles — зелёные точки.
        #    Оператор видит ГДЕ окажется lift-точка после конверсии, и может
        #    понять, дрейфует ли координата на каком-то этапе.
        if not self._verify_trajectory_points(
            image, lift_pixel, placement_pixel, middle_pixels, lift_angle_deg
        ):
            print("[PIPELINE] User aborted at trajectory verification")
            return False

        # 8. Выполнение траектории роботом.
        #    Передаём пиксели lift/placement/middles для отрисовки в окнах
        #    подтверждения — оператор видит их при каждом шаге движения.
        return self.robot.move_through_trajectory(
            camera_points, image=image, lift_angle_deg=lift_angle_deg,
            lift_pixel=lift_pixel, placement_pixel=placement_pixel,
            middle_pixels=middle_pixels,
        )

    def _verify_trajectory_points(self, image, lift_pixel, placement_pixel,
                                  middle_pixels, lift_angle_deg) -> bool:
        """Рисует все точки траектории на кадре и ждёт подтверждения.

        LIFT    — красный круг + метка "LIFT"
        PLACE   — синий  круг + метка "PLACE"
        MIDDLE  — маленькие зелёные точки с номерами

        Возвращает True если 'C' (continue), False если 'Q' (abort).
        """
        disp = image.copy()

        # Middle waypoints (зелёные точки)
        if middle_pixels:
            for i, mp in enumerate(middle_pixels):
                cv2.circle(disp, (int(mp.x), int(mp.y)), 4, (0, 255, 0), -1)
                cv2.putText(disp, f"M{i+1}", (int(mp.x) + 6, int(mp.y) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Lift (красный круг + метка)
        if lift_pixel is not None:
            cv2.circle(disp, (int(lift_pixel.x), int(lift_pixel.y)),
                       8, (0, 0, 255), -1)
            cv2.putText(disp, "LIFT",
                        (int(lift_pixel.x) + 12, int(lift_pixel.y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Рисуем ось угла lift
            import math as _m
            end_x = int(lift_pixel.x + 40 * _m.cos(_m.radians(lift_angle_deg)))
            end_y = int(lift_pixel.y - 40 * _m.sin(_m.radians(lift_angle_deg)))
            cv2.line(disp, (int(lift_pixel.x), int(lift_pixel.y)),
                     (end_x, end_y), (0, 0, 255), 2)

        # Placement (синий круг + метка)
        if placement_pixel is not None:
            cv2.circle(disp, (int(placement_pixel.x), int(placement_pixel.y)),
                       8, (255, 0, 0), -1)
            cv2.putText(disp, "PLACE",
                        (int(placement_pixel.x) + 12, int(placement_pixel.y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.putText(disp,
                    f"LIFT pixel=({lift_pixel.x},{lift_pixel.y})  angle={lift_angle_deg}°  "
                    f"PLACE pixel=({placement_pixel.x},{placement_pixel.y})",
                    (10, disp.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp, "Verify points. Press 'C' to continue, 'Q' to abort",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Trajectory Points Verification", disp)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()
        return key in (ord('c'), ord('C'))

    def run(self, pick_prompt=None, put_prompt=None):
        """Полный цикл: setup -> pick -> put -> cleanup."""
        try:
            self.setup()
            self.pick(prompt=pick_prompt)
            self.put(prompt=put_prompt)
        finally:
            self.cleanup()

    # ------------------------------------------------------------------
    # Внутренний цикл одной операции
    # ------------------------------------------------------------------
    def _run_action(self, prompt, action):
        """Выполняет одну операцию (pick/put) от захвата кадра до движения."""
        if not self._initialized:
            print(f"[PIPELINE] Hardware not initialized. Call setup() first.")
            return False

        # 1. Захват кадра.
        image, depth_image, intrin = self.cam.capture_frame()
        if image is None:
            print(f"[PIPELINE] No image captured, aborting '{action}'")
            return False

        # 2. Промпт.
        if prompt is None:
            prompt = input(f"Get prompt for '{action}': ").strip()
            if not prompt:
                print(f"[PIPELINE] Empty prompt, aborting '{action}'")
                return False

        # 3. Полный поиск цели (VLM + контур + угол + второй VLM + 3D).
        target = self.acquire_target(prompt, image, depth_image, intrin)
        if target is None:
            print(f"[PIPELINE] Target acquisition failed, aborting '{action}'")
            return False

        # 4. Передача цели роботу.
        return self.robot.move_to_object(target, image, action=action)

    # ------------------------------------------------------------------
    # Поиск цели: VLM -> контур -> угол -> второй VLM -> 3D
    # ------------------------------------------------------------------
    def acquire_target(self, prompt, image, depth_image, intrin):
        """Полный конвейер приобретения цели.

        Шаги:
          1. VLM grounding — bbox объекта.
          2. Конур + угол (cv2 внутри bbox).
          3. VLM pointing (второй проход) — точный pickup-point.
          4. Депроекция пикселя в 3D камеры.

        Возвращает DetectedObject со всеми координатами либо None.
        """
        # 1. VLM grounding: bbox искомого объекта.
        coords, name, vlm_img = self.vlm.find_target(
            image, prompt, task=VLMProcessor.TASK_GROUNDING, plot=False
        )
        if coords is None:
            return None
        if vlm_img is not None:
            self.cam.show_img(img=vlm_img, title="VLM grounding")

        target = DetectedObject(
            name=name,
            vlm_point=Point2D(x=int(coords[0]), y=int(coords[1])),
            angle=0.0,
        )
        target.convert_vlm_to_image(self.cam.width, self.cam.height)
        target.create_area(self.cam.width, self.cam.height, coords)

        # 2. Контур и угол (через cv2 внутри bbox).
        contour_img, contour, center, angle = self.cam.find_object(image, target)
        if contour_img is not None:
            self.cam.show_img(img=contour_img, title="Contour")
        if center is not None:
            print(f"[PIPELINE] Contour center: {center}")
        target.contour = contour
        target.angle = angle
        print(f"[PIPELINE] angle = {angle}")

        # 3. Уточняем точку захвата вторым проходом VLM.
        coords2, name2, pickup_img = self.vlm.find_pickup_point(
            contour_img, prompt, plot=True
        )
        if coords2 is None:
            return None
        target.update_vlm_point(int(coords2[0]), int(coords2[1]),
                                self.cam.width, self.cam.height)
        if pickup_img is not None:
            cv2.circle(
                pickup_img,
                (target.image_point.x, target.image_point.y),
                5, (0, 0, 255), -1
            )
            self.cam.show_img(img=pickup_img, title="Pickup point")

        # 4. Депроекция в 3D координаты камеры.
        self.cam.compute_camera_point(target, depth_image, intrin)
        if target.camera_point is None:
            print("[PIPELINE] Failed to deproject 3D point")
            return None
        print(f"[PIPELINE] Camera 3D coords: "
              f"({target.camera_point.x:.3f}, "
              f"{target.camera_point.y:.3f}, "
              f"{target.camera_point.z:.3f})")

        return target