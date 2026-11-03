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
        #    ВСЕ way-ы через depth_image (lift, middles, placement) — как просил.
        #    z middles потом принудительно станет 0.4 в robot.move_through_trajectory.
        all_pixels = [lift_pixel] + middle_pixels + [placement_pixel]
        camera_points = []
        n_wp = len(waypoints)
        print(f"[PIPELINE] ---- PIXEL → 3D CAMERA (depth_image) ----")
        for i, (wp, pix) in enumerate(zip(waypoints, all_pixels)):
            is_lift = (i == 0)
            is_placement = (i == n_wp - 1)
            pt3d = self.cam.deproject_pixel(pix.x, pix.y, depth_image, intrin)
            if pt3d is None:
                print(f"[PIPELINE] Failed to deproject pixel ({pix.x}, {pix.y}) "
                      f"(VLM {wp[0]}, {wp[1]}) — skipping")
                continue
            label = "LIFT" if is_lift else ("PLACE" if is_placement else f"MID[{i}]")
            print(f"[PIPELINE] {label:5s} VLM=({wp[0]}, {wp[1]})  "
                  f"pixel=({pix.x}, {pix.y})  "
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
    # Plan mode — высокоуровневое планирование.
    #
    # ДВА режима (оператор выбирает в начале через input):
    #
    #   1) RESTORE  — восстановление последовательности (2 кадра).
    #      Оператор расставляет объекты в TARGET конфигурацию → 'p' (REFERENCE)
    #      → меняет сцену на CURRENT → 'p' (CURRENT).
    #      VLM получает [REFERENCE, CURRENT] + задачу и находит МИНИМАЛЬНЫЙ
    #      набор pick-and-place действий, преобразующих CURRENT в REFERENCE.
    #
    #   2) HIGH-LEVEL — просто high-level задача (1 кадр).
    #      Оператор жмёт 'p' один раз (CURRENT).
    #      VLM получает [CURRENT] + текстовую задачу и находит МИНИМАЛЬНЫЙ
    #      набор pick-and-place действий, удовлетворяющих задаче на видимой сцене.
    #
    # После получения плана никакой проверки оператором НЕТ — план сразу
    # уходит на выполнение. Каждый шаг по очереди идёт в run_trajectory
    # (который сам делает свежий кадр, строит траекторию и подтверждает
    # каждое движение робота). На первом же False шаге выполнение плана
    # прекращается.
    # ------------------------------------------------------------------
    def run_plan(self, prompt=None):
        """Полный цикл PLANNING (два режима — RESTORE / HIGH-LEVEL).

        Шаги:
          1. Выбор режима через input(): 1 = RESTORE (2 кадра), 2 = HIGH-LEVEL (1 кадр).
          2. Захват кадра(ов) — оператор видит live стрим и жмёт 'p':
               RESTORE     → capture_two_frames("REFERENCE", "CURRENT")
               HIGH-LEVEL  → capture_frame()
          3. Запрос high-level задачи (input()) — оператор вводит промпт,
             когда кадр(ы) уже зафиксирован(ы).
          4. VLM plan_task(images, prompt) → list[str] описаний траекторий.
          5. План выводится в консоль (также внутри plan_task).
             Никакой верификации оператором — сразу к выполнению.
          6. Цикл по шагам плана:
               ok = run_trajectory(step)
               если ok == False → прекратить выполнение плана.
          7. Return True если ВСЕ шаги выполнены, False — если прервано.

        GPU-кэш чистится автоматически: send_request (внутри plan_task и
        внутри run_trajectory) дёргает _cleanup_gpu() в finally.
        """
        if not self._initialized:
            print("[PIPELINE] Hardware not initialized. Call setup() first.")
            return False

        # 1. Выбор режима.
        print("\n" + "=" * 60)
        print("[PIPELINE] === PLAN MODE — choose mode ===")
        print("[PIPELINE]   1) RESTORE   — restore a sequence (2 frames: REFERENCE + CURRENT)")
        print("[PIPELINE]   2) HIGH-LEVEL — single-frame task (1 frame: CURRENT only)")
        print("=" * 60)
        mode_input = input("[PIPELINE] Enter mode (1 or 2): ").strip()
        if mode_input == "1":
            mode = "RESTORE"
        elif mode_input == "2":
            mode = "HIGH-LEVEL"
        else:
            print(f"[PIPELINE] Invalid mode '{mode_input}', aborting plan")
            return False
        print(f"[PIPELINE] Selected mode: {mode}\n")

        # 2. Захват кадра(ов) — оператор видит live стрим и жмёт 'p'.
        if mode == "RESTORE":
            print("\n" + "=" * 60)
            print("[PIPELINE] === RESTORE — two-frame capture ===")
            print("[PIPELINE] Step 1: arrange the scene in the TARGET configuration")
            print("[PIPELINE]         and press 'p' to capture REFERENCE (IMAGE 1).")
            print("[PIPELINE] Step 2: change the scene to the CURRENT state")
            print("[PIPELINE]         and press 'p' again to capture CURRENT (IMAGE 2).")
            print("=" * 60 + "\n")

            (ref_img, _ref_depth, _ref_intrin), \
                (cur_img, _cur_depth, _cur_intrin) = self.cam.capture_two_frames(
                    "REFERENCE", "CURRENT"
                )
            if ref_img is None or cur_img is None:
                print("[PIPELINE] Two-frame capture aborted, aborting plan")
                return False
            images = [ref_img, cur_img]
            print("\n[PIPELINE] Frames captured (REFERENCE + CURRENT).")
        else:  # HIGH-LEVEL
            print("\n" + "=" * 60)
            print("[PIPELINE] === HIGH-LEVEL — single-frame capture ===")
            print("[PIPELINE] Press 'p' to capture CURRENT (the only frame).")
            print("=" * 60 + "\n")

            cur_img, _cur_depth, _cur_intrin = self.cam.capture_frame()
            if cur_img is None:
                print("[PIPELINE] Capture aborted, aborting plan")
                return False
            images = [cur_img]
            print("\n[PIPELINE] Frame captured (CURRENT).")

        # 3. Промпт high-level задачи — оператор вводит его, когда кадр(ы)
        #    уже зафиксирован(ы).
        if prompt is None:
            prompt = input("\nGet high-level task: ").strip()
            if not prompt:
                print("[PIPELINE] Empty prompt, aborting plan")
                return False

        # 4. VLM plan_task — возвращает list[str] описаний траекторий.
        #    send_request внутри сам чистит temp-файлы и GPU-кэш.
        print(f"\n[PIPELINE] === Sending {len(images)} frame(s) to VLM planner ===")
        plan, _ = self.vlm.plan_task(images, prompt)
        if plan is None or len(plan) == 0:
            print("[PIPELINE] Plan generation failed, aborting")
            return False

        # 5. План уже выведен в консоль внутри vlm.plan_task.
        #    Дублируем кратко для удобства. Никакой верификации — сразу к
        #    выполнению.
        print("\n" + "=" * 60)
        print(f"[PIPELINE] === PLAN READY — {len(plan)} step(s), mode={mode} ===")
        for i, step in enumerate(plan, 1):
            print(f"[PIPELINE]   {i}/{len(plan)}: {step}")
        print("=" * 60 + "\n")

        # 6. Выполнение шагов плана.
        n = len(plan)
        print(f"[PIPELINE] === EXECUTING PLAN ({n} steps) ===\n")
        for i, step in enumerate(plan, 1):
            print("\n" + "-" * 60)
            print(f"[PIPELINE] === PLAN STEP {i}/{n} ===")
            print(f"[PIPELINE] Step description: {step}")
            print("-" * 60)

            # run_trajectory сам: делает свежий кадр → VLM trajectory →
            # конвертация в пиксели/3D → угол lift → подтверждение оператором
            # → robot.move_through_trajectory (с подтверждением КАЖДОГО шага
            # движения).
            ok = self.run_trajectory(step)
            if not ok:
                print(f"\n[PIPELINE] Step {i}/{n} FAILED — aborting plan execution")
                return False
            print(f"[PIPELINE] Step {i}/{n} OK")

        print(f"\n[PIPELINE] === PLAN COMPLETE ({n} steps executed) ===")
        self.vlm._cleanup_gpu()
        return True

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
