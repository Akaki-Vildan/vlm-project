import cv2
import numpy as np
import pyrealsense2 as rs

from utils import Point2D, Point3D


class RealSenseCamera:
    """Обёртка над RealSense: захват, выравнивание depth/color, депроекция
    пикселя в 3D-точку камеры, поиск контура объекта внутри bbox и расчёт
    угла поворота по контуру.
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()

        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = 0.0

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------
    def start(self):
        """Включает камеру и прогревает её (30 фреймов)."""
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = (
            self.profile.get_device().first_depth_sensor().get_depth_scale()
        )
        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("Camera ready!\n")

    def stop(self):
        """Безопасно останавливает поток камеры (idempotent)."""
        try:
            self.pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # Захват
    # ------------------------------------------------------------------
    def capture_frame(self):
        """Живой поток: 'p' - захват, 'q' - выход.

        Возвращает (color_image, depth_image, intrinsics) либо (None, None, None).
        """
        print("Streaming... Press 'p' to capture, or 'q' to quit.")
        while True:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intrin = (
                depth_frame.get_profile()
                .as_video_stream_profile()
                .get_intrinsics()
            )

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )
            cv2.imshow('Simple Stream', color_image)
            cv2.imshow('Depth Stream', depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.pipeline.stop()
                cv2.destroyAllWindows()
                return color_image.copy(), depth_image.copy(), intrin
            elif key == ord('q'):
                self.pipeline.stop()
                cv2.destroyAllWindows()
                return None, None, None

    # ------------------------------------------------------------------
    # 3D-депроекция
    # ------------------------------------------------------------------
    def deproject_pixel(self, px, py, depth_image, intrin):
        """Депроектирует один пиксель в 3D-точку камеры.
        Глубина берётся как медиана по окну 11x11 вокруг пикселя.

        Возвращает Point3D либо None, если в окне нет валидной глубины.
        """
        half = 5
        y0, y1 = max(0, py - half), min(self.height, py + half + 1)
        x0, x1 = max(0, px - half), min(self.width, px + half + 1)

        window = depth_image[y0:y1, x0:x1].astype(np.float32) * self.depth_scale
        valid = window[window > 0]
        if valid.size == 0:
            print("[CAMERA] depth is None (no valid depth in window)")
            return None

        depth = float(np.median(valid))
        point_3d = rs.rs2_deproject_pixel_to_point(intrin, [px, py], depth)
        return Point3D(point_3d[0], point_3d[1], point_3d[2])

    def compute_camera_point(self, target, depth_image, intrin):
        """Депроектирует target.image_point в 3D и сохраняет в target.camera_point.
        Возвращает target.camera_point (Point3D или None).
        """
        target.camera_point = self.deproject_pixel(
            target.image_point.x, target.image_point.y, depth_image, intrin
        )
        return target.camera_point

    # ------------------------------------------------------------------
    # Контур и угол
    # ------------------------------------------------------------------
    def process_image_from_vlm(self, image, first_point, second_point):
        """Находит контур искомого объекта по bbox и возвращает изображение с
        нарисованным контуром, координаты центра и сам контур.

        !!! КОНТРАКТ: всегда возвращает 3-элементный кортеж
            (result_image, center_or_None, contour_or_None)
        На любой отказ (пустой ROI, нет контуров) возвращается
            (image.copy(), None, None)
        Это устраняет костыльную проверку типа возвращаемого значения
        в вызывающем коде.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("На вход ожидается изображение в формате numpy.ndarray")

        x1 = first_point.x - 10
        y1 = first_point.y - 10
        x2 = second_point.x + 10
        y2 = second_point.y + 10
        print(f"[CAMERA] Coords in the start of the process image from vlm "
              f"function: x1: {x1}, y1: {y1}, x2: {x2}, y2:{y2}")

        # Защита от выхода за границы изображения.
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        # 1. Вырезаем ROI.
        roi = image[y1:y2, x1:x2]
        rect_img = image.copy()
        cv2.rectangle(rect_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        self.show_img(img=rect_img, title="Area")

        if roi.size == 0:
            print("[CAMERA] Roi size is wrong")
            return image.copy(), None, None

        # 2. Подготовка: Ч/Б + размытие.
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        self.show_img(img=gray, title="Roi")

        # 3. Canny + dilate (замыкаем разрывы контура).
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 4. Поиск контуров.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("[CAMERA] There are no countors")
            return image.copy(), None, None

        # 5. Самый большой контур.
        largest_contour = max(contours, key=cv2.contourArea)

        # 6. Сдвиг в систему координат исходного изображения.
        shifted_contour = largest_contour + np.array([x1, y1])

        # 7. Центр через пространственные моменты (fallback — центр bbox).
        M = cv2.moments(shifted_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx = x1 + (x2 - x1) // 2
            cy = y1 + (y2 - y1) // 2

        # 8. Рисуем контур и центр на копии исходного изображения.
        result_image = image.copy()
        cv2.drawContours(result_image, [shifted_contour], -1, (0, 255, 0), 2)
        cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)

        return result_image, (cx, cy), shifted_contour

    def find_object(self, image, target):
        """Высокоуровневая обёртка: ищет контур внутри target.area и считает угол.

        Возвращает кортеж (result_image, contour, center, angle).
        Если bbox или контур отсутствуют:
            (image.copy(), None, None, 0.0)
        """
        if target.area is None:
            print("[CAMERA] Target has no bbox, skipping contour search")
            return image.copy(), None, None, 0.0

        result_image, center, contour = self.process_image_from_vlm(
            image, target.area.first_point, target.area.second_point
        )
        if contour is None:
            return result_image, None, None, 0.0
        angle = self.get_rotation_angle(contour)
        return result_image, contour, center, angle

    def find_angle_around_point(self, image, point, window_size=30):
        """Определяет угол объекта вокруг заданной точки через контуры.

        Алгоритм (по спеке оператора):
          1. Создаём прямоугольник window_size × window_size px с центром
             в `point` (по умолчанию 30 → 30×30 px, увеличено с 10 т.к.
             на 10×10 контуры детектировались плохо).
          2. Передаём его как bbox в process_image_from_vlm (которая внутри
             добавит ещё по 10 px безопасности для Canny, найдёт контуры и
             вернёт самый большой) — эффективный ROI ~ (window_size + 20) px.
          3. Считаем угол через get_rotation_angle(contour).

        Args:
            image:        numpy BGR изображение.
            point:        Point2D с пиксельными координатами (x, y).
            window_size:  размер бокса в пикселях. По умолчанию 30 → 30×30 px.

        Returns:
            Угол в градусах (float). 0.0, если контур не найден.
        """
        half = max(1, window_size // 2)
        first = Point2D(x=int(point.x) - half, y=int(point.y) - half)
        second = Point2D(x=int(point.x) + half, y=int(point.y) + half)

        print(f"[CAMERA] find_angle_around_point: center=({point.x}, {point.y}), "
              f"window={window_size}x{window_size} px "
              f"(effective ROI ~{window_size + 20}x{window_size + 20} with padding)")

        _, _, contour = self.process_image_from_vlm(image, first, second)
        if contour is None:
            print(f"[CAMERA] No contour found in {window_size}x{window_size} "
                  f"window around ({point.x}, {point.y}) — returning 0.0")
            return 0.0
        angle = self.get_rotation_angle(contour)
        print(f"[CAMERA] Angle around ({point.x}, {point.y}): {angle}°")
        return angle

    def get_rotation_angle(self, contour):
        """Возвращает угол поворота объекта относительно оси X (в градусах).
        Положительный угол — против часовой стрелки.
        """
        rect = cv2.minAreaRect(contour)   # (center, (w, h), angle)
        angle = rect[2]

        # cv2.minAreaRect возвращает угол в диапазоне (-90, 0].
        # Приводим к диапазону [0, 180) — угол длинной стороны относительно оси X.
        w, h = rect[1]
        if w < h:
            angle = 90 + angle
        return angle

    # ------------------------------------------------------------------
    # Утилиты отображения
    # ------------------------------------------------------------------
    @staticmethod
    def show_img(img, title="Image"):
        """Показывает изображение и ждёт 'i' или 'q' для закрытия окна."""
        print("[CAM] show image starts")
        cv2.imshow(title, img)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('i'), ord('q')):
                cv2.destroyAllWindows()
                break
