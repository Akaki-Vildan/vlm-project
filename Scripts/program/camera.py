import pyrealsense2 as rs
import numpy as np
import cv2
from utils import Point3D

class RealSenseCamera:

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        
        # ADD self. HERE
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = 0.0

    def start(self):
        """Call this to actually turn on the hardware"""
        # ADD self. HERE
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        
        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("Camera ready!\n")

    def capture_frame(self):
        """Returns color_image, depth_image, intrinsics. Returns None if quit."""
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
            intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
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
            

    def deproject_pixel(self, px, py, depth_image, intrin):
        """Deprojects a single pixel using median depth. Returns Point3D"""
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

    def process_image_from_vlm(image, bbox):
        """
        Находит контур искомого объекта по bounding box и возвращает 
        изображение с нарисованным контуром и координаты центра.
        
        :param image: Исходное изображение (numpy.ndarray, формат BGR)
        :param bbox: Координаты области в формате [x1, y1, x2, y2]
        :return: Изображение с контуром (numpy.ndarray), координаты центра (cx, cy)
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("На вход ожидается изображение в формате numpy.ndarray")

        x1, y1, x2, y2 = bbox

        # Защита от выхода за границы изображения
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        # 1. Вырезаем область (ROI)
        roi = image[y1:y2, x1:x2]

        # Если область получилась пустой (например, некорректные координаты)
        if roi.size == 0:
            return image.copy(), None

        # 2. Подготовка: перевод в Ч/Б и размытие для удаления шума
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. Выделение границ (Canny)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Замыкаем возможные разрывы в контуре
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # 4. Поиск контуров внутри вырезанной области
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image.copy(), None

        # 5. Выбираем самый большой контур
        largest_contour = max(contours, key=cv2.contourArea)

        # 6. Сдвигаем координаты контура в систему координат ИСХОДНОГО изображения
        shifted_contour = largest_contour + np.array([x1, y1])

        # 7. Вычисление центра объекта с помощью пространственных моментов
        M = cv2.moments(shifted_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # Резервный вариант: центр Bounding Box
            cx = x1 + (x2 - x1) // 2
            cy = y1 + (y2 - y1) // 2

        # 8. Рисуем контур и центр на копии исходного изображения
        result_image = image.copy()
        
        cv2.drawContours(result_image, [shifted_contour], -1, (0, 255, 0), 2)
        cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)

        return result_image, (cx, cy)

    @staticmethod
    def show_img(img, title = "Image"):
        print("[CAM] show image starts")
        cv2.imshow(title, img)
        
        # Бесконечный цикл, который держит окно открытым
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Если нажали 'i' - закрываем окно и выходим из функции (идем дальше по коду)
            if key == ord('i'):
                cv2.destroyAllWindows()
                break
                
            # Опционально: если нажали 'q' - тоже закрываем (на всякий случай)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                break
