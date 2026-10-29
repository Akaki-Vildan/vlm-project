import os

# [MEM FIX] Должно быть ДО импорта torch / inference,
# иначе PyTorch уже инициализирует кэш аллокатора со старыми настройками.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import ast
import gc
import math
import tempfile

import cv2
import numpy as np
import torch  # [MEM FIX] нужен для empty_cache / synchronize / mem info

from inference import UnifiedInference
from utils import DetectedObject

LOCAL_MODEL_PATH = "/home/vildan/projects/vml-first-try/RoboBrain2.5/models"


class VLMProcessor:
    """Обёртка над UnifiedInference.

    Скрывает от вызывающего кода временные файлы, очистку GPU-памяти и
    разбор «грязного» ответа модели. На верхнем уровне exposes
        send_request(prompt, image, task)         -> (pred_text, annotated_image_or_None)
        find_target(image, prompt, task)          -> (coords, name, annotated_image)
        find_pickup_point(contour_image, prompt)  -> (coords, name, annotated_image)
    """

    # Задачи, для которых есть отдельный промпт в UnifiedInference.
    TASK_POINTING = "pointing"
    TASK_TRAJECTORY = "trajectory"
    TASK_GROUNDING = "grounding"
    TASK_POSITIONING = "positioning"
    TASK_FIND_ANGLE = "find_angle"
    TASK_PLANING = "planing"
    TASK_GENERAL = "general"

    def __init__(self):
        if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
            print("ОШИБКА: config.json не найден в папке!")
            exit()
        print("[STATUS] Loading local model weights...")
        self.model = UnifiedInference(LOCAL_MODEL_PATH)
        # [MEM FIX] Чистим кэш сразу после загрузки весов —
        # during from_pretrained накапливается много временных тензоров.
        self._cleanup_gpu()

    # ------------------------------------------------------------------
    # [MEM FIX] Единая точка очистки GPU памяти.
    # ------------------------------------------------------------------
    def _cleanup_gpu(self, label: str = ""):
        """Сборка мусора + сброс кэша аллокатора PyTorch + синхронизация."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # дождаться окончания асинхронных операций
            torch.cuda.empty_cache()   # вернуть "reserved but unallocated" в CUDA
            torch.cuda.ipc_collect()   # подчистить IPC-блоки (на всякий)
        if label:
            self._log_mem(label)

    def _log_mem(self, label: str):
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free, total = torch.cuda.mem_get_info()
        free /= 1024**3
        total /= 1024**3
        print(f"[MEM {label}] alloc={allocated:.2f}G  reserved={reserved:.2f}G  "
              f"free={free:.2f}G / total={total:.2f}G")

    # ------------------------------------------------------------------
    # Низкоуровневый запрос
    # ------------------------------------------------------------------
    def send_request(self, prompt_text, image_data, task,
                     do_sample=True, temperature=0.7, plot=False):
        """Отправляет изображение + промпт в модель и возвращает
        (pred_text, annotated_image_or_None).

        Временные файлы и GPU-кэш чистятся в finally даже при падении инференса.
        """
        temp_file_path = None
        if isinstance(image_data, np.ndarray):
            tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_file_path = tf.name
            tf.close()
            cv2.imwrite(temp_file_path, image_data)
            image_to_send = temp_file_path
        else:
            image_to_send = image_data

        try:
            return self.model.inference(
                prompt_text, image_to_send, task=task,
                do_sample=do_sample, temperature=temperature, plot=plot,
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            self._cleanup_gpu(label=f"after:{task}")

    # ------------------------------------------------------------------
    # Высокоуровневые методы —这才是caller должен использовать.
    # ------------------------------------------------------------------
    def find_target(self, image, prompt, task=TASK_GROUNDING, plot=False):
        """Один проход VLM с готовым парсингом ответа.

        Возвращает кортеж (coords, name, annotated_image_or_None).
        На любой неудаче возвращает (None, None, None) — вызывающий код
        не должен сам разбирать pred и проверять falsy.
        """
        pred, img = self.send_request(prompt, image, task, plot=plot)
        if not pred:
            print(f"[VLM] No prediction for task='{task}'")
            return None, None, None
        print(f"[VLM] prediction: {pred}")

        data = self.safe_parse(pred)
        if not data or not isinstance(data, (list, tuple)) or len(data) < 2:
            print("[VLM] Failed to parse VLM response")
            return None, None, None

        coords = data[0]
        name = data[-1]
        print(f"[VLM] coords: {coords}, name: {name}")
        return coords, name, img

    def find_pickup_point(self, contour_image, prompt, plot=True):
        """Второй проход VLM: ищет точку захвата на вырезанном контуре.

        Возвращает (coords, name, annotated_image_or_None) или (None, None, None).
        Здесь промпт конструируется ЦЕНТРАЛИЗОВАННО, а не через f-string
        конкатенацию в main.py.
        """
        full_prompt = self._build_pickup_prompt(prompt)
        return self.find_target(contour_image, full_prompt,
                                task=self.TASK_POINTING, plot=plot)

    # ------------------------------------------------------------------
    # Trajectory (независимая функция — НЕ часть pick/put pipeline)
    # ------------------------------------------------------------------
    def find_trajectory(self, image, prompt, plot=True):
        """Двухстадийный поиск траектории.

        Stage 1: модель строит траекторию по исходному изображению + промпту.
                 Выход: список [[x, y, d], ...] (БЕЗ object_name).
        Stage 2 (внутри inference): модель получает ИЗОБРАЖЕНИЕ С НАРИСОВАННОЙ
                 траекторией + Stage-1 ответ и верифицирует/исправляет его
                 (проверка end-points, плавности, обхода препятствий).

        Args:
            image: np.ndarray или путь к картинке.
            prompt: текстовая команда оператора.
                    Примеры: "Pick up the red cube and place it on the chair"
                             "Lift the blue cup"
            plot: если True — вернуть финальное изображение с траекторией.

        Returns:
            (waypoints, annotated_image_or_None)
            waypoints: список вида [[x, y, d], ...] (int, int, float)
            На неудаче: (None, None).

        Примечание: if в промпте упомянут только один объект (без placement),
        Stage 2 принудит LAST = FIRST (поставить обратно на то же место).
        """
        pred, img = self.send_request(prompt, image, self.TASK_TRAJECTORY, plot=plot)
        if not pred:
            print("[VLM] No trajectory prediction")
            return None, None
        print(f"[VLM] trajectory prediction: {pred}")

        data = self.safe_parse(pred)
        # New CLICHE: [[x,y,d], ...] — plain list of waypoints, no name string.
        if not data or not isinstance(data, (list, tuple)):
            print("[VLM] Failed to parse trajectory response")
            return None, None

        # Defensive: some old answers may still wrap waypoints + name.
        # If data looks like [waypoints_list, "name"], unwrap.
        if (len(data) == 2
                and isinstance(data[0], (list, tuple))
                and isinstance(data[1], str)):
            waypoints = list(data[0])
        else:
            waypoints = list(data)

        # Basic sanity: each waypoint should be [x, y, d].
        clean = []
        for wp in waypoints:
            if isinstance(wp, (list, tuple)) and len(wp) == 3:
                clean.append([int(wp[0]), int(wp[1]), float(wp[2])])
        if len(clean) < 2:
            print(f"[VLM] Trajectory has < 2 valid waypoints: {clean}")
            return None, None

        print(f"[VLM] trajectory waypoints: {clean}")
        return clean, img

    @staticmethod
    def _build_pickup_prompt(user_prompt: str) -> str:
        """Формирует промпт для второго прохода (поиск точки захвата)."""
        return (
            f"{user_prompt}. "
            f"Look at the contour image and find the exact pickup point "
            f"inside the framed object area."
        )

    # ------------------------------------------------------------------
    # Служебные статические методы (из оригинала, без изменений)
    # ------------------------------------------------------------------
    @staticmethod
    def get_coordinates(pred):
        def extract_string(data):
            if isinstance(data, str):
                return data
            if isinstance(data, (tuple, list, set)):
                for item in data:
                    res = extract_string(item)
                    if res is not None:
                        return res
            return None

        text = extract_string(pred)
        if not text:
            return None

        try:
            parsed_data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if not isinstance(parsed_data, list):
            return None

        result = [list(coords) for coords in parsed_data]
        if len(result) == 1:
            return result[0]
        elif len(result) > 1:
            return result
        return None

    @staticmethod
    def create_angle_image(image, px, py, radius=100):
        img = image.copy()

        overlay = img.copy()
        cv2.circle(overlay, (px, py), radius, (255, 255, 255), 1)

        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi,
                  -3*math.pi/4, -math.pi/2, -math.pi/4]
        labels = ["0", "pi/4", "pi/2", "3pi/4", "pi", "-3pi/4", "-pi/2", "-pi/4"]

        for angle, label in zip(angles, labels):
            shifted_angle = angle - math.pi / 2
            x = int(px + radius * math.cos(shifted_angle))
            y = int(py - radius * math.sin(shifted_angle))
            cv2.line(overlay, (px, py), (x, y), (255, 255, 255), 1)
            cv2.circle(overlay, (x, y), 2, (200, 200, 200), -1)

        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        cv2.circle(img, (px, py), 4, (0, 0, 255), -1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)

        for angle, label in zip(angles, labels):
            shifted_angle = angle - math.pi / 2
            text_x = int(px + (radius + 20) * math.cos(shifted_angle))
            text_y = int(py - (radius + 20) * math.sin(shifted_angle))
            cv2.putText(img, label, (text_x, text_y), font, font_scale,
                        (0, 0, 0), thickness + 2)
            cv2.putText(img, label, (text_x, text_y), font, font_scale,
                        color, thickness)

        return img

    def get_object_angle(self, target_obj: DetectedObject, camera_instance):
        prompt = f"Get an angle of the {target_obj.name}"
        camera_instance.start()
        image, depth_image, intrin = camera_instance.capture_frame()

        if image is None:
            print("Error: Camera returned None")
            return None

        if image.dtype != np.uint8:
            if np.max(image) <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        cv2.imshow('RAW CAMERA IMAGE (Press any key)', image)
        cv2.waitKey(0)

        image_with_angles = self.create_angle_image(
            image, int(target_obj.image_point.x), int(target_obj.image_point.y))
        cv2.imshow('image with angles', image_with_angles)

        print("Press 'p' to send to VLM, or 'q' to quit...")
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            cv2.destroyAllWindows()
            if hasattr(self, 'pipeline'):
                self.pipeline.stop()
            return None
        elif key == ord('p'):
            cv2.destroyAllWindows()
            self._cleanup_gpu(label="before:inference")
            return self.send_request(prompt, image_with_angles, "find_angle",
                                     do_sample=False, temperature=0.5)

        return None

    @staticmethod
    def safe_parse(text):
        """Бронебойный парсер, который чистит мусор от LLM."""
        if not text or not text.strip().startswith('['):
            return text

        text = text.strip()
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('(', "[").replace(")", "]")

        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            import string
            text = text.strip(string.whitespace + string.punctuation + '”“‘’«»\x00\x0b\x0c')
            try:
                return ast.literal_eval(text)
            except Exception:
                return text