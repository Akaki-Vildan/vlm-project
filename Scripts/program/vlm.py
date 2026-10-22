import os
# [MEM FIX] Должно быть ДО импорта torch / inference,
# иначе PyTorch уже инициализирует кэш аллокатора со старыми настройками.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import tempfile
import math
import numpy as np
import cv2
import ast
import torch  # [MEM FIX] нужен для empty_cache / synchronize / mem info
from inference import UnifiedInference
from utils import DetectedObject, Point2D

LOCAL_MODEL_PATH = "/home/vildan/projects/vml-first-try/RoboBrain2.5/models"


class VLMProcessor:
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
    # Вызывается: после загрузки модели, после каждого send_request,
    # и может вызываться вручную из main.py между итерациями.
    # ------------------------------------------------------------------
    def _cleanup_gpu(self, label: str = ""):
        """Сборка мусора + сброс кэша аллокатора PyTorch + синхронизация."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # дождаться окончания асинхронных операций
            torch.cuda.empty_cache()   # вернуть "reserved but unallocated" в CUDA
            torch.cuda.ipc_collect()    # подчистить IPC-блоки (на всякий)
        if label:
            self._log_mem(label)

    def _log_mem(self, label: str):
        """Логирование для отладки — сколько занято / свободно."""
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        free, total = torch.cuda.mem_get_info()
        free /= 1024**3
        total /= 1024**3
        print(f"[MEM {label}] alloc={allocated:.2f}G  reserved={reserved:.2f}G  "
              f"free={free:.2f}G / total={total:.2f}G")

    # ------------------------------------------------------------------
    # send_request — обёрнут в try/finally с очисткой GPU.
    # Важно: очистка срабатывает даже если inference упал с OOM,
    # чтобы следующий запрос мог стартовать на чистой памяти.
    # ------------------------------------------------------------------
    def send_request(self, prompt_text, image_data, task, do_sample=True, temperature=0.7, plot=False):
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
            # Сначала удаляем временный файл
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            # [MEM FIX] Потом чистим GPU — независимо от того,
            # успешно прошёл inference или упал.
            self._cleanup_gpu(label=f"after:{task}")

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
            # [MEM FIX] перед тяжёлым инференсом тоже подчищаем —
            # OpenCV мог насохранять всякого в GPU через CUDA-бэкенд.
            self._cleanup_gpu(label="before:inference")
            return self.send_request(prompt, image_with_angles, "find_angle",
                                     do_sample=False, temperature=0.5)

        return None

    @staticmethod
    def safe_parse(text):
        """Бронебойный парсер, который чистит мусор от LLM"""
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
            except:
                return text