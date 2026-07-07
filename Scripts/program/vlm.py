import os
import json
import tempfile
import math
import numpy as np
import cv2
import ast
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
            return self.model.inference(prompt_text, image_to_send, task=task, do_sample=do_sample, temperature=temperature, plot=plot)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @staticmethod
    def get_coordinates(pred):
        # 1. Рекурсивно ищем строку внутри любой вложенности (tuple, set, list)
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

        # 2. Безопасно превращаем строку в Python объект
        # Например: "[(1, 2, 3)]" превратится в реальный список кортежей [(1, 2, 3)]
        try:
            parsed_data = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

        if not isinstance(parsed_data, list):
            return None

        # 3. Превращаем все кортежи внутри в списки
        result = [list(coords) for coords in parsed_data]

        # 4. Возвращаем в нужном формате по условию:
        # Если один объект -> возвращаем плоский список [x, y, z]
        if len(result) == 1:
            return result[0]
        
        # Если несколько объектов -> возвращаем список списков [[x1,y1,z1], [x2,y2,z2]]
        elif len(result) > 1:
            return result

        return None

    @staticmethod
    def create_angle_image(image, px, py, radius=100):
        img = image.copy()
        
        # ==========================================
        # ШАГ 1: Полупрозрачные элементы (Круг и Линии)
        # ==========================================
        overlay = img.copy()
        cv2.circle(overlay, (px, py), radius, (255, 255, 255), 1)
        
        # Новая система координат: 0 внизу, pi/2 справа, pi вверху, -pi/2 слева
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, -3*math.pi/4, -math.pi/2, -math.pi/4]
        labels = ["0", "pi/4", "pi/2", "3pi/4", "pi", "-3pi/4", "-pi/2", "-pi/4"]
        
        for angle, label in zip(angles, labels):
            # Сдвигаем систему координат: минус pi/2 делает так, что 0 смотрит строго вниз
            shifted_angle = angle - math.pi / 2
            
            x = int(px + radius * math.cos(shifted_angle))
            y = int(py - radius * math.sin(shifted_angle))
            
            cv2.line(overlay, (px, py), (x, y), (255, 255, 255), 1)
            cv2.circle(overlay, (x, y), 2, (200, 200, 200), -1)
            
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # ==========================================
        # ШАГ 2: Четкие элементы (Точка в центре и Текст)
        # ==========================================
        cv2.circle(img, (px, py), 4, (0, 0, 255), -1)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255) 
        
        for angle, label in zip(angles, labels):
            # ИСПРАВЛЕНИЕ: Текст тоже использует сдвиг, чтобы не отрываться от линий!
            shifted_angle = angle - math.pi / 2
            
            text_x = int(px + (radius + 20) * math.cos(shifted_angle))
            text_y = int(py - (radius + 20) * math.sin(shifted_angle))
            
            # Черная обводка для читаемости
            cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2)
            # Белый текст
            cv2.putText(img, label, (text_x, text_y), font, font_scale, color, thickness)

        return img

    def get_object_angle(self, target_obj: DetectedObject, camera_instance):
            prompt = f"Get an angle of the {target_obj.name}"
            camera_instance.start()
            image, depth_image, intrin = camera_instance.capture_frame()
            
            if image is None: 
                print("Error: Camera returned None")
                return None
                
            # FIX 3: Convert float32 (0.0-1.0) to uint8 (0-255) if necessary
            if image.dtype != np.uint8:
                if np.max(image) <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)

            # FIX 2: Convert RGB to BGR if the camera outputs RGB (common in RealSense)
            # Uncomment the next line if your raw image looks like it has inverted/strange colors
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Debug step: Check if the RAW image is black
            cv2.imshow('RAW CAMERA IMAGE (Press any key)', image)
            cv2.waitKey(0) # Wait for a key press

            image_with_angles = self.create_angle_image(image, int(target_obj.image_point.x), int(target_obj.image_point.y))
            cv2.imshow('image with angles', image_with_angles)
            
            # FIX 1: Use waitKey(0) instead of waitKey(1). 
            # This stops the code here indefinitely, allowing the OS to paint the pixels!
            print("Press 'p' to send to VLM, or 'q' to quit...")
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                # Stopping camera safely
                if hasattr(self, 'pipeline'): 
                    self.pipeline.stop()
                return None
                
            elif key == ord('p'):
                cv2.destroyAllWindows()
                # NOW we do the heavy inference, after the window is closed or updated
                return self.send_request(prompt, image_with_angles, "find_angle", do_sample=False, temperature=0.5)

            return None
    
    
        # Return unified format
    @staticmethod
    def safe_parse(text):
            """Бронебойный парсер, который чистит мусор от LLM"""
            if not text or not text.strip().startswith('['):
                return text  # Если это не список, возвращаем как есть (для general)
            
            text = text.strip()
            
            # 1. Заменяем "умные" кавычки на обычные (очень частая проблема)
            text = text.replace('\u201c', '"').replace('\u201d', '"')
            text = text.replace('\u2018', "'").replace('\u2019', "'")
            
            # 2. Пытаемся распарсить
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                # 3. Если не вышло, вырезаем ВСЕ невидимые спецсимволы по краям и пробуем еще раз
                import string
                text = text.strip(string.whitespace + string.punctuation + '”“‘’«»\x00\x0b\x0c')
                try:
                    return ast.literal_eval(text)
                except:
                    return text # Если всё равно ошибка, возвращаем строку
