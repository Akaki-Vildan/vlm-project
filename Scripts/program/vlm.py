import os
import json
import tempfile
import math
import numpy as np
import cv2
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

    def send_request(self, prompt_text, image_data, task, do_sample=True, temperature=0.7):
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
            return self.model.inference(prompt_text, image_to_send, task=task, do_sample=do_sample, temperature=temperature)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @staticmethod
    def get_json_from_text(pred):
        if not pred: return None
        text = list(pred)[0].strip()
        for i in range(len(text)):
            if text[i] == "{" and text[i+1] == '"':
                s = i
            if text[i] == "}" and text[i-1] == '"':
                return json.loads(text[s:i+1])
        return None

    @staticmethod
    def create_angle_image(image, px, py, radius=100):
        img = image.copy()
        height, width = img.shape[:2]
        overlay = img.copy()
        cv2.circle(overlay, (px, py), radius, (255, 255, 255), 2)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
        labels = ["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4"]
        
        for angle, label in zip(angles, labels):
            x = int(px + radius * math.cos(angle))
            y = int(py - radius * math.sin(angle))
            cv2.circle(img, (x, y), 3, (100, 100, 100), -1)
            text_x = int(px + (radius + 20) * math.cos(angle))
            text_y = int(py - (radius + 20) * math.sin(angle))
            cv2.putText(img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        return img

    def get_object_angle(self, target_obj: DetectedObject, camera_instance):
        prompt = f"Get an angle of the {target_obj.name}"
        image, depth_image, intrin = camera_instance.capture_frame()
        if not image: return None
        
        image_with_angles = self.create_angle_image(image, int(target_obj.image_point.x), int(target_obj.image_point.y))
        return float(self.send_request(prompt, image_with_angles, "find_angle", do_sample=False))