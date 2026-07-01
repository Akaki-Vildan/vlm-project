import os
import re
import tempfile
import math
from main import objects

import numpy as np
import cv2

from inference import UnifiedInference
import camera
import robot

# ==========================================
# Локальная модель RoboBrain 2.5
# ==========================================
LOCAL_MODEL_PATH = "/home/vildan/projects/vml-first-try/RoboBrain2.5/models"

if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
    print("ОШИБКА: config.json не найден в папке!")
    exit()

print("[STATUS] Loading local model weights...")
model = UnifiedInference(LOCAL_MODEL_PATH)


class object:
    """
    A class of object, with each we work at this moment
    Params:
        coords_vln - it is a coordinates, wich we have from vlm i.e. in 1000 x 1000 image
        coords_image - it's a vlm coordinated transformed in our image shape
        coords_camera - it's a coordinates in camera system returns in meters and calculates using image. Where are x, y and z coords
        coors_root - it's coordinates in base system
        angle - rotation angle
    """

    def __init__(self,
                 name = None, 
                 coords_vlm = None, #px_vlm
                 coords_camera = None, #px
                 coords_robot = None, 
                 angle = None):
                 
        self.name = name
        self.coords_vlm = coords_vlm
        self.coords_image = {
            "x": 0, 
            "y": 0, 
            "z": 0
        }
        self.coords_camera = coords_camera
        self.coords_robot = coords_robot
        self.angle = angle


        self.coords_image["x"] = int(round(self.coords_vlm["x"] / camera.VLM_SPACE * camera.CAMERA_WIDTH))
        self.coords_image["y"] = int(round(self.coords_vlm["y"] / camera.VLM_SPACE * camera.CAMERA_HEIGHT))




def send_a_request(prompt_text, image_data, task, do_sample=True, temperature=0.7):
    '''
    tasks: "general", "pointing", "trajectory", "grounding", "positioning"
    '''
    print("[STATUS] Sending request to model... This might take a moment.")
    temp_file_path = None

    # Если на вход пришёл кадр (numpy) — сохраняем во временный .jpg
    if isinstance(image_data, np.ndarray):
        tf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_file_path = tf.name
        tf.close()
        cv2.imwrite(temp_file_path, image_data)
        image_to_send = temp_file_path
    else:
        image_to_send = image_data

    try:
        pred = model.inference(prompt_text, image_to_send, task=task, do_sample=do_sample, temperature=temperature) 
        return pred
    finally:
        if temp_file_path is not None and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def get_coords_for_robot(pred, img, depth_image, depth_scale, intrin):
    new_obj = object()
    """
    Парсит пиксельные координаты из ответа VLM, депроецирует в 3D
    и переводит в позицию робота.
    """
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(pred))]
    if len(nums) < 2:
        print(f"[UTILS] Could not parse pixel coords from pred: {pred}")
        return None

    new_obj.coords_vlm["x"], new_obj.coords_vlm["y"], ang = nums[0], nums[1], nums[3]
    print(f"[UTILS] VLM pixel: x={new_obj.coords_vlm["x"]}, y={new_obj.coords_vlm["y"]}, angle={ang}")

    objects.append(new_obj)

    camera.pixel_to_camera_3d(
        depth_image, depth_scale, intrin, img)
    if objects[0].coords_camera is None:
        return None


    print(f"[UTILS] Camera 3D coords: {objects[0].coords_camera}")
    return robot.get_pos_from_cord(img, ang)



def create_angle_reference_image(image, px, py, radius=100):
    """
    Создаёт изображение с кругом, линиями и углами для VLM
    
    Args:
        image: исходное изображение с камеры
        px, py: координаты центра объекта
        radius: радиус круга
    
    Returns:
        image: изображение с визуализацией углов
    """
    img = image.copy()
    height, width = img.shape[:2]
    
    # 1. Рисуем круг (еле заметный)
    overlay = img.copy()
    cv2.circle(overlay, (px, py), radius, (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    # 2. Рисуем линии от края до края
    overlay_lines = img.copy()
    cv2.line(overlay_lines, (0, py), (width, py), (200, 200, 200), 1)
    cv2.line(overlay_lines, (px, 0), (px, height), (200, 200, 200), 1)
    cv2.addWeighted(overlay_lines, 0.3, img, 0.7, 0, img)
    
    # 3. Добавляем углы в радианах
    angles = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi, 
              5*math.pi/4, 3*math.pi/2, 7*math.pi/4]
    
    angle_labels = ["0", "π/4", "π/2", "3π/4", "π", 
                    "5π/4", "3π/2", "7π/4"]
    
    for angle, label in zip(angles, angle_labels):
        # Точка на круге
        x = int(px + radius * math.cos(angle))
        y = int(py - radius * math.sin(angle))
        
        # Рисуем точку
        cv2.circle(img, (x, y), 3, (100, 100, 100), -1)
        
        # Текст с отступом
        text_x = int(px + (radius + 20) * math.cos(angle))
        text_y = int(py - (radius + 20) * math.sin(angle))
        
        cv2.putText(img, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    return img


def get_an_angle():
    """
    Получает угол поворота объекта от VLM
    
    Returns:
        angle: угол в радианах
    """
    prompt = f"Get an angle of the {objects[0].name}"
    image, depth_image, depth_scale, intrin = camera.start_stream()
    
    # Получаем координаты центра объекта
    px = int(objects[0].coords_image["x"])
    py = int(objects[0].coords_image["y"])
    
    # Создаём изображение с визуализацией углов для VLM
    image_with_angles = create_angle_reference_image(image, px, py)
    
    # Отправляем изображение с визуализацией в VLM
    angle = float(send_a_request(prompt, image_with_angles, "find_angle", do_sample=False))
    
    return angle

