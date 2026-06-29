import os
import re
import tempfile

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


def send_a_request(prompt_text, image_data):
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
        pred = model.inference(prompt_text, image_to_send, task="pointing")
        return pred
    finally:
        if temp_file_path is not None and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def get_coords_for_robot(pred, img, depth_image, depth_scale, intrin):
    """
    Парсит пиксельные координаты из ответа VLM, депроецирует в 3D
    и переводит в позицию робота.
    """
    nums = [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", str(pred))]
    if len(nums) < 2:
        print(f"[UTILS] Could not parse pixel coords from pred: {pred}")
        return None

    px_vlm, py_vlm = nums[0], nums[1]
    print(f"[UTILS] VLM pixel: x={px_vlm}, y={py_vlm}")

    coord_camera = camera.pixel_to_camera_3d(
        depth_image, depth_scale, intrin, px_vlm, py_vlm, img)
    if coord_camera is None:
        return None

    print(f"[UTILS] Camera 3D coords: {coord_camera}")
    return robot.get_pos_from_cord(coord_camera, img)
