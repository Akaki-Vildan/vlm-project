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



def get_an_angle():
    prompt = f"Get an agle, of the {objects[0].name}"
    image, depth_image, depth_scale, intrin = camera.start_stream()

    

    angle = float(send_a_request(prompt, image, "find_angle", do_sample=False))
    return angle

