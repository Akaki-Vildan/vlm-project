import sys
import os
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import DetectedObject, Point2D
from camera import RealSenseCamera
from robot import RobotController
from vlm import VLMProcessor

def main():
    # # 1. Initialize Systems (Hardware doesn't turn on until we call .start() or .connect())
    cam = RealSenseCamera()
    vlm = VLMProcessor()
    robot = RobotController()

    try:
        cam.start()
        robot.connect()
        robot.toggle_free_drive()
    except Exception as e:
        print(f"Failed to initialize hardware: {e}")
        return

    # 2. Capture Frame
    image, depth_image, intrin = cam.capture_frame()
    if image is None:
        print("No image captured.")
        return

    prompt = input('Get prompt: ')
    if not prompt:
        return

    '''
    Изменить структуру промпта можно и в inference. Идея следующая: Сделать цикличный промпт. Т.е. первый этап поиск точки захвата, второй этап контроль качества: 
    проверяет полученный ответ с клише, если сходство более 95% то выдавать ответ, если нет, изменять так, чтоб подходил под клише. 
    Так же в начале задается задача и клише ответа, который он должен выдать.

    Теперь угол: попробовать выдавать 3 точки: центр и крайнюю точку, но это плохо будет работать, можно попроботвать додумать мб что-то получится
    2-я идея про угол: берем bounding box и внутри него работает методами cv находим угол. 
    '''

    # 3. Get VLM Coordinates
    pred, img = vlm.send_request(prompt, image, "pointing")
    if not pred:
        print("[MAIN] No prediction.")
        return
    print("[MAIN]", pred)

    if(img is not None):
        cam.show_img(img=img)

    data = vlm.safe_parse(pred)
    coords = data[0]
    print(f"coords: {coords}, name: {data[-1]}")

    # data = vlm.get_coordinates(pred)
    # if not data:
    #     print("[MAIN] Failed to parse JSON.")
    #     return
    

    # 4. Create Object & Calculate Image Coords
    target = DetectedObject(
        name=data[-1], 
        vlm_point=Point2D(x=int(coords[0]), y=int(coords[1])),
        angle=0
    )
    target.convert_vlm_to_image(cam.width, cam.height)

    # 5. Get Angle (Requires a second camera capture in your logic)
    angle, img = vlm.get_object_angle(target, cam)
    if target.angle is None:
        return

    print(f"angle = {angle}")

    angle_data = vlm.safe_parse(angle)
    print(angle_data)
    target.angle = angle_data[0][0]

    # 6. Calculate Camera 3D Coords
    target.camera_point = cam.deproject_pixel(
        target.image_point.x, 
        target.image_point.y, 
        depth_image, 
        intrin
    )
    if target.camera_point is None:
        print("[MAIN] Failed to deproject 3D point.")
        return

    # 7. Calculate Robot Position & Move
    print(f"[MAIN] Camera 3D coords: ({target.camera_point.x:.3f}, {target.camera_point.y:.3f}, {target.camera_point.z:.3f})")
    
    move_targets = robot.get_target_positions(target, image)
    print(move_targets)
    if move_targets:
        robot.execute_move(move_targets)

if __name__ == "__main__":
    main()