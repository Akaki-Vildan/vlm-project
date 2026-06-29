from inference import UnifiedInference
import pyrealsense2 as rs
import numpy as np
import cv2
import textwrap
import tempfile
import os
import ast

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

LOCAL_MODEL_PATH = "/home/vildan/projects/vml-first-try/RoboBrain2.5/models"

if not os.path.exists(os.path.join(LOCAL_MODEL_PATH, "config.json")):
    print("ОШИБКА: config.json не найден в папке!")
    exit()

print("[STATUS] Loading local model weights...")
model = UnifiedInference(LOCAL_MODEL_PATH)


def send_a_request(prompt_text, images_data, task):
    print("[STATUS] Sending request to model... This might take a moment.")
    temp_file_path = None
    
    if isinstance(images_data, (list, tuple)) and len(images_data) == 2:
        color_img, depth_img = images_data
        combined_image = np.hstack((color_img, depth_img))
        image_to_save = combined_image
    elif isinstance(images_data, np.ndarray):
        image_to_save = images_data
    else:
        image_to_send = images_data
        image_to_save = None

    try:
        if image_to_save is not None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            cv2.imwrite(temp_file_path, image_to_save)
            image_to_send = temp_file_path

        pred = model.inference(prompt_text, image_to_send, task=task)
        return pred
    finally:
        if temp_file_path is not None and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def start_stream(pipeline):
    print("Streaming... Press 'p' to capture, or 'q' to quit.")
    
    align = rs.align(rs.stream.color)

    while True:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        combined_display = np.hstack((color_image, depth_colormap))
        cv2.imshow('Color (Left) + Depth (Right)', combined_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            prompt = input("Write a prompt: ")
            # ИЗМЕНЕНО: Возвращаем depth_frame и intrinsics для вычисления координат
            color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            return color_image, depth_colormap, prompt, depth_frame, color_intrinsics
            
        elif key == ord('q'):
            print("Cancelled.")
            return None, None, None, None, None


def process_model_output_and_get_3d(model_output_string, depth_frame, intrinsics):
    # 1. Parse the string
    try:
        parsed_data = ast.literal_eval(model_output_string)
    except (ValueError, SyntaxError):
        print("Failed to parse model output!")
        return None
        
    # 2. Extract x, y (assuming the model returned at least one prediction)
    # Format is usually: [(x, y, confidence), ...]
    if len(parsed_data) > 0:
        first_prediction = parsed_data[0]
        
        x = int(first_prediction[0])
        y = int(first_prediction[1])
        # confidence = first_prediction[2] # You probably don't need this for coords
        
        # 3. Call your coordinate conversion function
        coords_meters = convert_to_cord(x, y, depth_frame, intrinsics)
        return coords_meters
    else:
        print("Model returned empty list!")
        return None

def convert_to_cord(x, y, depth_frame, intrinsics):
    """
    Переводит пиксели (x, y) в 3D координаты [X, Y, Z] в метрах.
    x - координата по горизонтали (u)
    y - координата по вертикали (v)
    depth_frame - выровненный кадр глубины из RealSense
    intrinsics - параметры RGB-камеры (т.к. глубина выровнена под RGB)
    """
    # Получаем дистанцию в метрах прямо из RealSense SDK
    depth_in_meters = depth_frame.get_distance(x, y)
    
    # Если камера не видит эту точку (блик, слишком близко/далеко), возвращаем нули
    if depth_in_meters <= 0:
        print(f"Внимание: нет данных глубины для пикселя ({x}, {y})")
        return [0.0, 0.0, 0.0]
        
    # Конвертируем пиксель + глубину в 3D координаты
    point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth_in_meters)
    
    X = point_3d[0]
    Y = point_3d[1]
    Z = point_3d[2] # То же самое, что depth_in_meters
    
    return [X, Y, Z]


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    pipeline.start(config)

    print("Warming up camera (please wait 1 second)...")
    for _ in range(30):
        pipeline.wait_for_frames()
    print("Camera ready!\n")

    try:
        # ИСПОЛЬЗУЕМ КАМЕРУ ВМЕСТО КАРТИНОК ИЗ ФАЙЛОВ
        # start_stream теперь возвращает depth_frame и intrinsics
        color_img, depth_colormap, prompt, depth_frame, intrinsics = start_stream(pipeline)
        
        if color_img is None:
            print("Захват отменен.")
            return

        taks_ar = ['general', 'pointing', 'trajectory', 'grounding']
        pred = send_a_request(prompt, [color_img, depth_colormap], taks_ar[1])
        
        # ПРИМЕР: Как использовать convert_to_cord после получения координат от модели
        # Допустим, модель вернула координаты пикселя x=300, y=200
        
        print(pred)
        #coords_meters = process_model_output_and_get_3d(pred, depth_frame, intrinsics)
        coords_meters = convert_to_cord(int(pred)[0][0], int(pred)[0][1], depth_frame, intrinsics)
        print(f"Пиксель ({pred[0]}, {pred[1]}) -> Координаты в метрах: X={coords_meters[0]:.3f}, Y={coords_meters[1]:.3f}, Z={coords_meters[2]:.3f}")
        
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()