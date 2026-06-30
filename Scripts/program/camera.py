import pyrealsense2 as rs
import numpy as np
import cv2
from main import objects

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Пространство координат, в котором VLM (RoboBrain pointing) выдаёт пиксели.
# По логам кадр у тебя 1000x1000 (678/1.5625 == 678/1000*640).
VLM_SPACE = 1000.0

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)

align = rs.align(rs.stream.color)  # выравниваем Depth к Color
profile = pipeline.start(config)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

# --- Прогрев камеры ---
print("Warming up camera (please wait ~1 second)...")
for _ in range(30):
    pipeline.wait_for_frames()
print("Camera ready!\n")


def start_stream():
    """
    Поток с камеры. По 'p' захватывает кадр, по 'q' выходит.

    Возвращает (color_image, prompt, depth_image, depth_scale, intrin), где
    depth_image — numpy z16 (выровнен к color), intrin — интринсики кадра.
    Возвращаем данные (а не rs.frame), чтобы они оставались валидными после
    pipeline.stop() — на время медленного инференса VLM.
    """
    print("Streaming... Press 'p' to capture, or 'q' to quit.")
    while True:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow('Simple Stream', color_image)
        cv2.imshow('Depth Stream', depth_colormap)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            #prompt = "get center of the black little cube"
            pipeline.stop()
            cv2.destroyAllWindows()
            # .copy() — отвязываемся от внутреннего буфера RealSense
            return color_image.copy(), depth_image.copy(), depth_scale, intrin

        elif key == ord('q'):
            print("Cancelled.")
            pipeline.stop()
            cv2.destroyAllWindows()
            return None, None, None, None, None


def pixel_to_camera_3d(depth_image, depth_scale, intrin, color_image):
    """
    Депроецирует пиксель (в системе VLM, 1000x1000) в 3D точку (м) в оптическом
    фрейме камеры. Медиана глубины по окну для устойчивости к шуму/нулям.
    """


    # Медиана глубины по окну 11x11, игнорируя нули
    half = 5
    y0, y1 = max(0, objects[0].coords_image["y"] - half), min(CAMERA_HEIGHT, objects[0].coords_image["y"] + half + 1)
    x0, x1 = max(0, objects[0].coords_image["x"] - half), min(CAMERA_WIDTH, objects[0].coords_image["x"] + half + 1)
    window = depth_image[y0:y1, x0:x1].astype(np.float32) * depth_scale
    valid = window[window > 0]
    if valid.size == 0:
        print("[CAMERA] depth is None (no valid depth in window)")
        return None
    depth = float(np.median(valid))

    print(f"[CAMERA] point in 3D: ({objects[0].coords_image["x"]}, {objects[0].coords_image["y"]}, {depth:.4f})")

    # Депроекция: пиксель + глубина -> 3D точка в оптическом фрейме камеры
    point_3d = rs.rs2_deproject_pixel_to_point(intrin, [objects[0].coords_image["x"], objects[0].coords_image["y"]], depth)
    res = list(point_3d)
    objects[0].coords_camera["x"] = res[0]
    objects[0].coords_camera["y"] = res[1]
    objects[0].coords_camera["z"] = res[2]
    return 
