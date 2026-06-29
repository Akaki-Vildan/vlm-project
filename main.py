from Scripts.inference import UnifiedInference
import pyrealsense2 as rs
import numpy as np
import cv2
import textwrap
import tempfile
import os

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


# ИЗМЕНЕНО: Теперь принимает СПИСОК изображений (color + depth)
def send_a_request(prompt_text, images_data):
    print("[STATUS] Sending request to model... This might take a moment.")

    task = int(input('Which task u want to do: 0-general 1-trajectory 2-pointing 3-grounding 4-positioning: '))
    plot_inf = int(input("Are u need a drawing result? 0 - no, 1 - yes: "))
    sample_inf = int(input("Are u want to do sample? 0 - no, 1 - yes: "))
    temperature = float(input("White temperature u want: "))

    if(plot_inf == 1):
        plot = True
    else:
        plot = False

    if(sample_inf == 1):
        do_sample = True
    else:
        do_sample = False


    taks_ar = ['general', 'trajectory', 'pointing', 'grounding', "positioning"]
    
    temp_file_path = None
    
    # 1. STITCH IMAGES TOGETHER IF THERE ARE MULTIPLE
    if isinstance(images_data, (list, tuple)) and len(images_data) == 2:
        color_img, depth_img = images_data
        
        # Stitch them horizontally (side-by-side)
        # Color is on the left, Depth is on the right
        combined_image = np.hstack((color_img, depth_img))
        image_to_save = combined_image
        
    elif isinstance(images_data, np.ndarray):
        # If only one image is passed, just use it
        image_to_save = images_data
    else:
        # If it's a string path
        image_to_send = images_data
        image_to_save = None

    # 2. SAVE TO TEMP FILE
    try:
        if image_to_save is not None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            temp_file_path = temp_file.name
            temp_file.close()
            cv2.imwrite(temp_file_path, image_to_save)
            image_to_send = temp_file_path

        # 3. SEND TO MODEL
        image_to_send = "Scripts/captured_photos/new image_1.png"
        pred, image = model.inference(prompt_text, image_to_send, task=taks_ar[task], plot=plot, do_sample=do_sample, temperature=temperature)
        print(f"\n=== Prediction ===\n{pred}\n==================\n")
    finally:
        # 4. CLEAN UP
        if temp_file_path is not None and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
    return pred, image


def start_stream(pipeline):
    print("Streaming... Press 'p' to capture, or 'q' to quit.")
    
    # Create alignment object (align depth to color)
    align = rs.align(rs.stream.color)

    while True:
        frames = pipeline.wait_for_frames(timeout_ms=1000)
        
        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame() # Get aligned depth
        
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # CONVERT DEPTH TO COLOR MAP so the VLM can understand it
        # alpha=0.03 adjusts the depth sensitivity (change if depth is too dark/bright)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Show both images stitched together in the OpenCV window
        combined_display = np.hstack((color_image, depth_colormap))
        cv2.imshow('Color (Left) + Depth (Right)', combined_display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            prompt = input("Write a prompt: ")
            #prompt = 'i need to have a black cube bounding box'
            # Return BOTH images
            return color_image, depth_colormap, prompt
            
        elif key == ord('q'):
            print("Cancelled.")
            return None, None, None


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    # ИЗМЕНЕНО: Включаем поток глубины
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
    pipeline.start(config)

    print("Warming up camera (please wait 1 second)...")
    for _ in range(30):
        pipeline.wait_for_frames()
    print("Camera ready!\n")

    try:
        color_img, depth_img, prompt = start_stream(pipeline)
        #prompt = 'i need to pick at first black cube and place it into right side then pick up white cube and place it into left side'
        
        if color_img is not None and prompt is not None:
            # ИЗМЕНЕНО: Отправляем ОБА изображения в модель
            # The prompt should tell the model that it's looking at two images side-by-side
            pred, image = send_a_request(prompt, color_img)
            print(type(image))
            cv2.imshow("Result", image)
            cv2.waitKey(0) # Wait for a key press to close the window
            cv2.destroyAllWindows() 
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()