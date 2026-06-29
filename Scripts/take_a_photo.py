import pyrealsense2 as rs
import numpy as np
import cv2
import os  # Import os to handle directory creation and paths

# Define camera constants
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Define the directory where you want to save the photos
# (This creates a folder called "captured_photos" in the same directory as this script)
# To use an absolute path (Windows), you can write: save_directory = r"C:\Users\Name\Pictures"
# To use an absolute path (Linux/Mac), you can write: save_directory = "/home/name/Pictures"
save_directory = "./captured_photos"

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
# ИЗМЕНЕНО: Включаем поток глубины
config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)
pipeline.start(config)

# Create the alignment object (must be done after pipeline.start)
align = rs.align(rs.stream.color)

print("Warming up camera (please wait 1 second)...")
for _ in range(30):
    pipeline.wait_for_frames()
print("Camera ready! Press 'p' to take photo, 'q' to quit.\n")

try:
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

        # Show the live video feed
        cv2.imshow("video", color_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            # Create the directory if it doesn't exist
            os.makedirs(save_directory, exist_ok=True)
            
            index = int(input("0 - correct image 1 - current image"))

            # Build the full file path
            filepath = os.path.join(save_directory, f"new image_{index}.png")
            
            # Save the image to the new directory
            cv2.imwrite(filepath, color_image)
            print(f"Photo saved to: {filepath}")
            break
            
        elif key == ord('q'):
            print("Cancelled.")
            break

finally:
    # Stop the pipeline and close windows cleanly
    pipeline.stop()
    cv2.destroyAllWindows()