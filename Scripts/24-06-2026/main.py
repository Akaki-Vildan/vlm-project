import sys
import os

# Путь к родительской директории (папка Scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import camera
import utils
import robot


def main():
    pred = None
    image = None
    depth_image = depth_scale = intrin = None

    try:
        image, prompt, depth_image, depth_scale, intrin = camera.start_stream()
        if image is not None and prompt is not None:
            pred = utils.send_a_request(prompt, image)
    except Exception as e:
        print(f"[MAIN] Error during stream/request: {e}")
        import traceback
        traceback.print_exc()

    if pred is not None:
        print(f"[MAIN] pred: {pred}")
        
        robot_position = utils.get_coords_for_robot(
            pred, 
            image, 
            depth_image, 
            depth_scale, 
            intrin)
        
        if robot_position is not None:
            print(f"Final point: {robot_position}")
            robot.move_robot_to_pos(robot_position)
            
    else:
        print("[MAIN] No prediction available, skipping robot command.")


if __name__ == "__main__":
    main()
