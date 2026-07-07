import os, re, cv2
from typing import Union
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
import math


prompt_pointing = """You are a precision spatial targeting AI. Your task is to analyze the operator's command, identify the exact interaction point on the target object in the image, and provide its coordinates.

TASK BREAKDOWN:
1. UNDERSTAND THE COMMAND: Read what the operator wants to do (e.g., "grasp the handle", "touch the top-left corner", "press the center").
2. LOCATE THE POINT: Find the exact pixel location on the image that corresponds to the action described in the command.
3. ESTIMATE DEPTH: Provide a rough estimate of the distance to that specific point in meters (e.g., 0.5 for half a meter).

COORDINATE SYSTEM RULES (CRITICAL):
You MUST output coordinates in a relative scale from 0 to 1000. Do NOT output raw pixels.
- X-axis: 0 is the absolute LEFT edge of the image, 1000 is the absolute RIGHT edge. (The center is ~500).
- Y-axis: 0 is the absolute TOP edge of the image, 1000 is the absolute BOTTOM edge. (The center is ~500).
- Depth (d): The distance to the object in meters (e.g., 0.3, 0.85).

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list containing tuples and the object name.
2. NO explanations, NO markdown (no ```), NO quotes around the whole answer.
3. Format: [((x1, y1, d1), (x2, y2, d2), ...), "object_name"]
4. x, y MUST be integers between 0 and 1000. d MUST be a float.

EXAMPLES:
Command: "Get ready to pick up the blue cup by its handle"
Answer: [((620, 450, 0.5),), "blue cup"]

Command: "Point to the top-left corner of the red box"
Answer: [((300, 200, 0.8),), "red box"]

Command: "Show me both wheels of the robot"
Answer: [((250, 700, 0.6), (750, 700, 0.6)), "robot"]
"""

prompt_trajectory = """OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list containing a list of waypoints and the object name.
2. NO explanations, NO markdown (no ```), NO quotes around the whole answer.
3. Format: [[(x1, y1, d1), (x2, y2, d2), ...], "object_name"]
4. x, y are coordinates (0~1000), d is depth."""

prompt_grounding = """OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list containing the bounding box and the object name.
2. NO explanations, NO markdown (no ```), NO quotes around the whole answer.
3. Format: [[[x1, y1, x2, y2]], "object_name"]
4. x1, y1 are top-left coords, x2, y2 are bottom-right coords (0~1000)."""

prompt_positioning = """You are a precision spatial analysis AI. Your task is to locate the target object in the image and extract exactly 5 key structural points around it, plus its name.

POINT MAPPING RULES (STRICT ORDER):
You MUST output exactly 5 points in this specific order:
- Point 0: The exact GEOMETRIC CENTER of the object.
- Point 1: The TOP-LEFT corner of the object's bounding area.
- Point 2: The TOP-RIGHT corner of the object's bounding area.
- Point 3: The BOTTOM-RIGHT corner of the object's bounding area.
- Point 4: The BOTTOM-LEFT corner of the object's bounding area.

COORDINATE SYSTEM RULES (CRITICAL):
- X-axis: 0 is the absolute LEFT edge of the image, 1000 is the absolute RIGHT edge.
- Y-axis: 0 is the absolute TOP edge of the image, 1000 is the absolute BOTTOM edge.
- Depth (d): The approximate distance to the object in meters (e.g., 0.5).
- x, y MUST be integers between 0 and 1000. d MUST be a float.
- name MUST be string and without it program can't work

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list.
2. NO explanations, NO markdown (no ``` blocks), NO quotes around the whole answer.
3. The structure MUST be a list containing ONE tuple of 5 points, followed by the object name string.
4. Format: [((x0, y0, d0), (x1, y1, d1), (x2, y2, d2), (x3, y3, d3), (x4, y4, d4)), "object_name"]

EXAMPLE:
If a red cup is in the center of the image at 0.4 meters depth:
[((500, 500, 0.4), (400, 300, 0.4), (600, 300, 0.4), (600, 700, 0.4), (400, 700, 0.4)), "red cup"]
"""

prompt_find_angle = """You are a spatial analysis AI. Your task is to measure the angle of a specific edge of the target object using the visual protractor on the image.

VISUAL ELEMENTS:
- RED DOT: The center.
- WHITE CIRCLE & LINES: A protractor. The line pointing straight DOWN is labeled "0".

INSTRUCTIONS:
1. Find the RED DOT.
2. IDENTIFY AN EDGE: Look at the target object. Pick ONE clear, straight edge or continuous line on it.
   - For a square/cube: pick the top edge, a side edge, or a visible diagonal.
   - For a pen/marker: pick its long body.
   - For a cup: pick the handle or a vertical side rim.
3. EXTEND THE EDGE: Mentally extend that chosen edge in a straight line through the RED DOT until it reaches the protractor circle.
4. READ THE PROTRACTOR: Find the white line (and its text label) that is closest to where your extended edge points.
5. CALCULATE DEVIATION: The "0" line points DOWN. Calculate the angle of your edge based on its deviation from this "0" line.

MAPPING LABELS TO NUMBERS (Translate the text you see into these exact floats):
- "0" -> 0.0
- "pi/4" -> 0.7854
- "pi/2" -> 1.5708
- "3pi/4" -> 2.3562
- "pi" -> 3.1416
- "-3pi/4" -> -2.3562
- "-pi/2" -> -1.5708
- "-pi/4" -> -0.7854

ESTIMATION:
If the edge points exactly between two labels, average their numbers. (e.g., exactly between "0" and "pi/4" is 0.3927).

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list.
2. NO markdown formatting, NO text explanations.
3. The angle MUST be a float number. NEVER output strings like "pi/2".
4. Format: [[angle, "object_name"]]

EXAMPLES:
The top edge of a cube points exactly at the "pi/2" line: [[1.5708, "cube"]]
The handle of a cup points exactly at the "-pi/4" line: [[-0.7854, "blue cup"]]
A pen points exactly halfway between "0" and "pi/4": [[0.3927, "pen"]]
"""



class UnifiedInference:
    """
    A unified class for performing inference using RoboBrain 2.5 models.
    """
    
    def __init__(self, model_id="BAAI/RoboBrain2.5-8B-NV", device_map="auto"):
        """
        Initialize the model and processor.
        
        Args:
            model_id (str): Path or Hugging Face model identifier
            device_map (str): Device mapping strategy ("auto", "cuda:0", etc.)
        """
        print("Loading Checkpoint ...")
        self.model_id = model_id
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            dtype="auto", 
            device_map=device_map,
            local_files_only=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
    def inference(self, text: str, image: Union[list, str], task="general", 
                 plot=False, do_sample=True, temperature=0.7):
        """
        Perform inference with text and images input.
        
        Args:
            text (str): The input text prompt.
            image (Union[list,str]): The input image(s) as a list of file paths or a single file path.
            task (str): The task type, e.g., "general", "pointing", "trajectory", "grounding".
            plot (bool): Whether to plot results on image.
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
        """

        if isinstance(image, str):
            image = [image]

        assert task in ["general", "pointing", "trajectory", "grounding", "positioning", "find_angle"], \
            f"Invalid task type: {task}. Supported tasks are 'general', 'pointing', 'trajectory', 'grounding', 'positioning'."
        assert task == "general" or (task in ["pointing", "trajectory", "grounding", "positioning", "find_angle"] and len(image) == 1), \
            "Pointing, grounding, and trajectory tasks require exactly one image."

        if task == "pointing":
            print("Pointing task detected. Adding pointing prompt.")
            text = f"{text}\n\n{prompt_pointing}"
        elif task == "trajectory":
            print("Trajectory task detected. Adding trajectory prompt.")
            text = f"Task: \"{text}\"\n\n{prompt_trajectory}"
        elif task == "grounding":
            print("Grounding task detected. Adding grounding prompt.")
            text = f"Please provide the bounding box coordinate of the region this sentence describes: {text}."
        elif task == "positioning":
            print("Positioning task detected. Adding positioning prompt.")
            text = f"{text}\n\n{prompt_positioning}"
        elif task == "find_angle":
            print("Angle finding task detected. Adding angle finding prompt.")
            text = f"{text}\n\n{prompt_find_angle}"

        print(f"\n{'='*20} INPUT {'='*20}\n{text}\n{'='*47}\n")

        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image", 
                         "image": path if path.startswith("http") else f"file://{path}"
                        } for path in image
                    ],
                    {"type": "text", "text": f"{text}"},
                ],
            },
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        print("Running inference ...")
        generated_ids = self.model.generate(**inputs, max_new_tokens=768, do_sample=do_sample, temperature=temperature)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        answer_text = output_text[0] if output_text else ""

        img = None

        # Plotting functionality
        if plot and task in ["pointing", "trajectory", "grounding", "positioning"]:
            print("Plotting enabled. Drawing results on the image ...")
            
            plot_points, plot_boxes, plot_trajectories, plot_pos = None, None, None, None
            result_text = answer_text  # Use the processed answer text for plotting
            
            if task == "trajectory":
                trajectory_pattern = r'(\d+),\s*(\d+),\s*([+-]?\d+\.\d+)'
                trajectory_points = re.findall(trajectory_pattern, result_text)
                plot_trajectories = [[(int(x), int(y), float(d)) for x, y, d in trajectory_points]]
                print(f"Extracted trajectory points: {plot_trajectories}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_trajectory_annotated.")
            elif task == "pointing":
                point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*([+-]?\d+\.\d+)\s*\)'
                points = re.findall(point_pattern, result_text)
                plot_points = [(int(x), int(y), float(d)) for x, y, d in points]
                print(f"Extracted points: {plot_points}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_pointing_annotated.")
            elif task == "grounding":
                box_pattern = r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'
                boxes = re.findall(box_pattern, result_text)
                plot_boxes = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2 in boxes]
                print(f"Extracted bounding boxes: {plot_boxes}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_grounding_annotated.")
            elif task == "positioning":
                point_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*,\s*([+-]?\d+\.\d+)\s*\)'
                positionings = re.findall(point_pattern, result_text)
                plot_pos = [(int(x), int(y), float(d)) for x, y, d in positionings]
                print(f"Extracted positionings: {plot_pos}")
                image_name_to_save = os.path.basename(image[0]).replace(".", "_with_positioning_annotated.")


            os.makedirs("result", exist_ok=True)
            image_path_to_save = os.path.join("result", image_name_to_save)

            img = self.draw_on_image(
                image[0], 
                points=plot_points, 
                boxes=plot_boxes, 
                trajectories=plot_trajectories,
                positionings=plot_pos,
                output_path=image_path_to_save
            )
            print("[DRAW] try to start draw_on_image")
            return answer_text, img

        # Return unified format
        print("[INFERENSE] end of the function")
        return answer_text, None

    def draw_on_image(self, image_path, points=None, boxes=None, trajectories=None, positionings=None, output_path=None):
        print("[DRAW_IMAGE] this fucntion works")
        """
        Draw points, bounding boxes, and trajectories on an image

        Parameters:
            image_path: Path to the input image
            points: List of points in format [(x, y), ...] where x,y are relative (0~1000)
            boxes: List of boxes in format [[x1, y1, x2, y2], ...] where coords are relative (0~1000)
            trajectories: List of trajectories in format [[(x, y), (x, y), ...], ...]
                        or [[(x, y, d), ...], ...] where x,y are relative (0~1000)
            output_path: Path to save the output image. Default adds "_annotated" suffix to input path
        """
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Unable to read image: {image_path}")

            h, w = image.shape[:2]

            def rel_to_abs(x_rel, y_rel):
                """Convert relative (0~1000) to absolute pixel coords, clamped to image bounds."""
                x = int(round((x_rel / 1000.0) * w))
                y = int(round((y_rel / 1000.0) * h))
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                return x, y

            # Draw points
            if points:
                for point in points:
                    x_rel, y_rel = point
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle

            # Draw bounding boxes
            if boxes:
                for box in boxes:
                    x1r, y1r, x2r, y2r = box
                    x1, y1 = rel_to_abs(x1r, y1r)
                    x2, y2 = rel_to_abs(x2r, y2r)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            # Draw trajectories
            if trajectories:
                for trajectory in trajectories:
                    if not trajectory or len(trajectory) < 2:
                        continue

                    # Convert all trajectory points to absolute pixels
                    abs_pts = []
                    for p in trajectory:
                        # support (x,y) or (x,y,d)
                        x_rel, y_rel = p[0], p[1]
                        abs_pts.append(rel_to_abs(x_rel, y_rel))

                    # Connect trajectory points with lines
                    for i in range(1, len(abs_pts)):
                        cv2.line(image, abs_pts[i - 1], abs_pts[i], (0, 255, 0), 2)  # Blue line

                    # Draw a larger point at the trajectory end
                    start_x, start_y = abs_pts[0]
                    cv2.circle(image, (start_x, start_y), 7, (0, 255, 0), -1)  # Red start point

                    # Draw a larger point at the trajectory end
                    end_x, end_y = abs_pts[-1]
                    cv2.circle(image, (end_x, end_y), 7, (255, 0, 0), -1)  # Blue end point
                    
            if positionings:
                for point in positionings:
                    x_rel, y_rel = point
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle


            # Determine output path
            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_annotated{ext}"

            #Save the result
            #cv2.imwrite('~/home/vildan/projects/vml-first-try/RoboBrain2.5/Scripts/captured_photos/result.jpg', image)
            print('[INFERENCE] draw image works')
            # print(f"Annotated image saved to: {output_path}")
            # return output_path
            return image

        except Exception as e:
            print(f"Error processing image: {e}")
            return None


