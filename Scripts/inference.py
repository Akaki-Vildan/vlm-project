import os
import re
import ast
import math
from typing import Union

import cv2
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


# =============================================================================
# TASK PROMPTS  (each has: TASK body / CLICHE / RULES)
# =============================================================================

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

=== CLICHE (REQUIRED OUTPUT TEMPLATE — MATCH AT LEAST 95%) ===
Single point:    [[x, y, d], "object_name"]
Multiple points: [[[x1, y1, d1], [x2, y2, d2], ...], "object_name"]

Where x, y are integers in [0, 1000], d is a float, "object_name" is a non-empty string.

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list matching the CLICHE.
2. NO explanations, NO markdown (no ```), NO quotes around the whole answer.
3. x, y MUST be integers between 0 and 1000. d MUST be a float.
4. NEVER use placeholders (x, y, ?, ...) or descriptive substitutes ("top-left corner").

EXAMPLES:
Command: "Get ready to pick up the blue cup by its handle"
Answer: [[620, 450, 0.5], "blue cup"]

Command: "Point to the top-left corner of the red box"
Answer: [[300, 200, 0.8], "red box"]

Command: "Show me both wheels of the robot"
Answer: [[[250, 700, 0.6], [750, 700, 0.6]], "robot"]
"""

prompt_trajectory = """You are a trajectory planning AI. Given the operator's command, generate a sequence of waypoints (x, y, depth) describing the path the robot end-effector should follow to accomplish the task.

COORDINATE SYSTEM:
- X: 0 = absolute LEFT, 1000 = absolute RIGHT (center ~500).
- Y: 0 = absolute TOP, 1000 = absolute BOTTOM (center ~500).
- d: depth in meters (float, e.g., 0.5).

=== CLICHE (REQUIRED OUTPUT TEMPLATE — MATCH AT LEAST 95%) ===
[[[x1, y1, d1], [x2, y2, d2], ...], "object_name"]

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list containing a list of waypoints and the object name.
2. NO explanations, NO markdown (no ```), NO quotes around the whole answer.
3. x, y are integers in [0, 1000]. d is a float.
4. At least 2 waypoints are required for a meaningful trajectory.
5. NEVER use placeholders (x, y, ?, ...) or descriptive substitutes ("top-left corner").
"""

prompt_grounding = """You are a spatial grounding AI. Locate the bounding box of the object described by the operator's command.

COORDINATE SYSTEM:
- All coords are integers in [0, 1000] relative scale (NOT raw pixels).
- X: 0 = LEFT, 1000 = RIGHT.
- Y: 0 = TOP, 1000 = BOTTOM.

=== CLICHE (REQUIRED OUTPUT TEMPLATE — MATCH AT LEAST 95%) ===
[[x1, y1, x2, y2], "object_name"]

Where (x1, y1) is the TOP-LEFT corner and (x2, y2) is the BOTTOM-RIGHT corner of the bounding box, and "object_name" is a non-empty string.

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list containing the bounding box and the object name.
2. NO explanations, NO markdown (no ```), NO quotes around the whole answer.
3. NEVER use placeholders (x, y, ?, ...) or descriptive substitutes ("top-left corner").
"""

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

=== CLICHE (REQUIRED OUTPUT TEMPLATE — MATCH AT LEAST 95%) ===
[[[x0, y0, d0], [x1, y1, d1], [x2, y2, d2], [x3, y3, d3], [x4, y4, d4]], "object_name"]

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list matching the CLICHE.
2. NO explanations, NO markdown (no ``` blocks), NO quotes around the whole answer.
3. The structure MUST be a list containing ONE list of 5 points, followed by the object name string.

EXAMPLE:
If a red cup is in the center of the image at 0.4 meters depth:
[[[500, 500, 0.4], [400, 300, 0.4], [600, 300, 0.4], [600, 700, 0.4], [400, 700, 0.4]], "red cup"]
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

=== CLICHE (REQUIRED OUTPUT TEMPLATE — MATCH AT LEAST 95%) ===
[[angle, "object_name"]]

Where angle is a FLOAT in radians (e.g., 1.5708, -0.7854), and "object_name" is a non-empty string.

OUTPUT RULES (STRICTLY ENFORCED):
1. Output ONLY a valid Python list matching the CLICHE.
2. NO markdown formatting, NO text explanations.
3. The angle MUST be a float number. NEVER output strings like "pi/2".

EXAMPLES:
The top edge of a cube points exactly at the "pi/2" line: [[1.5708, "cube"]]
The handle of a cup points exactly at the "-pi/4" line: [[-0.7854, "blue cup"]]
A pen points exactly halfway between "0" and "pi/4": [[0.3927, "pen"]]
"""


# =============================================================================
# GENERIC RULES  (appended to Stage-1 prompt to remind model of constraints)
# =============================================================================

prompt_rules = """
RESPONSE VALIDATION RULES (MANDATORY — APPLY BEFORE RETURNING THE ANSWER):

1. TEMPLATE COMPLIANCE (>= 90% match required):
   - Your output MUST match the CLICHE template specified above by at least 90% in structure, field order, and element types.
   - If your draft does not match, re-reason and reformat it strictly to the CLICHE before returning.

2. NO PLACEHOLDER VARIABLES:
   - Wherever coordinates (x, y) or depth (d) are required, output CONCRETE numeric values ONLY.
   - FORBIDDEN: x, y, (x, y), ?, ..., null, None, "unknown", or any symbolic substitute.
   - x, y MUST be integers in [0, 1000]. d MUST be a float (e.g., 0.5, 1.25).

3. NO DESCRIPTIVE SUBSTITUTES:
   - Never replace coordinates with positional text such as "top-left corner", "center of the image", "bottom-right".
   - Always output exact numeric coordinates in the required format.

4. NUMERIC FORMAT:
   - Coordinates: integers 0-1000 (NOT raw pixels, NOT normalized 0-1 floats).
   - Depth: float in meters (e.g., 0.3, 0.85).
   - Angle: float in radians (e.g., 1.5708). NEVER output strings like "pi/2".

5. OBJECT NAME:
   - MUST be a Python string (e.g., "red cup", "robot").
   - MUST be present in the output. NEVER use "", "object", "unknown", or omit it.

6. STRUCTURE RULES:
   - Output ONLY a valid Python list as specified by the task.
   - NO markdown formatting (no ``` blocks).
   - NO explanations, NO commentary, NO reasoning text.
   - NO quotes around the whole answer.

7. PRE-RETURN SELF-CHECK:
   Before returning, verify ALL of the following:
   [ ] All required fields are present.
   [ ] Field order matches the CLICHE exactly.
   [ ] All coordinates are concrete integers in [0, 1000].
   [ ] Depth is a float; angle (if required) is a float in radians.
   [ ] Object name is a non-empty string.
   [ ] No markdown, no extra text, no placeholders.
   [ ] Overall CLICHE match is >= 90%.
   If ANY check fails, fix the answer and return ONLY the corrected version.

8. FINAL OUTPUT:
   Return ONLY the final valid Python list. No additional text before or after.
"""


# =============================================================================
# STAGE-2 QUALITY-CONTROL PROMPT  (used after Stage 1 to validate / rewrite)
# =============================================================================

prompt_validation = """
=== QUALITY CONTROL STAGE ===
You are a strict output validator. Your job is to verify that the ANSWER below
matches the REQUIRED CLICHE template by at least 95%. You are given the SAME image
that produced the answer, so you can re-check spatial correctness if needed.

REQUIRED CLICHE (the answer MUST conform to this structure):
{cliche}

OPERATOR ANSWER (from Stage 1):
{answer}

=== VALIDATION CHECKLIST (apply EVERY check) ===
[ ] Is it a valid Python list literal? (mentally run ast.literal_eval)
[ ] Does the outer structure match the CLICHE? (list of points + string)
[ ] Are all coordinates present and concrete integers in [0, 1000]?
[ ] Are depths floats (when required by CLICHE)?
[ ] Is the object_name a non-empty string at the end?
[ ] No markdown (no ```), no extra text, no placeholders (x, y, ?, ..., null, "unknown")?
[ ] No descriptive substitutes ("top-left corner" instead of coordinates)?
[ ] Number of points / fields matches the CLICHE exactly?
[ ] Angle (if any) is a float in radians, NOT a string like "pi/2"?

=== DECISION ===
- If ALL checks pass (>= 95% match to CLICHE): return the ANSWER verbatim.
- If ANY check fails: REWRITE the ANSWER to strictly match the CLICHE.
  Use the SAME coordinates / angle / object name as the original answer whenever
  possible. Only invent a plausible value if the original was a placeholder
  (x, y, ?, ..., null) or a descriptive substitute ("top-left corner").

=== OUTPUT ===
Return ONLY the final valid Python list. No explanations, no markdown,
no "Here is the corrected version", no extra text before or after.
"""


# =============================================================================
# INFERENCE CLASS
# =============================================================================

class UnifiedInference:
    """
    A unified class for performing inference using RoboBrain 2.5 models.

    Two-stage cyclic pipeline:
      Stage 1 — Solve the task with TASK + CLICHE + RULES prompt.
      Stage 2 — Quality-control pass: validate the Stage-1 answer against the
                CLICHE; if <95% match, rewrite it to strictly conform.
                Optional extra retries via _looks_valid() Python-side check.
    """

    SUPPORTED_TASKS = (
        "general", "pointing", "trajectory", "grounding", "positioning", "find_angle"
    )

    TASK_PROMPTS = {
        "pointing":    prompt_pointing,
        "trajectory":  prompt_trajectory,
        "grounding":   prompt_grounding,
        "positioning": prompt_positioning,
        "find_angle":  prompt_find_angle,
    }

    TASK_CLICHES = {
        "pointing":    '[[x, y, d], "object_name"]   OR   [[[x1,y1,d1],[x2,y2,d2],...], "object_name"]',
        "trajectory":  '[[[x1,y1,d1],[x2,y2,d2],...], "object_name"]',
        "grounding":   '[[x1, y1, x2, y2], "object_name"]',
        "positioning": '[[[x0,y0,d0],[x1,y1,d1],[x2,y2,d2],[x3,y3,d3],[x4,y4,d4]], "object_name"]',
        "find_angle":  '[[angle_float_radians, "object_name"]]',
    }

    def __init__(self, model_id="BAAI/RoboBrain2.5-8B-NV", device_map="auto"):
        """
        Initialize the model and processor.

        Args:
            model_id (str): Path or Hugging Face model identifier.
            device_map (str): Device mapping strategy ("auto", "cuda:0", etc.).
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

    # ------------------------------------------------------------------ helpers

    def _get_device(self):
        """Return the device the model actually lives on (respects device_map='auto')."""
        try:
            return next(self.model.parameters()).device
        except Exception:
            return "cuda"

    def _generate(self, text_prompt, image, do_sample=True, temperature=0.7):
        """Low-level: run the model on (text, images) and return decoded text.

        Defensive w.r.t. CUDA — uses the model's own device (not hardcoded 'cuda'),
        wraps generation in try/except so a single failure does not abort the whole
        pipeline, and clears the CUDA cache after each call.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    *[
                        {"type": "image",
                         "image": p if p.startswith("http") else f"file://{p}"}
                        for p in image
                    ],
                    {"type": "text", "text": text_prompt},
                ],
            },
        ]

        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        # IMPORTANT: do NOT hardcode "cuda". device_map="auto" may place the
        # model on CPU, MPS, or split across devices. Use the model's own device.
        device = self._get_device()
        try:
            inputs = inputs.to(device)
        except Exception as e:
            print(f"[WARN] Failed to move inputs to {device}, falling back to model placement: {e}")

        print(f"Running inference on device={device} ...")
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=do_sample,
                temperature=temperature,
            )
        except RuntimeError as e:
            print(f"[ERROR] model.generate() failed: {e}")
            # Clear cache and re-raise so the caller can decide what to do
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Free intermediate tensors and clear CUDA cache so the next _generate
        # call starts from a clean state (helps avoid stale CUDA errors).
        del inputs, generated_ids, trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return decoded[0] if decoded else ""

    @staticmethod
    def _strip_markdown(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            s = re.sub(r"^```\w*\n?", "", s)
            s = re.sub(r"\n?```$", "", s)
        return s.strip()

    def _looks_valid(self, text: str, task: str) -> bool:
        """Cheap AST-based sanity check whether the output matches the CLICHE."""
        s = self._strip_markdown(text)
        try:
            val = ast.literal_eval(s)
        except Exception:
            return False
        if not isinstance(val, list) or len(val) != 2:
            return False
        payload, name = val
        if not isinstance(name, str) or not name.strip():
            return False

        if task == "grounding":
            return (isinstance(payload, list) and len(payload) == 4
                    and all(isinstance(v, int) for v in payload))
        if task == "find_angle":
            return isinstance(payload, (int, float))
        if task == "positioning":
            return (isinstance(payload, list) and len(payload) == 5
                    and all(isinstance(p, list) and len(p) == 3 for p in payload))
        # pointing / trajectory
        if not isinstance(payload, list):
            return False
        if payload and isinstance(payload[0], (int, float)):
            # flat single point like [x, y, d]
            return len(payload) == 3 and all(isinstance(v, (int, float)) for v in payload)
        return all(isinstance(p, list) and len(p) == 3 for p in payload)

    # ------------------------------------------------------------------ main API

    def inference(self,
                  text: str,
                  image: Union[list, str],
                  task: str = "general",
                  plot: bool = False,
                  do_sample: bool = True,
                  temperature: float = 0.7,
                  validate: bool = True,
                  max_retries: int = 1):
        """
        Two-stage cyclic inference.

        Stage 1 — Solve the task (TASK + CLICHE + RULES).
        Stage 2 — QC pass: validate the Stage-1 answer against the CLICHE.
                  If _looks_valid() still fails after Stage 2, retry up to
                  `max_retries` additional correction passes.

        Args:
            text (str): The input text prompt.
            image (Union[list,str]): The input image(s) — list of paths or single path.
            task (str): One of SUPPORTED_TASKS.
            plot (bool): Draw results on the image (pointing/trajectory/grounding/positioning).
            do_sample (bool): Whether to use sampling during generation.
            temperature (float): Temperature for sampling.
            validate (bool): If True, run Stage-2 QC pass. Set False to skip and
                             return Stage-1 answer directly.
            max_retries (int): Extra correction passes if Python-side _looks_valid still fails.
        """
        if isinstance(image, str):
            image = [image]

        assert task in self.SUPPORTED_TASKS, \
            f"Invalid task type: {task}. Supported tasks are {self.SUPPORTED_TASKS}."
        assert task == "general" or len(image) == 1, \
            "Pointing/grounding/trajectory/positioning/find_angle tasks require exactly one image."

        # -------- Build Stage-1 prompt --------
        if task == "general":
            stage1_prompt = f"{text}"
        elif task == "grounding":
            stage1_prompt = (
                f"Please provide the bounding box coordinate of the region this sentence describes: {text}.\n\n"
                f"{self.TASK_PROMPTS[task]}\n{prompt_rules}"
            )
        elif task == "trajectory":
            stage1_prompt = (
                f"Task: \"{text}\"\n\n"
                f"{self.TASK_PROMPTS[task]}\n{prompt_rules}"
            )
        else:  # pointing / positioning / find_angle
            stage1_prompt = f"{text}\n\n{self.TASK_PROMPTS[task]}\n{prompt_rules}"

        print(f"\n{'='*20} STAGE 1 — TASK {'='*20}\n{stage1_prompt}\n{'='*52}\n")

        # -------- Pass 1: solve --------
        try:
            raw_answer = self._generate(stage1_prompt, image, do_sample, temperature)
        except RuntimeError as e:
            print(f"[FATAL] Stage 1 generation failed: {e}")
            raise
        print(f"\n[Stage 1 raw answer]\n{raw_answer}\n")

        # -------- Pass 2: QC + retries --------
        if not validate or task == "general":
            final_answer = raw_answer
        else:
            validation_prompt = prompt_validation.format(
                cliche=self.TASK_CLICHES[task],
                answer=raw_answer,
            )
            print(f"\n{'='*20} STAGE 2 — QC {'='*20}\n{validation_prompt}\n{'='*53}\n")
            try:
                final_answer = self._generate(validation_prompt, image, do_sample, temperature)
                print(f"\n[Stage 2 validated answer]\n{final_answer}\n")
            except RuntimeError as e:
                # If Stage 2 fails (CUDA OOM, driver glitch, etc.), fall back to Stage 1.
                print(f"[WARN] Stage 2 generation failed ({e}). Falling back to Stage 1 answer.")
                final_answer = raw_answer

            for attempt in range(max_retries):
                if self._looks_valid(final_answer, task):
                    print(f"[QC] Output passes _looks_valid check.")
                    break
                print(f"[QC] Attempt {attempt+1}: output still invalid — retrying correction pass...")
                retry_prompt = prompt_validation.format(
                    cliche=self.TASK_CLICHES[task],
                    answer=final_answer,
                )
                try:
                    final_answer = self._generate(retry_prompt, image, do_sample, temperature)
                    print(f"\n[Retry {attempt+1} answer]\n{final_answer}\n")
                except RuntimeError as e:
                    print(f"[WARN] Retry {attempt+1} failed ({e}). Stopping retries, keeping current answer.")
                    break

        answer_text = final_answer

        # -------- Plotting (uses validated answer) --------
        img = None
        if plot and task in ["pointing", "trajectory", "grounding", "positioning"]:
            print("Plotting enabled. Drawing results on the image ...")

            plot_points, plot_boxes, plot_trajectories, plot_pos = None, None, None, None
            result_text = answer_text

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

        print("[INFERENSE] end of the function")
        return answer_text, None

    # ------------------------------------------------------------------ drawing

    def draw_on_image(self, image_path, points=None, boxes=None, trajectories=None,
                      positionings=None, output_path=None):
        print("[DRAW_IMAGE] this function works")
        """
        Draw points, bounding boxes, and trajectories on an image.

        Parameters:
            image_path: Path to the input image.
            points: List of points in format [(x, y, d), ...] where x,y are relative (0~1000).
            boxes: List of boxes in format [[x1, y1, x2, y2], ...] where coords are relative (0~1000).
            trajectories: List of trajectories in format [[(x, y, d), ...], ...].
            positionings: Same format as points.
            output_path: Path to save the output image. Default adds "_annotated" suffix.
        """
        try:
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

            if points:
                for point in points:
                    x_rel, y_rel = point[0], point[1]
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # Red solid circle

            if boxes:
                for box in boxes:
                    x1r, y1r, x2r, y2r = box
                    x1, y1 = rel_to_abs(x1r, y1r)
                    x2, y2 = rel_to_abs(x2r, y2r)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

            if trajectories:
                for trajectory in trajectories:
                    if not trajectory or len(trajectory) < 2:
                        continue
                    abs_pts = [rel_to_abs(p[0], p[1]) for p in trajectory]
                    for i in range(1, len(abs_pts)):
                        cv2.line(image, abs_pts[i - 1], abs_pts[i], (0, 255, 0), 2)
                    start_x, start_y = abs_pts[0]
                    cv2.circle(image, (start_x, start_y), 7, (0, 255, 0), -1)
                    end_x, end_y = abs_pts[-1]
                    cv2.circle(image, (end_x, end_y), 7, (255, 0, 0), -1)

            if positionings:
                for point in positionings:
                    x_rel, y_rel = point[0], point[1]
                    x, y = rel_to_abs(x_rel, y_rel)
                    cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

            if not output_path:
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_annotated{ext}"

            print('[INFERENCE] draw image works')
            return image

        except Exception as e:
            print(f"Error processing image: {e}")
            return None