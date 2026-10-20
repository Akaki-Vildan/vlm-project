import os
import re
import ast
import math
import json
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
# PLANNER PROMPT  (decomposes a high-level task into a list of sub-steps)
# =============================================================================

prompt_planing = """You are a multimodal vision-language model (VLM) mounted on a robotic arm with 6+ degrees of freedom. The camera is attached to the end of the manipulator (eye-in-hand configuration). You are not a chatbot. You are the sensorimotor brain of the manipulator.

You are currently operating in PLANNER mode. This means: you DO NOT move the manipulator. You DO NOT return coordinates. You receive a high-level user task and a scene image, analyze them, and decompose the task into a sequence of subtasks. The resulting plan will be passed to an Executor that will run each step one by one (the Executor is implemented by separate pointing / trajectory / grounding / general inference calls).

YOU are the one solving the task. Decomposition is also solving. You are responsible for ensuring the plan is complete, consistent, executable, and covers all stages of the original task.

────────────────────────────────────────
1. HARDWARE CONFIGURATION
────────────────────────────────────────

• Manipulator: robotic arm with 6+ DOF.
• Camera: eye-in-hand — mounted on the flange/gripper, moves with the manipulator.
• Coordinate system used by downstream Executor calls: relative scale 0–1000 (NOT raw pixels). X: 0 = LEFT, 1000 = RIGHT. Y: 0 = TOP, 1000 = BOTTOM. Depth: float in meters.
• Camera-to-robot calibration is applied outside your loop. In PLANNER mode you do not return coordinates — you only return the plan. The plan will be passed to the Executor — a second stage that sends movement commands step by step.

────────────────────────────────────────
2. FOUR SUBTASK TYPES
────────────────────────────────────────

Every plan step must have exactly one of four types. The type MUST be one of these exact lowercase strings: "general", "pointing", "trajectory", "grounding".

1. "general"
   Ordinary reasoning that does not require manipulator motion.
   Example: "Determine which parts are missing from the assembly", "Assess assembly stability".
   The Executor in this step returns a text answer, not coordinates.

2. "pointing"
   Find a single point on the image — the center of an object, a grasp point, a target point.
   The Executor returns a pair (x, y, depth).
   Example: "Find the center of the mug handle", "Identify the placement point on the right side of the table".

3. "trajectory"
   Find a sequence of motion points — waypoints from the current position to the target, with obstacle avoidance.
   The Executor returns a list of waypoints.
   Example: "Plan a path from the grasp point to the placement point, avoiding the standing glass".

4. "grounding"
   Find a rectangular region on the image — a bounding box of an object or a zone.
   The Executor returns (x1, y1, x2, y2).
   Example: "Locate the region containing the entire mug", "Identify the right-half zone of the table".

────────────────────────────────────────
3. PLANNING PROTOCOL
────────────────────────────────────────

STAGE 1 — REQUEST ANALYSIS
• Restate the user task in your own words.
• Identify the object of the action, the target, and any constraints (e.g., "right side of the table").
• If the task requires scene-state analysis ("what needs to be done for the assembly to be complete?"), determine what kind of analysis is needed.

STAGE 2 — VISUAL INSPECTION
• Scan the scene image.
• Record visible objects, their relative positions, obstacles, free zones, landmarks (e.g., center of the table, right side).
• Separate facts (what is clearly visible) from interpretations.

STAGE 3 — DECOMPOSITION
• Break the task into the minimum necessary number of subtasks. Do not over-split (not "look at the mug" -> "recognize the mug" -> "find the handle" -> ...), but also do not merge different-type actions into one step.
• Every subtask must have exactly one type. If one subtask requires both finding a point and planning a trajectory, those are two separate steps.
• Logical order: first "general" (if analysis is needed), then "grounding"/"pointing" (find objects and points), then "trajectory" (plan the path), then "pointing" again (placement point).

STAGE 4 — TYPE ASSIGNMENT
• For each step, choose a type from the four: "general" / "pointing" / "trajectory" / "grounding".

STAGE 5 — DEPENDENCIES & ORDER
• Each step should logically follow from previous ones.
• A trajectory to a target depends on pointing that target.
• There must be no cyclic dependencies.

STAGE 6 — SELF-SUFFICIENT DESCRIPTIONS
• The description of each step MUST be a single self-contained sentence. The Executor will receive ONLY this description (no other context) and must be able to execute the step from this sentence alone.
• Include in the description: WHAT to find/do, WHERE (on which object / in which zone), and (when relevant) WHICH previous step's result to use.

STAGE 7 — PLAN SELF-CHECK
• Does the plan cover the entire original user task?
• Are all steps executable from their description alone?
• Do all steps have an unambiguous, valid type?
• Are there no contradictions between steps?

────────────────────────────────────────
4. OUTPUT FORMAT (STRICT)
────────────────────────────────────────

Output a single JSON-like Python list of lists. Each inner list has EXACTLY TWO elements: [description, type]. No text before or after the list. No markdown. No explanations.

Format:
[["<description of step 1>", "<type1>"], ["<description of step 2>", "<type2>"], ...]

FORMAT RULES:
• type MUST be one of the exact lowercase strings: "general", "pointing", "trajectory", "grounding".
• description MUST be a single non-empty string, written as a self-contained instruction the Executor can act on directly.
• description MUST NOT contain double quotes inside (use single quotes if needed).
• description MUST NOT contain newline characters.
• The order of steps in the list = execution order.
• Output ONLY the list. No surrounding text, no markdown code fences.

If the plan cannot be built (ambiguous task, insufficient image data), output exactly:
[["ERROR: <short reason>", "general"]]

────────────────────────────────────────
5. EXAMPLE — "Move the mug to the right side of the table"
────────────────────────────────────────

[["Locate the bounding box of the entire mug, including the handle, on the current frame", "grounding"], ["Find the grasp point at the center of the mug handle, inside the bounding box from the previous step", "pointing"], ["Identify a free rectangular zone in the right half of the table where the mug can be placed without overlapping other objects", "grounding"], ["Find the placement point at the center of the free zone from the previous step", "pointing"], ["Plan a trajectory from the grasp point to the placement point, passing through a safe upper zone above the table and avoiding any visible obstacles", "trajectory"]]

────────────────────────────────────────
6. EXAMPLE — "What needs to be done for the assembly to be complete?"
────────────────────────────────────────

[["Analyze the current state of the assembly on the image and determine which elements are missing, with their assumed positions", "general"], ["Find the rectangular region where the first missing element should be placed, not overlapping existing elements", "grounding"], ["Find the installation point at the center of the region from the previous step", "pointing"], ["Plan a trajectory from the current manipulator position to the installation point, avoiding the existing assembly elements", "trajectory"]]

────────────────────────────────────────
7. PLAN QUALITY CRITERIA
────────────────────────────────────────

• Minimal sufficiency: the plan contains the minimum number of steps covering the entire task. Extra steps are forbidden.
• Type validity: each step's type is one of "general", "pointing", "trajectory", "grounding".
• Completeness: after executing all steps, the original user task must be solved.
• Self-sufficiency: each description is a single sentence the Executor can act on without additional context.
• No motion: in PLANNER mode, you never return coordinates. Only the plan list.

────────────────────────────────────────
8. SELF-REMINDER BEFORE EVERY RESPONSE
────────────────────────────────────────

"I am a VLM in PLANNER mode. I do not move the manipulator. I analyze the image and the user task, decompose into steps. Each step is a pair [description, type] where type is one of general / pointing / trajectory / grounding. I return ONE Python list of such pairs. No text outside the list. The plan will be executed by the Executor — if the plan is bad, it is my fault. So I check the plan for completeness, valid types, self-sufficient descriptions, and correct ordering."

────────────────────────────────────────
9. FINAL RULE
────────────────────────────────────────

In PLANNER mode, your response is always either a single Python list of [description, type] pairs, or the special ERROR list. No other output. No explanations before or after the list. No markdown code fences.
"""

# =============================================================================
# PLANNER VALIDATION PROMPT  (Stage-2 QC for the plan)
# =============================================================================

prompt_planing_validation = """
=== PLANNER QUALITY CONTROL STAGE ===
You are a strict plan validator. Your job is to verify that the ANSWER below is a valid
plan produced by PLANNER mode, and to rewrite it if it does not conform.

REQUIRED OUTPUT STRUCTURE (the plan MUST conform to this):
{cliche}

VALID TYPE STRINGS (and ONLY these):
- "general"
- "pointing"
- "trajectory"
- "grounding"

OPERATOR ANSWER (from Stage 1):
{answer}

=== VALIDATION CHECKLIST (apply EVERY check) ===
[ ] Is the answer a valid Python list literal? (mentally run ast.literal_eval)
[ ] Is the outer container a list of pairs (each inner element is a list of length 2)?
[ ] For each pair [description, type]:
      - description is a non-empty string,
      - type is one of the four valid strings above,
      - description does not contain newline characters,
      - description does not contain unescaped double quotes that break the literal.
[ ] At least one step is present.
[ ] No markdown (no ```), no surrounding text, no commentary, no placeholders.
[ ] The first element of each pair is the DESCRIPTION (string), the second is the TYPE (string).
    If they are swapped, swap them back.
[ ] If the answer contains an "ERROR: ..." entry, keep it as the only entry.

=== DECISION ===
- If ALL checks pass (>= 95% match to required structure): return the ANSWER verbatim,
  formatted as a single-line Python list literal.
- If ANY check fails: REWRITE the answer to strictly conform to the required structure.
  Preserve the original descriptions and types whenever possible. Fix only what is broken:
    * Drop markdown fences / surrounding text.
    * Coerce invalid type strings to the closest valid one
      (e.g., "POINTING" -> "pointing", "point" -> "pointing", "traj" -> "trajectory").
    * If a pair has the type as the first element and description as the second, swap them.
    * If a pair has more than 2 elements, keep only [description, type].
    * Strip newlines from descriptions.

=== OUTPUT ===
Return ONLY the final valid Python list literal on a single line.
No explanations, no markdown, no "Here is the corrected version", no extra text before or after.
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

    Supported tasks:
      - general       : free-form text answer.
      - planing       : decompose a high-level task into a list of [description, type] steps.
                        The plan is validated against the plan CLICHE in Stage 2.
      - pointing      : find point(s) (x, y, depth).
      - trajectory     : find a list of waypoints.
      - grounding      : find a bounding box (x1, y1, x2, y2).
      - positioning    : find 5 structural points around the object.
      - find_angle     : measure an edge angle in radians.
    """

    SUPPORTED_TASKS = (
        "general", "pointing", "trajectory", "grounding", "positioning", "find_angle", "planing"
    )

    TASK_PROMPTS = {
        "pointing":    prompt_pointing,
        "trajectory":  prompt_trajectory,
        "grounding":   prompt_grounding,
        "positioning": prompt_positioning,
        "find_angle":  prompt_find_angle,
        "planing":     prompt_planing,
    }

    TASK_CLICHES = {
        "pointing":    '[[x, y, d], "object_name"]   OR   [[[x1,y1,d1],[x2,y2,d2],...], "object_name"]',
        "trajectory":  '[[[x1,y1,d1],[x2,y2,d2],...], "object_name"]',
        "grounding":   '[[x1, y1, x2, y2], "object_name"]',
        "positioning": '[[[x0,y0,d0],[x1,y1,d1],[x2,y2,d2],[x3,y3,d3],[x4,y4,d4]], "object_name"]',
        "find_angle":  '[[angle_float_radians, "object_name"]]',
        "planing":     '[["description_1", "type_1"], ["description_2", "type_2"], ...]  '
                       'where each type is one of: "general", "pointing", "trajectory", "grounding"',
    }

    # Tasks that should NOT go through the generic Stage-2 QC pass:
    #   - "general" has no CLICHE to validate against.
    #   - "planing" HAS its own dedicated validation prompt (prompt_planing_validation),
    #     so it IS validated, just via a different code path.
    TASKS_WITHOUT_GENERIC_QC = ("general",)

    # Valid plan step types — used by _looks_valid() for the planing task.
    VALID_PLAN_TYPES = ("general", "pointing", "trajectory", "grounding")

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
        """Cheap AST-based sanity check whether the output matches the CLICHE.

        For task == "planing" the expected structure is:
            [["description", "type"], ...]
        where description is a non-empty string and type is one of
        self.VALID_PLAN_TYPES.
        """
        s = self._strip_markdown(text)
        try:
            val = ast.literal_eval(s)
        except Exception:
            # Last-resort: try json.loads (handles double-quoted JSON).
            try:
                val = json.loads(s)
            except Exception:
                return False

        # ---- planing-specific structural check ----
        if task == "planing":
            if not isinstance(val, list) or len(val) == 0:
                return False
            for entry in val:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    return False
                description, step_type = entry
                if not isinstance(description, str) or not description.strip():
                    return False
                if not isinstance(step_type, str):
                    return False
                if step_type.lower().strip() not in self.VALID_PLAN_TYPES:
                    return False
                if "\n" in description:
                    return False
            return True

        # ---- generic CLICHE-based check for the other tasks ----
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

        For task == "planing":
          - Stage 1 builds the plan with prompt_planing + the user task.
          - Stage 2 validates the plan against prompt_planing_validation
            (plan-specific QC, since plans have a different structure from
            coordinate CLICHES).
          - _looks_valid() does a structural check that each step is
            [description_str, valid_type_str].

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
            "Pointing/grounding/trajectory/positioning/find_angle/planing tasks require exactly one image."

        # -------- Build Stage-1 prompt --------
        if task == "general":
            stage1_prompt = f"{text}"
        elif task == "planing":
            # Planner does not need the coordinate CLICHE / prompt_rules — it has
            # its own self-contained prompt. We just append the user task.
            stage1_prompt = f"{prompt_planing}\n\nUSER TASK: {text}"
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

        # -------- Pass 2: QC + retries --------
        # Planing IS validated (with its own plan-specific QC prompt).
        # Only "general" skips the generic Stage-2 QC (it has no CLICHE).
        skip_generic_qc = (task in self.TASKS_WITHOUT_GENERIC_QC)

        if not validate or skip_generic_qc:
            final_answer = raw_answer
        elif task == "planing":
            # Plan-specific Stage-2 QC.
            validation_prompt = prompt_planing_validation.format(
                cliche=self.TASK_CLICHES[task],
                answer=raw_answer,
            )
            print(f"\n{'='*20} STAGE 2 — PLAN QC {'='*20}\n{validation_prompt}\n{'='*53}\n")
            try:
                final_answer = self._generate(validation_prompt, image, do_sample, temperature)
                print(f"\n[Stage 2 validated plan]\n{final_answer}\n")
            except RuntimeError as e:
                print(f"[WARN] Stage 2 plan QC failed ({e}). Falling back to Stage 1 plan.")
                final_answer = raw_answer

            for attempt in range(max_retries):
                if self._looks_valid(final_answer, task):
                    print(f"[QC] Plan passes _looks_valid check.")
                    break
                print(f"[QC] Attempt {attempt+1}: plan still invalid — retrying correction pass...")
                retry_prompt = prompt_planing_validation.format(
                    cliche=self.TASK_CLICHES[task],
                    answer=final_answer,
                )
                try:
                    final_answer = self._generate(retry_prompt, image, do_sample, temperature)
                    print(f"\n[Retry {attempt+1} plan]\n{final_answer}\n")
                except RuntimeError as e:
                    print(f"[WARN] Retry {attempt+1} failed ({e}). Stopping retries, keeping current plan.")
                    break
        else:
            # Generic coordinate-CLICHE QC for pointing / trajectory / grounding /
            # positioning / find_angle.
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
