import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from pipeline import PickAndPlacePipeline


def main():
    pipeline = PickAndPlacePipeline()
    try:
        pipeline.setup()
        # Plan mode: operator enters high-level task → captures REFERENCE + CURRENT
        # → VLM planner returns a flat list of trajectory descriptions → each
        # description is sent in turn to run_trajectory (which captures a fresh
        # frame, plans the trajectory and moves the robot). On the first failed
        # step, plan execution aborts.
        pipeline.run_plan()
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()