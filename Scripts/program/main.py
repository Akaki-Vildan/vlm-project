import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from pipeline import PickAndPlacePipeline


def main():
    pipeline = PickAndPlacePipeline()
    try:
        pipeline.setup()
        pipeline.run_trajectory()  # вместо pipeline.run()
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    main()