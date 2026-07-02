import pyrealsense2 as rs
import numpy as np
import cv2
from utils import Point3D

class RealSenseCamera:

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()
        
        # ADD self. HERE
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        self.align = rs.align(rs.stream.color)
        self.profile = None
        self.depth_scale = 0.0

    def start(self):
        """Call this to actually turn on the hardware"""
        # ADD self. HERE
        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        
        print("Warming up camera...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        print("Camera ready!\n")

    def capture_frame(self):
        """Returns color_image, depth_image, intrinsics. Returns None if quit."""
        print("Streaming... Press 'p' to capture, or 'q' to quit.")
        while True:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            intrin = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imshow('Simple Stream', color_image)
            cv2.imshow('Depth Stream', depth_colormap)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.pipeline.stop()
                cv2.destroyAllWindows()
                return color_image.copy(), depth_image.copy(), intrin
            elif key == ord('q'):
                self.pipeline.stop()
                cv2.destroyAllWindows()
                return None, None, None
            

    def deproject_pixel(self, px, py, depth_image, intrin):
        """Deprojects a single pixel using median depth. Returns Point3D"""
        half = 5
        y0, y1 = max(0, py - half), min(self.height, py + half + 1)
        x0, x1 = max(0, px - half), min(self.width, px + half + 1)
        
        window = depth_image[y0:y1, x0:x1].astype(np.float32) * self.depth_scale
        valid = window[window > 0]
        
        if valid.size == 0:
            print("[CAMERA] depth is None (no valid depth in window)")
            return None
            
        depth = float(np.median(valid))
        point_3d = rs.rs2_deproject_pixel_to_point(intrin, [px, py], depth)
        return Point3D(point_3d[0], point_3d[1], point_3d[2])



