from dataclasses import dataclass

@dataclass
class Point2D:
    x: float = 0.0
    y: float = 0.0

@dataclass
class Point3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

class DetectedObject:
    def __init__(self, name: str, vlm_point: Point2D, angle: float, img=None):
        self.name = name
        self.vlm_point = vlm_point
        self.angle = angle
        self.image = img
        
        self.image_point = Point2D()
        self.camera_point = Point3D()
        self.robot_point = Point3D()

    def convert_vlm_to_image(self, camera_width: int, camera_height: int, vlm_space: float = 1000.0):
        """Scales VLM coordinates (1000x1000) to camera pixel coordinates"""
        self.image_point.x = int(round(self.vlm_point.x / vlm_space * camera_width))
        self.image_point.y = int(round(self.vlm_point.y / vlm_space * camera_height))