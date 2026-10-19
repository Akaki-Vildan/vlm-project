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

@dataclass
class Area2D:
    first_point: Point2D = None
    second_point : Point2D = None


class DetectedObject:
    def __init__(self, name: str, vlm_point: Point2D, angle: float, area=None, img=None, countor=None):
        self.name = name
        self.vlm_point = vlm_point
        self.angle = angle
        self.image = img
        self.area = area
        self.countor = countor
        
        self.image_point = Point2D()
        self.camera_point = Point3D()
        self.robot_point = Point3D()

    def convert_vlm_to_image(self, camera_width: int, camera_height: int, vlm_space: float = 1000.0, points=None):
        """Scales VLM coordinates (1000x1000) to camera pixel coordinates"""
        res = []
        if(points == None):
            self.image_point.x = int(round(self.vlm_point.x / vlm_space * camera_width))
            self.image_point.y = int(round(self.vlm_point.y / vlm_space * camera_height))
        else:
            x, y = points
            res.append(int(round(x / vlm_space * camera_width)))
            res.append(int(round(y / vlm_space * camera_height)))
            return res

    def create_area(self, camera_width: int, camera_height: int, points):
        '''
        Get's a points and convert it into area in target
        '''
        x1, y1, x2, y2 = points
        first = self.convert_vlm_to_image(camera_width=camera_width, camera_height=camera_height, points=(int(x1), int(y1)))
        second = self.convert_vlm_to_image(camera_width=camera_width, camera_height=camera_height, points=(int(x2), int(y2)))

        first_point = Point2D(x = int(first[0]), y = int(first[1]))
        second_point = Point2D(x = int(second[0]), y = int(second[1]))

        res = Area2D(first_point=first_point, second_point=second_point)

        self.area = res
        
