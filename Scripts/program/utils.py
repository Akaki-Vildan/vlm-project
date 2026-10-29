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
    second_point: Point2D = None


class DetectedObject:
    """Найденный на сцене объект.

    Хранит представление координат во всех системах отсчёта, через которые
    проходитpipeline: VLM (0..1000) -> image (px) -> camera 3D (m) -> robot base (m),
    а также bbox (Area2D), контур (cv2 contour) и угол поворота.

    Все мутабельные операции над координатами сосредоточены здесь, чтобы
    внешний код не дёргал внутренние поля вручную.
    """

    # VLM использует относительную шкалу 0..1000, не сырые пиксели.
    VLM_SPACE: float = 1000.0

    def __init__(self, name: str = "", vlm_point: Point2D = None, angle: float = 0.0,
                 area: Area2D = None, image=None, contour=None):
        self.name = name
        self.vlm_point = vlm_point if vlm_point is not None else Point2D()
        self.angle = angle
        self.image = image
        self.area = area
        self.contour = contour

        # Производные поля, заполняются в процессе работы.
        self.image_point = Point2D()
        self.camera_point = Point3D()
        self.robot_point = Point3D()

    # ------------------------------------------------------------------
    # Координатные преобразования
    # ------------------------------------------------------------------
    def convert_vlm_to_image(self, camera_width: int, camera_height: int,
                             points=None, vlm_space: float = None):
        """Масштабирует координаты VLM (0..vlm_space) в пиксели камеры.

        - Если points=None: обновляет self.image_point по self.vlm_point.
        - Если points=(x, y): возвращает [px, py] без изменения состояния
          (используется, например, при расчёте bbox).
        """
        vlm_space = vlm_space or self.VLM_SPACE
        if points is None:
            self.image_point.x = int(round(self.vlm_point.x / vlm_space * camera_width))
            self.image_point.y = int(round(self.vlm_point.y / vlm_space * camera_height))
            return None

        x, y = points
        return [
            int(round(x / vlm_space * camera_width)),
            int(round(y / vlm_space * camera_height)),
        ]

    def create_area(self, camera_width: int, camera_height: int, points):
        """Создаёт Area2D (bbox) из VLM-координат вида [x1, y1, x2, y2].

        Записывает результат в self.area и возвращает его же.
        """
        x1, y1, x2, y2 = points
        first = self.convert_vlm_to_image(
            camera_width, camera_height, points=(int(x1), int(y1))
        )
        second = self.convert_vlm_to_image(
            camera_width, camera_height, points=(int(x2), int(y2))
        )
        self.area = Area2D(
            first_point=Point2D(x=first[0], y=first[1]),
            second_point=Point2D(x=second[0], y=second[1]),
        )
        return self.area

    def update_vlm_point(self, x, y, camera_width=None, camera_height=None):
        """Обновляет vlm_point новыми координатами и (опционально) сразу же
        пересчитывает image_point под текущие размеры кадра.
        """
        self.vlm_point.x = x
        self.vlm_point.y = y
        if camera_width is not None and camera_height is not None:
            self.convert_vlm_to_image(camera_width, camera_height)
        return self

    def set_image_point(self, x, y):
        """Прямая установка пиксельных координат (в обход VLM-масштаба)."""
        self.image_point.x = int(x)
        self.image_point.y = int(y)
        return self

    def is_resolved(self) -> bool:
        """True, если для объекта уже известны все нужные координаты."""
        return (
            self.image_point is not None
            and self.camera_point is not None
            and self.camera_point.x != 0.0
            and self.camera_point.y != 0.0
            and self.camera_point.z != 0.0
        )