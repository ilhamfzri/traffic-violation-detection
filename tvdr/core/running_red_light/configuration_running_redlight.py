from typing import List


class RunningRedLightConfig:
    """Running Red Light Configuration"""

    def __init__(
        self,
        traffic_light_area: List[List[float]] = [
            [0.4, 0.4],
            [0.6, 0.4],
            [0.6, 0.6],
            [0.4, 0.6],
        ],
        green_hsv_min: List[int] = [36, 25, 25],
        green_hsv_max: List[int] = [70, 255, 255],
        green_min_area: int = 10,
        red_hsv_min: List[int] = [0, 70, 50],
        red_hsv_max: List[int] = [10, 255, 255],
        red_min_area: int = 10,
        yellow_hsv_min: List[int] = [20, 100, 100],
        yellow_hsv_max: List[int] = [30, 255, 255],
        yellow_min_area: int = 10,
        stop_line: List[List[float]] = [[0.1, 0.2], [0.3, 0.4]],
        set_unknown_state: str = "yellow",  # red, yellow, green, or None
    ):
        self.traffic_light_area = traffic_light_area

        self.green_hsv_min = green_hsv_min
        self.green_hsv_max = green_hsv_max
        self.green_min_area = green_min_area

        self.red_hsv_min = red_hsv_min
        self.red_hsv_max = red_hsv_max
        self.red_min_area = red_min_area

        self.yellow_hsv_min = yellow_hsv_min
        self.yellow_hsv_max = yellow_hsv_max
        self.yellow_min_area = yellow_min_area

        self.stop_line = stop_line
        self.set_unknown_state = set_unknown_state
