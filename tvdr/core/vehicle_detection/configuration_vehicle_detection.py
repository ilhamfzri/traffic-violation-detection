import torch
from typing import List


class VehicleDetectionConfig:
    r"""
    Vehicle Detection Configuration
    """

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        imgsz: int = 640,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        classes_idx: List[int] = [0, 1, 2, 3, 4],
        classes_names: List[str] = ["Car", "Motorcycle", "Bus", "Truck", "Bycycle"],
        max_detection: int = 500,
        detection_area: List[List[float]] = [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ],
        tracker: str = "SORT",
        **kwargs,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes_idx = classes_idx
        self.classes_names = classes_names
        self.max_detection = max_detection
        self.tracker = tracker
        self.detection_area = detection_area

        if tracker == "SORT":
            self.init_sort()

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def init_sort(
        self,
        sort_max_age: int = 80,
        sort_min_hits: int = 3,
        sort_iou_thres: float = 0.3,
        **kwargs,
    ):
        self.sort_max_age = sort_max_age
        self.sort_min_hits = sort_min_hits
        self.sort_iou_thres = sort_iou_thres
