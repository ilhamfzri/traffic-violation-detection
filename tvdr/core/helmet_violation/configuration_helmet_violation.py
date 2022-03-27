import torch
from typing import List


class HelmetViolationConfig:
    """Helmet Violation Configuration"""

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        imgsz: int = 224,
        vehicle_idx: List[int] = [1, 4],  # Store 2 wheel vehicle class idx
        violation_idx: int = 1,  # Index no helmet model
        conf_thres: int = 0.9,
        min_age: int = 10,
        detect_interval: int = 10,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.vehicle_idx = vehicle_idx
        self.violation_idx = violation_idx
        self.conf_thres = conf_thres
        self.min_age = min_age
        self.detect_interval = detect_interval

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
