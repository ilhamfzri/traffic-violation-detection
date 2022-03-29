class WrongWayDetectionConfig:
    def __init__(
        self,
        direction_violation: int = 0,
        direction_violation_thr: int = 10,
        min_sigma_dy_dx_violation: int = 60,
        removal_miss_count: int = 300,
        **kwargs
    ):
        self.direction_violation = direction_violation
        self.direction_violation_thr = direction_violation_thr
        self.min_sigma_dy_dx_violation = min_sigma_dy_dx_violation
        self.removal_miss_count = removal_miss_count
