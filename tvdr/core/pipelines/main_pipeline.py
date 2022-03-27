from ..vehicle_detection.main_vehicle_detection import VehicleDetection
from .configuration_pipeline import PipelineConfig


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = PipelineConfig()
        self.vd = VehicleDetection(self.config.vd_config)

    def reload_config(self):
        self.vd.config = self.config.vd_config

    def ready_check(self):
        # Ready Check Vehicle Detection and Tracker
        self.vd_ready_status = self.vd.ready_check()
        if self.vd_ready_status == False:
            return False

        return True
