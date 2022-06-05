import cv2
import os

from pathlib import Path
from ..vehicle_detection.main_vehicle_detection import VehicleDetection
from ..running_red_light.main_running_redlight import RunningRedLight
from ..helmet_violation.main_helmet_violation import HelmetViolation
from ..wrongway_detection.main_wrongway_detection import WrongWayDetection
from .configuration_pipeline import PipelineConfig
from ...utils.annotator import Annotator


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = PipelineConfig()
        self.vd = VehicleDetection(self.config.vd_config)
        self.rrl = RunningRedLight(self.config.rrl_config)
        self.hv = HelmetViolation(self.config.hv_config)
        self.ww = WrongWayDetection(self.config.ww_config)
        self.annotator = Annotator(self.config)

    
    def video_recorder_init(self, video_path):
        if os.path.isdir("result")== False:
            os.mkdir("result")

        vid = cv2.VideoCapture(video_path)
        _, frame = vid.read()
        fps = vid.get(cv2.CAP_PROP_FPS)
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

        filename = os.path.basename(video_path).rsplit('.', 1)[0]
        index = 0
        
        while(1):
            current_output_path = os.path.join("result", f"{filename}_{index}.mp4")
            if os.path.exists(current_output_path) == False:
                break
            index +=1

        self.video_output = current_output_path
        self.video_writer = cv2.VideoWriter(
            self.video_output,
            fourcc=fourcc,
            fps=fps,
            frameSize=(width, height),
        )
    
    def video_recorder_update(self, frame):
        self.video_writer.write(frame)

    def video_recorder_close(self):
        self.video_writer.release()
        return self.video_output
    
    def reload_config(self):
        self.vd.config = self.config.vd_config
        self.rrl.config = self.config.rrl_config
        self.hv.config = self.config.hv_config
        self.ww.config = self.config.ww_config
        self.annotator.config = self.config
        self.annotator.vd_config = self.config.vd_config
        self.annotator.rrl_config = self.config.rrl_config


    def update(self, img):
        vd_result = self.vd.update(img)
        rrl_result = self.rrl.update(img, vd_result)
        hv_result = self.hv.update(img, vd_result)
        direction, ww_result = self.ww.update(vd_result)

        img_new = self.annotator.vehicle_detection(img, vd_result)
        img_new = self.annotator.running_red_light(img_new, vd_result, rrl_result, self.rrl.state)
        img_new = self.annotator.helmet_violation(img_new, vd_result, hv_result)
        img_new = self.annotator.wrongway_detection(img_new, vd_result, direction, ww_result)
        
        return img_new

    def ready_check(self):
        # Ready Check Vehicle Detection and Tracker
        self.vd_ready_status = self.vd.ready_check()
        self.rrl_ready_status = self.rrl.ready_check()
        self.hv_ready_status = self.hv.ready_check()
        self.ww_ready_status = self.ww.ready_check()

        for key in ["vd","rrl", "hv", "ww"]:
            status  = getattr(self, f"{key}_ready_status")
            if status == False:
                return False
        return True
