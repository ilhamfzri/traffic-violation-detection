#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 UGM

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Ilham Fazri - ilhamfazri3rd@gmail.com
import os
import sys
import logging
import cv2

from unittest import result
from tvdr.core import running_redlight
from tvdr.utils.params import Parameter
from tvdr.core.vehicle_detection import VehicleDetection
from tvdr.core.helmet_violation_detection import HelmetViolationDetection
from tvdr.core.helmet_violation_classifier import HelmetViolationDetectionClassifier
from tvdr.core.violation_recorder_class import ViolationRecorderMain
from tvdr.core.wrong_way import WrongWayViolationDetection
from tvdr.core.running_redlight import RunningRedLightViolationDetection

import time

PROFILING_PROCESS = True


logger = logging.getLogger("tvdr")


class TrafficViolationDetectionPipelines:
    """Traffic Violation Detection Pipelines"""

    def __init__(self, parameter: Parameter):
        self.parameter = parameter

        # Initialize Vehicle Detection
        logging.info("Initializing Vehicle Detection..")
        self.vd = VehicleDetection(self.parameter)

        # Initilalize helmet violation detection
        logging.info("Initializing Helmet Violation Detection..")
        if self.parameter.detect_helmet_violation:
            self.hvd = HelmetViolationDetectionClassifier(self.parameter)

        # Initilalize wrongway violation detection
        logging.info("Initializing WrongWay Violation Detection..")
        if self.parameter.detect_wrongway_violation:
            self.wvd = WrongWayViolationDetection(self.parameter)

        # Initializer running redlight violation detection
        logging.info("Initializing Running Red Light Detection..")
        if self.parameter.detect_running_redlight_violation:
            self.rrvd = RunningRedLightViolationDetection(self.parameter)

        # Initialize violation recorder
        logging.info("Initializing Violation Recorder..")
        self.vr = ViolationRecorderMain(self.parameter)

    def update(self, image, frame_idx):
        logger.debug(f"\n\n")

        check_pipeline = time.time()

        if self.vd.model_loaded == False:
            self.vd.load_model()

        # Vehicle Detection Process
        checkpoint = time.time()
        result_vd = self.vd.predict(image)
        logger.debug(
            f"Vehicle Detection Process Time : {(time.time()-checkpoint)*1000:.1f}ms"
        )

        # Helmet Violation Detection Process
        if self.parameter.detect_helmet_violation:
            checkpoint = time.time()
            result_hvd = self.hvd.update(image, result_vd)
            logger.debug(
                f"Helmet Violation Detection Process Time : {(time.time()-checkpoint)*1000:.1f}ms"
            )

        # Wrong-way Violation Detection Process
        if self.parameter.detect_wrongway_violation:
            checkpoint = time.time()
            result_wvd, direction_data = self.wvd.update(result_vd)
            logger.debug(
                f"Wrong-way Violaton Detection Process Time : {(time.time()-checkpoint)*1000:.1f}ms"
            )

        # Running Red Light Violation Detection Process
        if self.parameter.detect_running_redlight_violation:
            checkpoint = time.time()
            result_rrvd = self.rrvd.update(image, result_vd)
            state_tl = self.rrvd.state.title()
            logger.debug(
                f"Running Red Light Process Time : {(time.time()-checkpoint)*1000:.1f}ms"
            )

        # Print Violation Detected (For troubleshoot purpose)
        if result_hvd is not None and len(result_hvd) > 0:
            logger.debug(f"Helmet Violation : {result_hvd}")

        if result_wvd is not None and len(result_wvd) > 0:
            logger.debug(f"Helmet Violation : {result_wvd}")

        if result_rrvd is not None and len(result_rrvd) > 0:
            logger.debug(f"Running Red Light Violationn : {result_rrvd}")

        # Combine result of each downstream task to make the array result consistent

        checkpoint = time.time()
        result_combine = self.vr.detection_combiner(
            vehicle_detection_result=result_vd,
            direction_data=direction_data,
            helmet_violation_result=result_hvd,
            wrongway_violation_result=result_wvd,
            running_redlight_result=result_rrvd,
        )
        logger.debug(f"Combiner Process Time : {(time.time()-checkpoint)*1000:.1f}ms")

        # Create annotate of image
        checkpoint = time.time()
        image_out = self.vr.annotate_result(
            image,
            result_combine,
            traffic_light=state_tl if state_tl is not None else None,
        )
        logger.debug(f"Annotate Process Time : {(time.time()-checkpoint)*1000:.1f}ms")

        checkpoint = time.time()
        # self.vr.video_recorder_update(image_out)
        logger.debug(
            f"Video Recorder Process Time : {(time.time()-checkpoint)*1000:.1f}ms"
        )

        # if self.parameter.detect_running_redlight_violation:
        #     if len(result_rrvd) > 0:
        #         self.vr.write_violation_running_red_light(
        #             result_combine, image, frame_idx
        #         )

        # if self.parameter.detect_wrongway_violation:
        #     if len(result_wvd) > 0:
        #         self.vr.write_violation_wrongway(result_combine, image, frame_idx)

        logger.debug(
            f"Total Processing Time :  {(time.time()-check_pipeline)*1000:.1f}ms"
        )

        return image_out

    def update_parameter(self, parameter: Parameter):
        self.parameter = parameter
        self.vd.load_parameters(self.parameter)

        print("Reload Model")
        self.vd.load_model()
        self.vr.update_params(parameter)

        if self.parameter.detect_helmet_violation:
            self.hvd.update_params(parameter)

        if self.parameter.detect_wrongway_violation:
            self.wvd.update_params(parameter)

        if self.parameter.detect_running_redlight_violation:
            self.rrvd.update_params(parameter)

        self.vr.database_writer_init("result")

    def reset_state(self):
        self.vd.reset_state()
        self.hvd.reset_state()
        self.wvd.reset_state()
        self.rrvd = RunningRedLightViolationDetection(self.parameter)
        self.vr = ViolationRecorderMain(self.parameter)

    def video_recorder_init(self, video_path, output_path):
        vid = cv2.VideoCapture(video_path)
        _, frame = vid.read()
        fps = vid.get(cv2.CAP_PROP_FPS)
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc=fourcc,
            fps=fps,
            frameSize=(width, height),
        )

    def video_recorder_update(self, frame):
        self.video_writer.write(frame)

    def video_recorder_close(self):
        self.video_writer.release()

    def get_traffic_light_state(self):
        return self.rrvd.state
