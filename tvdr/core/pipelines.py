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

from unittest import result
from tvdr.core import running_redlight
from tvdr.utils.params import Parameter
from tvdr.core.vehicle_detection import VehicleDetection
from tvdr.core.helmet_violation_detection import HelmetViolationDetection
from tvdr.core.violation_recorder_class import ViolationRecorderMain
from tvdr.core.wrong_way import WrongWayViolationDetection
from tvdr.core.running_redlight import RunningRedLightViolationDetection

import time

PROFILING_PROCESS = True


class TrafficViolationDetectionPipelines:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter

        # Initialize vehicle detection
        self.vd = VehicleDetection(self.parameter)

        # Initilalize helmet violation detection
        if self.parameter.detect_helmet_violation:
            self.hvd = HelmetViolationDetection(self.parameter)

        # Initilalize wrongway violation detection
        if self.parameter.detect_wrongway_violation:
            self.wvd = WrongWayViolationDetection(self.parameter)

        # Initializer running redlight violation detection
        if self.parameter.detect_running_redlight_violation:
            self.rrvd = RunningRedLightViolationDetection(self.parameter)

        # Initialize violation recorder
        self.vr = ViolationRecorderMain(self.parameter)
        # self.update_parameter(parameter)

    def update(self, image):

        # Check if model vehicle detection is not loaded then load
        if self.vd.model_loaded == False:
            self.vd.load_model()

        # Check if helmet violation detection is not loaded then load
        if self.hvd.model_loaded == False:
            self.hvd.load_model()

        if PROFILING_PROCESS:
            checkpoint1 = time.time()

        # Result vehicle detection
        print("Process Result Vehicle Detection")
        result_vd = self.vd.predict(image)

        if PROFILING_PROCESS:
            checkpoint2 = time.time()

        # Update helmet violation detection
        print("Preocess Helmet Detection")

        if self.parameter.detect_helmet_violation:
            result_hvd = self.hvd.update(image, result_vd)

        if PROFILING_PROCESS:
            checkpoint3 = time.time()

        if self.parameter.detect_wrongway_violation:
            result_wvd, direction_data = self.wvd.update(result_vd)

        if PROFILING_PROCESS:
            checkpoint4 = time.time()

        if self.parameter.detect_running_redlight_violation:
            result_rrvd = self.rrvd.update(image, result_vd)

        if PROFILING_PROCESS:
            checkpoint5 = time.time()

        if PROFILING_PROCESS:
            os.system("cls" if os.name == "nt" else "clear")
            print(f"Task    \t\t\t\t| Time (ms)\t|")
            print(
                f"Vehicle Detection \t\t\t| {(checkpoint2-checkpoint1)*1000:.1f}\t\t|"
            )
            print(
                f"Helmet Violation Detection \t\t| {(checkpoint3-checkpoint2)*1000:.1f}\t\t|"
            )
            print(
                f"Wrongway Violation Detection \t\t| {(checkpoint4-checkpoint3)*1000:.1f}\t\t|"
            )
            print(
                f"Running Redlight Violation Detection \t| {(checkpoint5-checkpoint4)*1000:.1f}\t\t|"
            )

        # For troubleshoting
        if len(result_hvd) > 0:
            print(f"Helmet Violation : :{result_hvd}")

        if len(result_wvd) > 0:
            print(f"Wrongway Violation : {result_wvd}")

        if len(result_rrvd) > 0:
            print(f"Running Red Light Violation : {result_rrvd}")

        result_final = self.vr.detection_combiner(
            vehicle_detection_result=result_vd,
            direction_data=direction_data,
            helmet_violation_result=result_hvd,
            wrongway_violation_result=result_wvd,
            running_redlight_result=result_rrvd,
        )

        image_out = self.vr.annotate_result(image, result_final)
        return image_out

    def update_parameter(self, parameter: Parameter):
        self.parameter = parameter
        self.vd.load_parameters(self.parameter)

        print("Reload Model")
        self.vd.load_model()
        self.vr.update_params(parameter)

        if self.parameter.detect_helmet_violation:
            self.hvd.load_parameters(self.parameter)
            self.hvd.load_model()

        if self.parameter.detect_wrongway_violation:
            self.wvd.update_params(parameter)

        if self.parameter.detect_running_redlight_violation:
            self.rrvd.update_params(parameter)

    def get_traffic_light_state(self):
        return self.rrvd.state
