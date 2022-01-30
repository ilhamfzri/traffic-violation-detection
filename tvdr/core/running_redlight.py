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

import numpy as np
import cv2

from tvdr.utils.params import Parameter
from tvdr.core.algorithm import convert_bbox_to_polygon, is_line_intersection_polygon


class RunningRedLightViolationDetection:
    def __init__(self, parameter: Parameter):
        self.update_params(parameter)

    def traffic_light_state_recognition(self, img):

        # Set red light threshold parameter
        self.red_min = np.array(
            [
                self.red_light_params["h_min"],
                self.red_light_params["s_min"],
                self.red_light_params["v_min"],
            ]
        )

        self.red_max = np.array(
            [
                self.red_light_params["h_max"],
                self.red_light_params["s_max"],
                self.red_light_params["v_max"],
            ]
        )

        # Set green light threshold parameter
        self.green_min = np.array(
            [
                self.green_light_params["h_min"],
                self.green_light_params["s_min"],
                self.green_light_params["v_min"],
            ]
        )

        self.green_max = np.array(
            [
                self.green_light_params["h_max"],
                self.green_light_params["s_max"],
                self.green_light_params["v_max"],
            ]
        )

        # traffic light frame
        pos = self.traffic_light_area
        img_tl = img[pos[1] : pos[3], pos[0] : pos[2]]
        img_tl = np.ascontiguousarray(img_tl)

        # convert image from rgb to hsv colorspace
        image_csv = cv2.cvtColor(img_tl, cv2.COLOR_BGR2HSV)

        # segmentation colors of image based on hsv min and hsv max values
        self.image_red_light = cv2.inRange(image_csv, self.red_min, self.red_max)
        self.image_green_light = cv2.inRange(image_csv, self.green_min, self.green_max)

        # count total pixel each traffic light colors
        self.red_light_count = np.sum(self.image_red_light == 255)
        self.green_light_count = np.sum(self.image_green_light == 255)

        # return traffic light state based on number of pixel threshold
        if self.red_light_count >= self.red_light_params["threshold"]:
            self.last_state = "RED"
            return "RED"
        elif self.green_light_count >= self.green_light_params["threshold"]:
            self.last_state = "GREEN"
            return "GREEN"
        else:
            return self.last_state

    def update(self, img, result):
        running_redlight_result = []

        self.state = self.traffic_light_state_recognition(img)
        if self.state == "RED":
            for object in result:
                object_id = object[6]
                bbox = object[0:4]
                polygon = convert_bbox_to_polygon(bbox)
                if is_line_intersection_polygon(self.stop_line, polygon):
                    running_redlight_result.append(object_id)

        return running_redlight_result

    def update_params(self, parameter: Parameter):
        self.paramater = parameter
        self.red_light_params = parameter.traffic_light_red_light
        self.green_light_params = parameter.traffic_light_green_light
        self.traffic_light_area = parameter.traffic_light_area
        self.stop_line = parameter.stopline
        self.last_state = "RED"
