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

import time
from enum import EnumMeta, auto
from re import L
from typing import List, Text
from urllib.request import parse_http_list
from PIL import ImageDraw
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import append
from tvdr.core import sort
from tvdr.core import wrong_way
from tvdr.core.violation_recorder import ViolationRecorder
from tvdr.core.wrong_way import WrongWayViolationDetection
from tvdr.utils.params import Parameter
from tvdr.core.deepsort import DeepSort
from tvdr.core.sort import Sort
from tvdr.core.algorithm import (
    detection_area_filter,
    detection_running_redlight,
    calculate_center_of_box,
    cart2pol,
    pol2cart,
)
from tvdr.utils.general import (
    make_divisible,
    xyxy2xywh,
    check_img_size,
    letterbox,
    combine_yolo_deepsort_result,
    combine_yolo_sort_result,
    sort_validity,
)

import cv2
import torch
import numpy as np


class YOLOInference:
    def __init__(self, parameter: Parameter):

        super().__init__()
        self.parameter = parameter
        self.device = self.select_device(self.parameter.device)
        self.update_model_params = True

        # DeepSORT Initialize
        self.deepsort = DeepSort(
            self.parameter.deepsort_model_path,
            self.parameter.deepsort_max_dist,
            self.parameter.deepsort_min_confidence,
            self.parameter.deepsort_max_iou_distance,
            self.parameter.deepsort_max_age,
            self.parameter.deepsort_n_init,
            self.parameter.deepsort_nn_budget,
            self.parameter.deepsort_use_cuda,
        )

        # WrongWay Detection  Initialize
        self.wrongway = WrongWayViolationDetection(self.parameter)

        # SORT Initialize
        self.sort = Sort(
            self.parameter.sort_max_age,
            self.parameter.sort_min_hits,
            self.parameter.sort_iou_threshold,
        )

        self.imgsz = check_img_size(
            imgsz=self.parameter.yolo_imgsz, s=self.parameter.yolo_stride
        )

        self.parameter.output_dir = "outputs"
        self.violation_recorder = ViolationRecorder(self.parameter)

    def load_model(self, model_type: Text = "yolov5s"):

        try:
            self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

            return True
        except:
            return False

    def inference_frame(
        self, frame_data: np.ndarray, timestamp_video: int, status: str
    ):

        # print(status)

        # Update Model Params
        if self.update_model_params == True:
            self.model.conf = self.parameter.yolo_conf
            self.model.iou = self.parameter.yolo_iou
            self.model.classes = self.parameter.yolo_classes
            self.model.multi_label = self.parameter.yolo_multi_label
            self.model.max_det = self.parameter.yolo_max_detection
            self.update_model_params = False

        # Inference Vehicle Detection
        result = (
            self.model(frame_data, size=self.parameter.yolo_imgsz)
            .pandas()
            .xyxy[0]
            .to_numpy()
        )

        # Filtering Detection Area Object
        result = detection_area_filter(result, self.parameter.detection_area)

        # Tracking Algorithm
        if self.parameter.use_tracking == "Deep SORT":
            result = self.tracking_deepsort(result, frame_data)
        elif self.parameter.use_tracking == "SORT":
            result = self.tracking_sort(result, frame_data.shape)

        # Running Red Light Detection
        violation_result, non_violation_result = detection_running_redlight(
            result, self.parameter.stopline, status
        )

        # Running Wrong Way Detection
        self.wrongway.update(result)
        self.wrongway_violation_data = self.wrongway.get_wrong_way_list()

        # Save running light violation to database
        self.violation_recorder.write_violation_running_red_light(
            violation_result, frame_data, timestamp_video
        )

        # # Save Wrong Way Detection
        self.violation_recorder.write_violation_wrongway(
            result,
            frame_data,
            timestamp_video,
            self.wrongway.data_dict,
            self.wrongway_violation_data,
        )

        # Annotator Running Red Light
        frame_data = self.annotator(frame_data, non_violation_result)
        frame_data = self.annotator(frame_data, violation_result, violation=True)

        # Update Frame For Video Writer
        self.violation_recorder.update_video_writer(frame_data)

        return frame_data

    def tracking_deepsort(self, result, frame):
        det = result
        xywhs = xyxy2xywh(det[:, 0:4])
        if len(xywhs) > 0:
            confs = det[:, 4]
            clss = det[:, 5]
            outputs = self.deepsort.update(xywhs, confs, clss, frame)
            outputs = np.array(outputs)

            result_combine = combine_yolo_deepsort_result(yolo=det, deep_sort=outputs)
            return result_combine

        else:
            self.deepsort.increment_ages()
            return None

    def tracking_sort(self, result, frame_shape):
        det = result[:, 0:4]
        output = self.sort.update(det)
        combine_result = combine_yolo_sort_result(
            yolo=result, sort=output, frame_shape=frame_shape
        )
        return combine_result

    def annotator(self, frame: np.ndarray, result, violation=False):
        new_frame = frame.copy()
        if violation:
            color_box = (0, 0, 255)
        else:
            color_box = (0, 255, 0)

        ## draw object direction
        for i in range(0, result.shape[0]):
            arrow_length = 10
            data_tracker = result[i]
            object_id = data_tracker[6]

            # If object direction is wrong way set arrow color to red, else green
            if object_id in self.wrongway.data_dict.keys():
                x_center, y_center = calculate_center_of_box(data_tracker[0:4])
                if data_tracker[6] in self.wrongway_violation_data:
                    color_direction = (0, 0, 255)
                else:
                    color_direction = (0, 255, 0)

                _, phi = cart2pol(
                    self.wrongway.data_dict[object_id]["total_gx"],
                    self.wrongway.data_dict[object_id]["total_gy"],
                )

                pos1 = pol2cart(arrow_length, phi)
                pos2 = (-pos1[0], -pos1[1])

                new_frame = cv2.arrowedLine(
                    new_frame,
                    (int(x_center + pos2[0]), int(y_center + pos2[1])),
                    (int(x_center + pos1[0]), int(y_center + pos1[1])),
                    color_direction,
                    3,
                    cv2.LINE_AA,
                    tipLength=0.5,
                )

        # draw detection area
        if self.parameter.show_detection_area:
            new_frame = cv2.drawContours(
                new_frame,
                [np.array([self.parameter.detection_area])[0]],
                -1,
                (0, 255, 0),
                2,
            )

        # draw stop line
        if self.parameter.show_stopline:
            start_point = (
                self.parameter.stopline[0][0][0],
                self.parameter.stopline[0][0][1],
            )

            end_point = (
                self.parameter.stopline[1][0][0],
                self.parameter.stopline[1][0][1],
            )

            new_frame = cv2.line(
                new_frame, start_point, end_point, (0, 0, 255), 2, cv2.LINE_AA
            )

        # draw bounding boxes
        if self.parameter.show_bounding_boxes:
            for i in range(0, result.shape[0]):
                coordinate = result[i]

                new_frame = cv2.rectangle(
                    img=new_frame,
                    pt1=(int(coordinate[0]), int(coordinate[1])),
                    pt2=(int(coordinate[2]), int(coordinate[3])),
                    color=color_box,
                    thickness=2,
                )

        # draw label and confedence
        if self.parameter.show_label_and_confedence:

            font = cv2.FONT_HERSHEY_DUPLEX
            color = (0, 0, 0)
            thickness = 1
            fontScale = 0.5

            for i in range(0, result.shape[0]):
                data = result[i]
                org = (int(data[0]), int(data[1]))
                label = self.parameter.yolo_classes_name[str(int(data[5]))]
                if data[6] != None:
                    object_id = data[6]
                    text = f"{object_id} {label} {data[4]:.2f}"
                else:
                    text = f"{label}-{data[4]:.2f}"
                text_size, _ = cv2.getTextSize(
                    text, fontFace=font, fontScale=fontScale, thickness=thickness
                )
                text_w, text_h = text_size

                new_frame = cv2.rectangle(
                    new_frame,
                    pt1=(int(data[0]), int(data[1])),
                    pt2=(int(data[0] + text_w), int(data[1] - text_h)),
                    color=color_box,
                    thickness=-1,
                )

                new_frame = cv2.putText(
                    new_frame,
                    text,
                    org=org,
                    fontFace=font,
                    fontScale=fontScale,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )
        return new_frame

    def select_device(self, device: str):
        if device == "cpu":
            return torch.device("cpu")
        elif device == "gpu":
            return torch.device("gpu")
        else:
            raise Exception("Use 'cpu' or 'gpu' only!")

    def update_params(self, parameter: Parameter, update_model_params: bool = False):
        self.parameter = parameter
        self.wrongway.update_params(parameter)
        parameter.output_dir = "output"
        self.violation_recorder.update_params(parameter)
        self.update_model_params = update_model_params
