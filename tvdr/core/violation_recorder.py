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
import cv2
import time
import json

from tvdr.utils.params import Parameter
from tvdr.utils.path import create_folder
from tvdr.core.algorithm import cart2pol, pol2cart, calculate_center_of_box


class ViolationRecorder:
    """This class is to record violation data and inference result"""

    def __init__(self, parameter: Parameter):
        self.update_params(parameter)
        self.list_object_violation_running_redlight = {}
        self.list_object_violation_wrong_way = {}
        self.list_object_violation_not_using_helmet = {}
        self.json_data = {}

    def update_params(self, parameter: Parameter):

        self.parameter = parameter
        # Set output directory path
        self.output_dir = parameter.output_dir
        create_folder(self.output_dir)

        # Set path to save wrongway, running redlight, and not using helmet images
        self.wrongway_dir = os.path.join(self.output_dir, "wrongway")
        self.running_redlight_dir = os.path.join(self.output_dir, "running_redlight")
        self.not_using_helmet_dir = os.path.join(self.output_dir, "not_using_helmet")

        create_folder(self.wrongway_dir)
        create_folder(self.running_redlight_dir)
        create_folder(self.not_using_helmet_dir)

        # Set path to create database based on json format
        self.json_path = os.path.join(self.output_dir, "result.json")

        # Read video size and fps
        if parameter.video_path != "":
            vid = cv2.VideoCapture(parameter.video_path)
            _, frame = vid.read()
            fps = vid.get(cv2.CAP_PROP_FPS)
            height, width, _ = frame.shape

            # Set Video Writer
            self.video_output_path = os.path.join(self.output_dir, "result.avi")
            self.video_writer = cv2.VideoWriter(
                self.video_output_path,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                fps=fps,
                frameSize=(width, height),
            )

    def update_video_writer(self, img):
        # Update frame to video writer
        self.video_writer.write(img)

    def write_violation_wrongway(
        self, result, img0, timestamp_video, object_direction_data, violation_data
    ):
        for i in range(0, result.shape[0]):
            object_id = result[i][6]
            if (
                object_id in violation_data
                and object_id not in self.list_object_violation_wrong_way
            ):
                timestamp = int(time.time())
                img_proof_path = os.path.join(self.wrongway_dir, f"IMG_{timestamp}.jpg")
                img_proof = self.annotator_violation_wrongway(
                    result[i][0:4], img0.copy(), object_direction_data[object_id]
                )
                cv2.imwrite(img_proof_path, img_proof)

                minutes = timestamp_video // 60
                seconds = timestamp_video % 60

                violation_record = {}
                violation_record["vehicle_type"] = self.parameter.yolo_classes_name[
                    str(int(result[i][5]))
                ]
                violation_record["img_proof"] = img_proof_path
                violation_record["timestamp"] = f"{minutes:02}:{seconds:02}"

                self.list_object_violation_wrong_way[object_id] = violation_record

                self.database_writer()

    def write_violation_running_red_light(self, result, img0, timestamp_video):
        for i in range(0, result.shape[0]):
            object_id = result[i][6]
            if object_id not in self.list_object_violation_running_redlight.keys():
                timestamp = int(time.time())
                img_proof_path = os.path.join(
                    self.running_redlight_dir, f"IMG_{timestamp}.jpg"
                )
                img_proof = self.annotator_violation_redlight(result[i], img0.copy())
                cv2.imwrite(img_proof_path, img_proof)

                # self.list_object_violation_running_redlight.append(object_id)
                minutes = timestamp_video // 60
                seconds = timestamp_video % 60

                violation_record = {}
                violation_record["vehicle_type"] = self.parameter.yolo_classes_name[
                    str(int(result[i][5]))
                ]
                violation_record["img_proof"] = img_proof_path

                violation_record["timestamp"] = f"{minutes:02}:{seconds:02}"

                self.list_object_violation_running_redlight[
                    object_id
                ] = violation_record

                self.database_writer()

    def update_violation_not_using_helmet(self, img0, img1, result):
        pass

    def annotator_violation_wrongway(self, xyxy, img, data):
        img_new = img.copy()

        print(img_new)
        img_new = cv2.rectangle(
            img=img_new,
            pt1=(int(xyxy[0]), int(xyxy[1])),
            pt2=(int(xyxy[2]), int(xyxy[3])),
            color=(0, 0, 255),
            thickness=2,
        )

        x_center, y_center = calculate_center_of_box(xyxy)
        _, phi = cart2pol(
            data["total_gx"],
            data["total_gy"],
        )

        pos1 = pol2cart(5, phi)
        pos2 = (-pos1[0], -pos1[1])

        img_new = cv2.arrowedLine(
            img_new,
            (int(x_center + pos2[0]), int(y_center + pos2[1])),
            (int(x_center + pos1[0]), int(y_center + pos1[1])),
            (0, 0, 255),
            3,
            cv2.LINE_AA,
            tipLength=0.5,
        )

        return img_new

    def annotator_violation_redlight(self, xyxy, img):
        img_new = img.copy()
        img_new = cv2.rectangle(
            img=img_new,
            pt1=(int(xyxy[0]), int(xyxy[1])),
            pt2=(int(xyxy[2]), int(xyxy[3])),
            color=(0, 0, 255),
            thickness=2,
        )
        print(img_new)

        return img_new

    def database_writer(self):
        self.json_data[
            "running_red_light"
        ] = self.list_object_violation_running_redlight
        self.json_data["wrong_way "] = self.list_object_violation_wrong_way

        with open(self.json_path, "w", encoding="utf-8") as json_file:
            json.dump(self.json_data, json_file, indent=3)
