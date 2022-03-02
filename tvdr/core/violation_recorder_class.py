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

import cv2
import numpy as np
import os
import time
import json

from datetime import datetime as dt
from tvdr.utils.params import Parameter
from tvdr.core.algorithm import cart2pol, pol2cart, calculate_center_of_box
from tvdr.utils.path import create_folder


class ViolationRecorderMain:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter

        self.detect_running_redlight_violation = False
        self.detect_wrongway_violation = False
        self.detect_helmet_violation = False

        self.label_maping = ["car", "motorcycle", "bus", "truck", "bycycle"]

        self.vehicle_counter = {
            "car": [],
            "motorcycle": [],
            "bus": [],
            "truck": [],
            "bycycle": [],
        }

        self.show_traffic_light_status = True
        self.show_wrong_way_status = True
        self.show_running_redlight_status = True
        self.show_helmet_violation_status = True
        self.show_vehicle_counter = True

        self.helmet_violation_counter = []
        self.running_redlight_counter = []
        self.wrong_way_counter = []

        self.arrow_length = 10

        self.text_font = cv2.FONT_HERSHEY_DUPLEX
        self.text_color = (0, 0, 0)
        self.text_thickness = 1
        self.text_fontscale = 0.5

        self.arrow_length = 10
        self.helmet_annotate_history = []
        pass

    def annotate_result(
        self, img, result, remember_violation=True, traffic_light="Red"
    ):
        # result[i] = [xmin, ymin, xmax, ymax, confidence, class, object_id, direction, helmet_violation, running_redlight, wrongway]

        img_new = img.copy()

        for obj in result:
            # If object running red light color bounding box red, else green
            if obj[10] == 1:
                if obj[6] not in self.running_redlight_counter:
                    self.running_redlight_counter.append(obj[6])
                color_box = (0, 0, 255)
            else:
                color_box = (0, 255, 0)

            # create bounding_boxes
            if self.parameter.show_bounding_boxes:
                img_new = cv2.rectangle(
                    img=img_new,
                    pt1=(int(obj[0]), int(obj[1])),
                    pt2=(int(obj[2]), int(obj[3])),
                    color=color_box,
                    thickness=2,
                )

            if self.parameter.show_label_and_confedence:
                confidence = obj[4]
                label = obj[5]

                # text_bbox = f"{self.label_maping[label]} - {confidence:.2f} - {obj[6]}"
                text_bbox = f"{self.label_maping[int(label)]} - {obj[6]}"

                # Calculate text size
                text_size, _ = cv2.getTextSize(
                    text_bbox, self.text_font, self.text_fontscale, self.text_thickness
                )
                text_w, text_h = text_size

                # Create background color for text
                img_new = cv2.rectangle(
                    img_new,
                    pt1=(int(obj[0]), int(obj[1])),
                    pt2=(int(obj[0] + text_w), int(obj[1] - text_h)),
                    color=color_box,
                    thickness=-1,
                )

                # Generate text
                img_new = cv2.putText(
                    img_new,
                    text_bbox,
                    org=(int(obj[0]), int(obj[1])),
                    fontFace=self.text_font,
                    fontScale=self.text_fontscale,
                    color=self.text_color,
                    thickness=self.text_thickness,
                    lineType=cv2.LINE_AA,
                )

            # If object helmet violation add text no helmet below bounding boxes
            if (
                obj[8] == 1
                or obj[6] in self.helmet_annotate_history
                and remember_violation
            ):

                text = "No Helmet"

                # Calculate text size
                text_size, _ = cv2.getTextSize(
                    text_bbox, self.text_font, self.text_fontscale, self.text_thickness
                )
                text_w, text_h = text_size

                # Generate text
                img_new = cv2.putText(
                    img_new,
                    text,
                    org=(int(obj[0]), int(obj[3] + text_h)),
                    fontFace=self.text_font,
                    fontScale=self.text_fontscale,
                    color=(0, 0, 255),
                    thickness=self.text_thickness,
                    lineType=cv2.LINE_AA,
                )

                id = obj[6]
                if id not in self.helmet_violation_counter:
                    self.helmet_violation_counter.append(id)

                if id not in self.helmet_annotate_history:
                    self.helmet_annotate_history.append(id)

            # Create direction arrow
            if obj[9] == 1:
                color_arrow = (0, 0, 255)
                id = obj[6]
                if id not in self.wrong_way_counter:
                    self.wrong_way_counter.append(id)
            else:
                color_arrow = (0, 255, 0)

            direction_phi = obj[7]
            x_center, y_center = calculate_center_of_box(obj[0:4])
            pos1 = pol2cart(self.arrow_length, direction_phi)
            pos2 = (-pos1[0], -pos1[1])

            img_new = cv2.arrowedLine(
                img_new,
                (int(x_center + pos2[0]), int(y_center + pos2[1])),
                (int(x_center + pos1[0]), int(y_center + pos1[1])),
                color_arrow,
                3,
                cv2.LINE_AA,
                tipLength=0.5,
            )

            label = int(obj[5])
            id = obj[6]
            if id not in self.vehicle_counter[self.label_maping[label]]:
                self.vehicle_counter[self.label_maping[label]].append(id)

        # Create detection  area
        if self.parameter.show_detection_area:
            img_new = cv2.drawContours(
                img_new,
                [np.array([self.parameter.detection_area])[0]],
                -1,
                (0, 255, 255),
                2,
            )

        # Create stopline
        if self.parameter.show_stopline:
            start_point = (
                self.parameter.stopline[0][0][0],
                self.parameter.stopline[0][0][1],
            )

            end_point = (
                self.parameter.stopline[1][0][0],
                self.parameter.stopline[1][0][1],
            )

            img_new = cv2.line(
                img_new, start_point, end_point, (0, 0, 255), 2, cv2.LINE_AA
            )

        x_info, y_info = 30, 30
        spacing = 40

        if self.show_running_redlight_status:
            text = f"Traffic Light : {traffic_light}"
            text_size, _ = cv2.getTextSize(text, self.text_font, 1, 2)
            text_w, text_h = text_size

            img_new = cv2.rectangle(
                img_new,
                pt1=(x_info - 5, y_info - 10),
                pt2=(int(x_info + text_w + 5), int(y_info + text_h + 10)),
                color=(0, 0, 0),
                thickness=-1,
            )

            # Generate text
            img_new = cv2.putText(
                img_new,
                text,
                org=(x_info, int(y_info + text_h)),
                fontFace=self.text_font,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            y_info = y_info + text_h + spacing

        if self.show_running_redlight_status:
            text = f"Running Red Light: {len(self.running_redlight_counter)}"
            text_size, _ = cv2.getTextSize(text, self.text_font, 1, 2)
            text_w, text_h = text_size

            img_new = cv2.rectangle(
                img_new,
                pt1=(x_info - 5, y_info - 10),
                pt2=(int(x_info + text_w + 5), int(y_info + text_h + 10)),
                color=(0, 0, 0),
                thickness=-1,
            )

            # Generate text
            img_new = cv2.putText(
                img_new,
                text,
                org=(x_info, int(y_info + text_h)),
                fontFace=self.text_font,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            y_info = y_info + text_h + spacing

        if self.show_wrong_way_status:
            text = f"Wrong Way : {len(self.wrong_way_counter)}"
            text_size, _ = cv2.getTextSize(text, self.text_font, 1, 2)
            text_w, text_h = text_size

            img_new = cv2.rectangle(
                img_new,
                pt1=(x_info - 5, y_info - 10),
                pt2=(int(x_info + text_w + 5), int(y_info + text_h + 10)),
                color=(0, 0, 0),
                thickness=-1,
            )

            # Generate text
            img_new = cv2.putText(
                img_new,
                text,
                org=(x_info, int(y_info + text_h)),
                fontFace=self.text_font,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            y_info = y_info + text_h + spacing

        if self.show_helmet_violation_status:
            text = f"Helmet Violation : {len(self.helmet_violation_counter)}"
            text_size, _ = cv2.getTextSize(text, self.text_font, 1, 2)
            text_w, text_h = text_size

            img_new = cv2.rectangle(
                img_new,
                pt1=(x_info - 5, y_info - 10),
                pt2=(int(x_info + text_w + 5), int(y_info + text_h + 10)),
                color=(0, 0, 0),
                thickness=-1,
            )

            # Generate text
            img_new = cv2.putText(
                img_new,
                text,
                org=(x_info, int(y_info + text_h)),
                fontFace=self.text_font,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )
            y_info = y_info + text_h + spacing

        text_dummy = "Motorcycle : 213123"

        text_w, _ = text_size
        _, img_x, _ = img.shape

        x_info_vehicle, y_info_vehicle, space_y = img_x - text_w - 15, 20, 40
        # x_info_vehicle, y_info_vehicle, space_y = img_x - text_w - 20, 15, 40

        if self.show_vehicle_counter:
            car_count = len(self.vehicle_counter["car"])
            motorcycle_count = len(self.vehicle_counter["motorcycle"])
            bus_count = len(self.vehicle_counter["bus"])
            truck_count = len(self.vehicle_counter["truck"])

            text = f"Car        : {car_count}\nMotorcycle : {motorcycle_count}\nBus        : {bus_count}\nTruck      : {truck_count}"

            for i, line in enumerate(text.split("\n")):
                text_size, _ = cv2.getTextSize(line, self.text_font, 1, 2)
                text_w, text_h = text_size
                y = y_info_vehicle + i * (text_h + space_y)

                img_new = cv2.rectangle(
                    img_new,
                    pt1=(x_info_vehicle - 5, y - 10),
                    pt2=(
                        int(x_info_vehicle + text_w + 5),
                        int(y + text_h + 10),
                    ),
                    color=(0, 0, 0),
                    thickness=-1,
                )

                img_new = cv2.putText(
                    img_new,
                    line,
                    org=(x_info_vehicle, int(y + text_h)),
                    fontFace=self.text_font,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            # Generate text

        return img_new

    def detection_combiner(
        self,
        vehicle_detection_result,
        direction_data=None,
        helmet_violation_result=None,
        wrongway_violation_result=None,
        running_redlight_result=None,
    ):

        result_final = np.empty((0, 11))

        for object in vehicle_detection_result:
            object_id = object[6]

            # add direction data
            if direction_data is not None:
                if object_id in list(direction_data.keys()):
                    direction = direction_data[object_id]
                    object = np.append(object, [direction], axis=0)
                else:
                    object = np.append(object, [-1], axis=0)
            else:
                object = np.append(object, [-1], axis=0)

            # add helmet violation data
            if helmet_violation_result is not None:
                if object_id in helmet_violation_result:
                    object = np.append(object, [1], axis=0)
                else:
                    object = np.append(object, [0], axis=0)
            else:
                object = np.append(object, [0], axis=0)

            # add wrongway violation result
            if wrongway_violation_result is not None:
                if object_id in wrongway_violation_result:
                    object = np.append(object, [1], axis=0)
                else:
                    object = np.append(object, [0], axis=0)
            else:
                object = np.append(object, [0], axis=0)

            # add running red ligt result
            if running_redlight_result is not None:
                if object_id in running_redlight_result:
                    object = np.append(object, [1], axis=0)
                else:
                    object = np.append(object, [0], axis=0)
            else:
                object = np.append(object, [0], axis=0)

            result_final = np.append(result_final, object.reshape(1, 11), axis=0)

        return result_final

    def update_params(self, parameter: Parameter):
        self.parameter = parameter

    def video_recorder_init(self, video_path, output_dir):
        vid = cv2.VideoCapture(video_path)
        _, frame = vid.read()
        fps = vid.get(cv2.CAP_PROP_FPS)
        height, width, _ = frame.shape

        basename = os.path.basename(video_path)
        basename = os.path.splitext(basename)[0]

        # print(fps)
        self.video_size = (height, width)
        out_video_path = os.path.join(output_dir, f"result.mp4")
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        self.video_writer = cv2.VideoWriter(
            out_video_path,
            fourcc=fourcc,
            fps=fps,
            frameSize=(width, height),
        )

        self.list_object_violation_running_redlight = {}
        self.list_object_violation_wrong_way = {}
        self.list_object_violation_not_using_helmet = {}
        self.json_data = {}

    def video_recorder_update(self, frame):
        self.video_writer.write(frame)
        # print(self.video_size)
        # print(frame.shape)

    def database_writer_init(self, output_dir):
        # Set output directory path
        subfolder_name_date = dt.now().strftime("%m_%d_%Y_%H_%M_%S")
        subfolder_path = os.path.join(output_dir, f"run_{subfolder_name_date}")
        self.output_dir = subfolder_path

        create_folder(output_dir)
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

        self.video_recorder_init(self.parameter.video_path, self.output_dir)

    def write_violation_wrongway2(
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
                    int(result[i][5])
                ]
                violation_record["img_proof"] = img_proof_path
                violation_record["timestamp"] = f"{minutes:02}:{seconds:02}"

                self.list_object_violation_wrong_way[object_id] = violation_record

                self.database_writer()

    def write_violation_wrongway(self, result, img0, frame_index):
        for i in range(0, result.shape[0]):
            object_id = result[i][6]
            violation_state = result[i][9]
            if (
                object_id not in self.list_object_violation_wrong_way.keys()
                and violation_state == 1
            ):
                timestamp = int(time.time())
                img_proof_path = os.path.join(self.wrongway_dir, f"IMG_{timestamp}.jpg")
                direction = result[i][7]
                img_proof = self.annotator_violation_wrongway(
                    result[i], img0.copy(), direction
                )
                cv2.imwrite(img_proof_path, img_proof)

                minutes = frame_index // 60
                seconds = frame_index % 60

                violation_record = {}
                violation_record["vehicle_type"] = self.parameter.yolo_classes_name[
                    str(int(result[i][5]))
                ]
                violation_record["img_proof"] = img_proof_path
                violation_record["timestamp"] = f"{minutes:02}:{seconds:02}"

                self.list_object_violation_wrong_way[object_id] = violation_record
                self.database_writer()

    def write_violation_running_red_light(self, result, img0, frame_index):
        for i in range(0, result.shape[0]):
            object_id = result[i][6]
            violation_state = result[i][10]
            if (
                object_id not in self.list_object_violation_running_redlight.keys()
                and violation_state == 1
            ):

                timestamp = int(time.time())
                img_proof_path = os.path.join(
                    self.running_redlight_dir, f"IMG_{timestamp}.jpg"
                )
                img_proof = self.annotator_violation_redlight(result[i], img0.copy())
                cv2.imwrite(img_proof_path, img_proof)

                # self.list_object_violation_running_redlight.append(object_id)
                minutes = frame_index // 60
                seconds = frame_index % 60

                violation_record = {}
                violation_record["vehicle_type"] = self.label_maping[int(result[i][5])]
                violation_record["img_proof"] = img_proof_path

                violation_record["timestamp"] = f"{minutes:02}:{seconds:02}"

                self.list_object_violation_running_redlight[
                    object_id
                ] = violation_record

                self.database_writer()

    def annotator_violation_wrongway(self, xyxy, img, direction):
        img_new = img.copy()
        x_center, y_center = calculate_center_of_box(xyxy[0:4])
        pos1 = pol2cart(self.arrow_length, direction)
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
        img_new = cv2.rectangle(
            img=img_new,
            pt1=(int(xyxy[0]), int(xyxy[1])),
            pt2=(int(xyxy[2]), int(xyxy[3])),
            color=(0, 0, 255),
            thickness=2,
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

        return img_new

    def database_writer(self):
        self.json_data[
            "running_red_light"
        ] = self.list_object_violation_running_redlight
        self.json_data["wrong_way "] = self.list_object_violation_wrong_way

        with open(self.json_path, "w", encoding="utf-8") as json_file:
            json.dump(self.json_data, json_file, indent=3)
