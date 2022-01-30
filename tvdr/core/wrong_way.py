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

from tvdr.core.algorithm import calculate_center_of_box
from tvdr.core.algorithm import cart2pol
from tvdr.utils.params import Parameter
import numpy as np
import cv2


class WrongWayViolationDetection:
    def __init__(self, parameter: Parameter):
        self.update_params(parameter)

    def update_params(self, parameter: Parameter):
        # Get Video FPS
        vid = cv2.VideoCapture(parameter.video_path)
        self.fps = vid.get(cv2.CAP_PROP_FPS)

        # Set Wrong Way Params
        self.miss_count_threshold = parameter.wrongway_miss_count * self.fps
        self.min_value = parameter.wrongway_min_value
        self.direction_violation = parameter.wrongway_direction_degree
        self.direction_threshold = parameter.wrongway_threshold_degree
        self.object_history = {}

    def cartesian_to_polar(self, x, y):
        # Convert cartesian to polar
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return (rho, phi)

    def remove_object_miss(self):
        # Remove object if missing count above the threshold
        for id in self.object_history.copy():
            if self.object_history[id]["miss_count"] >= self.miss_count_threshold:
                self.object_history.pop(id)

    def update(self, result):
        id_show = []
        direction_data = {}

        for object in result:
            id = object[6]

            # Calculate centroid of the object
            x_center = object[0] + (object[0] + object[2]) / 2
            y_center = object[1] + (object[1] + object[3]) / 2

            # Check if object id not in object history then create a new data
            if id not in self.object_history.keys():
                id_dict = {
                    "age": 1,
                    "total_dx": 0,
                    "total_dy": 0,
                    "last_x": x_center,
                    "last_y": y_center,
                    "direction": None,
                    "miss_count": 0,
                }
                self.object_history[id] = id_dict

            else:
                last_x = self.object_history[id]["last_x"]
                last_y = self.object_history[id]["last_y"]

                self.object_history[id]["age"] += 1
                self.object_history[id]["total_dx"] += x_center - last_x
                self.object_history[id]["total_dy"] += y_center - last_y
                self.object_history[id]["last_x"] = x_center
                self.object_history[id]["last_y"] = y_center
                self.object_history[id]["miss_count"] = 0

                # Estimate object direction
                direction = np.degrees(
                    self.cartesian_to_polar(
                        self.object_history[id]["total_dx"],
                        -self.object_history[id]["total_dy"],
                    )[1]
                )

                if direction < 0:
                    direction = direction + 360

                self.object_history[id]["direction"] = int(direction)

            direction_data[id] = self.cartesian_to_polar(
                self.object_history[id]["total_dx"],
                self.object_history[id]["total_dy"],
            )[1]
            id_show.append(id)

        all_id = list(self.object_history.keys())
        id_not_show = list(set(all_id) - set(id_show))

        # Update age for object not shown
        for id in id_not_show:
            self.object_history[id]["age"] += 1
            self.object_history[id]["miss_count"] += 1

        # Remove missing object from object history
        self.remove_object_miss()

        # print(self.object_history)

        # Check direction of each object if
        wrong_way_result = self.check_wrong_way(id_show)

        return wrong_way_result, direction_data

        # print(f"Direction : {direction}")

        # print(self.object_history)

        #     pos = calculate_center_of_box(data[0:4])

        #     print(pos)

        #     # Update Direction For Each Object
        #     if id not in self.data_dict.keys():
        #         id_dict = {}
        #         id_dict["age"] = 1
        #         id_dict["total_gx"] = 0
        #         id_dict["total_gy"] = 0
        #         id_dict["last_x"] = pos[0]
        #         id_dict["last_y"] = pos[1]
        #         id_dict["show_count"] = 1
        #         id_dict["miss_count_repeat"] = 0
        #         id_dict["direction"] = None
        #         self.data_dict[id] = id_dict

        #         print(id_dict)

        #     else:
        #         self.data_dict[id]["age"] += 1
        #         self.data_dict[id]["total_gx"] += pos[0] - self.data_dict[id]["last_x"]
        #         self.data_dict[id]["total_gy"] += pos[1] - self.data_dict[id]["last_y"]
        #         self.data_dict[id]["last_x"] = pos[0]
        #         self.data_dict[id]["last_y"] = pos[1]
        #         self.data_dict[id]["show_count"] += 1
        #         self.data_dict[id]["miss_count_repeat"] = 0
        #         direction = np.degrees(
        #             cart2pol(
        #                 self.data_dict[id]["total_gx"],
        #                 -self.data_dict[id]["total_gy"],
        #             )[1]
        #         )
        #         if direction < 0:
        #             direction = 360 + direction

        #         self.data_dict[id]["direction"] = direction
        #     id_show.append(id)

        # all_id = list(self.data_dict.keys())
        # id_not_shown = list(set(all_id) - set(id_show))

        # for id in id_not_shown:
        #     id = int(id)
        #     self.data_dict[id]["age"] += 1
        #     self.data_dict[id]["miss_count_repeat"] += 1

        # # For Troubleshoot
        # print(f"Key : Direction | Total dXY | Miss")
        # for key in self.data_dict.keys():
        #     total = abs(self.data_dict[key]["total_gx"]) + abs(
        #         self.data_dict[key]["total_gy"]
        #     )
        #     direction = self.data_dict[key]["direction"]
        #     miss_count = self.data_dict[key]["miss_count_repeat"]
        #     print(f"{key} : {direction} | {total} | {miss_count}")
        #     print("\n")
        # print(
        #     f"Wrong Way Direction : {self.direction_violation}   |  Threshold Direction : {self.direction_threshold}"
        # )

        # self.remove_object_miss()
        # self.wrong_way_list = self.check_wrong_way()
        # return self.wrong_way_list

    # def get_wrong_way_list(self):
    #     # Get wrong way violation
    #     return self.wrong_way_list

    def check_wrong_way(self, list_id):
        wrong_way_id = []

        for id in list_id:
            total = abs(self.object_history[id]["total_dx"]) + abs(
                self.object_history[id]["total_dy"]
            )
            if total >= self.min_value:
                object_direction = self.object_history[id]["direction"]
                if self.detect_violation(object_direction):
                    wrong_way_id.append(id)

        return wrong_way_id

    def detect_violation(self, degree):
        # Check if degree direction of object is violation or not
        if self.direction_violation + self.direction_threshold > 360:
            th1_degrees = self.direction_threshold - (360 - self.direction_violation)
        else:
            th1_degrees = self.direction_violation + self.direction_threshold

        if self.direction_violation - self.direction_threshold < 0:
            th2_degrees = 360 - (self.direction_threshold - self.direction_violation)
        else:
            th2_degrees = self.direction_violation - self.direction_threshold

        if th1_degrees < th2_degrees:
            if degree <= th1_degrees or degree >= th2_degrees:
                return True
            else:
                return False

        else:
            if degree <= th1_degrees and degree >= th2_degrees:
                return True
            else:
                return False
