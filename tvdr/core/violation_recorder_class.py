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
from tvdr.utils.params import Parameter
from tvdr.core.algorithm import pol2cart, calculate_center_of_box


class ViolationRecorderMain:
    def __init__(self, parameter: Parameter):
        self.parameter = parameter

        self.detect_running_redlight_violation = False
        self.detect_wrongway_violation = False
        self.detect_helmet_violation = False

        self.arrow_length = 10

        self.text_font = cv2.FONT_HERSHEY_DUPLEX
        self.text_color = (0, 0, 0)
        self.text_thickness = 1
        self.text_fontscale = 0.5

        self.arrow_length = 10
        pass

    def annotate_result(self, img, result):
        # result[i] = [xmin, ymin, xmax, ymax, confidence, class, object_id, direction, helmet_violation, running_redlight, wrongway]

        img_new = img.copy()

        for obj in result:
            # If object running red light color bounding box red, else green
            if obj[10] == 1:
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

                text_bbox = f"{label} - {confidence:.2f} - {obj[6]}"

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
            if obj[8] == 1:
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

            # Create direction arrow
            if obj[9] == 1:
                color_arrow = (0, 0, 255)
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

            # Create detection  area
            print(f"Detection Area : {[np.array([self.parameter.detection_area])[0]]}")
            # if self.parameter.show_detection_area:
            #     img_new = cv2.drawContours(
            #         img_new,
            #         [np.array([self.parameter.detection_area])[0]],
            #         -1,
            #         (0, 255, 0),
            #         2,
            #     )
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
