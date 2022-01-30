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

from re import X
import numpy as np
from numpy.lib.polynomial import poly


def detection_area_filter(detection_result, detection_area):
    new_detection = np.empty((0, detection_result.shape[1]))
    for data in detection_result:
        point = calculate_center_of_box(data[0:4])
        if is_point_in_polygon(point, detection_area):
            new_detection = np.vstack((new_detection, data))
    return new_detection


def is_point_in_polygon(point, polygon_point):

    minX = polygon_point[0][0][0]
    maxX = polygon_point[0][0][0]
    minY = polygon_point[0][0][1]
    maxY = polygon_point[0][0][1]

    for poly_point in polygon_point:
        minX = min(poly_point[0][0], minX)
        maxX = max(poly_point[0][0], maxX)
        minY = min(poly_point[0][1], minY)
        maxY = max(poly_point[0][1], maxY)

    if point[0] < minX or point[0] > maxX or point[1] < minY or point[1] > maxY:
        return False

    length = len(polygon_point)
    j = length - 1
    for i in range(0, length):
        state_1 = (polygon_point[i][0][1] > point[1]) != (
            polygon_point[j][0][1] > point[1]
        )
        state_2 = point[0] < (
            (polygon_point[j][0][0] - polygon_point[i][0][0])
            * (point[1] - polygon_point[i][0][1])
            / (polygon_point[j][0][1] - polygon_point[i][0][1])
            + polygon_point[i][0][0]
        )
        if state_1 and state_2:
            return True
        j = i
    return False


def calculate_center_of_box(point_xyxy):
    x_min = point_xyxy[0]
    y_min = point_xyxy[1]
    x_max = point_xyxy[2]
    y_max = point_xyxy[3]

    x_center = x_min + ((x_max - x_min) / 2)
    y_center = y_min + ((y_max - y_min) / 2)
    return (x_center, y_center)


def detection_running_redlight(detection_result, line, status):
    non_violation_result = np.empty((0, detection_result.shape[1]))
    violation_result = np.empty((0, detection_result.shape[1]))
    for data in detection_result:
        bbox = data[0:4]
        polygon = convert_bbox_to_polygon(bbox)
        if status == "Red":
            if is_line_intersection_polygon(line, polygon):
                violation_result = np.vstack((violation_result, data))
            else:
                non_violation_result = np.vstack((non_violation_result, data))
        else:
            non_violation_result = np.vstack((non_violation_result, data))

    return violation_result, non_violation_result


def convert_bbox_to_polygon(bbox):
    poly_point = []
    poly_point.append([bbox[0], bbox[1]])
    poly_point.append([bbox[0], bbox[3]])
    poly_point.append([bbox[2], bbox[1]])
    poly_point.append([bbox[2], bbox[3]])

    return poly_point


def is_line_intersection_polygon(line, polygon):

    line_point_1 = line[0][0]
    line_point_2 = line[1][0]

    x1 = line_point_1[0]
    y1 = line_point_1[1]

    x2 = line_point_2[0]
    y2 = line_point_2[1]

    index_next = 0
    for index_current in range(0, len(polygon)):

        if index_current == len(polygon) - 1:
            index_next = 0
        else:
            index_next = index_current + 1

        current_point = (polygon[index_current][0], polygon[index_current][1])
        next_point = (polygon[index_next][0], polygon[index_next][1])

        x3 = current_point[0]
        y3 = current_point[1]

        x4 = next_point[0]
        y4 = next_point[1]

        uA = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (
            (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        )

        uB = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / (
            (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        )

        if uA >= 0 and uA <= 1 and uB >= 0 and uB <= 1:
            return True

    return False


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)
