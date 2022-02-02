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
from numpy import half
import torch
import logging
import cv2
import numpy as np

from tvdr.utils.params import Parameter
from tvdr.core import Sort
from tvdr.utils.general import combine_yolo_sort_result

from yolov5_repo.models.common import DetectMultiBackend
from yolov5_repo.utils.torch_utils import select_device
from yolov5_repo.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5_repo.utils.augmentations import letterbox
from yolov5_repo.utils.datasets import LoadImages

logging_root = "Vehicle Detection"


class VehicleDetection:
    def __init__(self, parameter: Parameter):
        self.load_parameters(parameter)

        # SORT initialize
        self.sort = Sort(
            self.parameter.sort_max_age,
            self.parameter.sort_min_hits,
            self.parameter.sort_iou_threshold,
        )

        self.model_loaded_state = self.load_model()

    def predict(self, img0):

        # Padding image
        im = letterbox(
            img0,
            new_shape=self.imgsz,
            stride=self.model_stride,
            auto=True,
            scaleFill=True,
        )[0]

        # Convert cv2 colorspace from bgr to rgb
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device_torch).float()

        # Normalize
        im /= 255

        # if inference is single image size then add dimension
        if len(im.shape) == 3:
            im = im[None]

        # Process inference
        result = self.model(im)

        # Non max suppression
        result = non_max_suppression(
            prediction=result,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            max_det=self.max_detection,
        )

        # Rescale result to original resolution
        result = result[0]
        result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img0.shape).round()

        if result.is_cuda:
            result = result.cpu()

        result = result.numpy()
        self.result_without_tracking = result

        # Tracking algorithm
        if self.parameter.use_tracking != "No Tracker":
            result = self.object_tracking(result)

        # Post processing
        if len(self.parameter.detection_area) > 3:
            result = self.post_processing(result)

        return result

    def get_without_tracking(self):
        return self.result_without_tracking

    def load_model(self):

        print("Load Vehicle Detection Model")
        try:
            # Check inference device
            if self.device == "gpu":
                if torch.cuda.is_available() == False:
                    logging.warning(
                        f"{logging_root}: CUDA not available!, change to CPU"
                    )
                    self.select_device = "cpu"
                else:
                    self.select_device = "0"
                    logging.info(f"{logging_root}: model use CUDA")
            else:
                self.select_device = "cpu"
                logging.info(f"{logging_root}: model use CPU")

            # Load model
            self.device_torch = select_device(device=self.select_device)
            self.model = DetectMultiBackend(self.model_path, device=self.device_torch)
            self.model_stride = self.model.stride
            self.imgsz = check_img_size(imgsz=self.inference_size, s=self.model_stride)

            # Warmup
            self.model.warmup(imgsz=(1, 3, *self.imgsz))
            return True

        except:
            return False

    def load_parameters(self, parameter: Parameter):

        # Set model and inference parameters
        self.parameter = parameter
        self.model_path = self.parameter.yolo_model_path
        self.device = self.parameter.device
        self.inference_size = (self.parameter.yolo_imgsz, self.parameter.yolo_imgsz)
        self.conf_thres = self.parameter.yolo_conf
        self.iou_thres = self.parameter.yolo_iou
        self.classes = self.parameter.yolo_classes
        self.classes_name = self.parameter.yolo_classes_name
        self.max_detection = self.parameter.yolo_max_detection
        self.multi_label = self.parameter.yolo_multi_label

    def object_tracking(self, yolo_result):
        # Process SORT tracking algorithm
        if self.parameter.use_tracking == "SORT":
            bbox_result = yolo_result[:, 0:4]
            sort_result = self.sort.update(bbox_result)
            tracking_result = combine_yolo_sort_result(yolo_result, sort_result)
            return tracking_result

    def post_processing(self, result):
        # Filtering only objects inside the detection area will be evaluated.
        # This algorithm taken from here : https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon

        result_post = np.empty((0, result.shape[1]))
        for object in result:

            # calculate centroid for object
            x_center = object[0] + abs(object[2] - object[0]) / 2
            y_center = object[1] + abs(object[3] - object[1]) / 2

            # check if centroid inside polygon points of the detection area
            np_detection_area = np.array(self.parameter.detection_area)

            x_min_da = np.min(np_detection_area[:, 0, 0])
            y_min_da = np.min(np_detection_area[:, 0, 1])

            x_max_da = np.max(np_detection_area[:, 0, 0])
            y_max_da = np.max(np_detection_area[:, 0, 1])

            if (
                x_center < x_min_da
                or x_center > x_max_da
                or y_center < y_min_da
                or y_center > y_max_da
            ):
                continue

            inside = False

            j = np_detection_area.shape[0] - 1

            for i in range(0, np_detection_area.shape[0]):
                xi_da, yi_da = np_detection_area[i][0]
                xj_da, yj_da = np_detection_area[j][0]

                state_1 = (yi_da > y_center) != (yj_da > y_center)
                state_2 = x_center < (
                    (xj_da - xi_da) * (y_center - yi_da) / (yj_da - yi_da) + xi_da
                )

                if state_1 and state_2:
                    inside = not inside

                j = i

            if inside:
                result_post = np.append(
                    result_post, object.reshape(1, result.shape[1]), axis=0
                )

        return result_post

    def model_loaded(self):
        # To check if model loaded
        if self.model_loaded_state == True:
            return True
        else:
            return False
