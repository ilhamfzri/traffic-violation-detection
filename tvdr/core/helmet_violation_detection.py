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
import time
import torch
import logging
import cv2
import numpy as np
from tvdr.utils.params import Parameter

from yolov5_repo.models.common import DetectMultiBackend
from yolov5_repo.utils.torch_utils import select_device
from yolov5_repo.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5_repo.utils.augmentations import letterbox
from yolov5_repo.utils.datasets import LoadImages

logging_root = "Helmet Detection"


class HelmetViolationDetection:
    def __init__(self, parameter):
        self.load_parameters(parameter)
        self.motorcycle_idx = 1
        self.bicycle_idx = 4
        self.padding_color = (0, 0, 0)
        self.object_history = {}
        self.missing_removal_threshold = 150
        self.model_loaded_state = False

    def padding_image(self, img):
        new_height = int(img.shape[0] * self.padding_height_mul)
        new_width = int(img.shape[1] * self.padding_width_mul)

        thickness_height = int(abs(new_height - img.shape[0]) / 2)
        thickness_width = int(abs(new_width - img.shape[1]) / 2)

        new_img = np.zeros((new_height, new_width, 3))
        new_img[0:thickness_height, :] = self.padding_color
        new_img[:, 0:thickness_width] = self.padding_color
        new_img[
            thickness_height : thickness_height + img.shape[0],
            thickness_width : thickness_width + img.shape[1],
        ] = img

        return new_img

    def image_croping(self, img, bbox):
        # croping image to get motorcycle and bicycle frame
        x, y = int(bbox[0]), int(bbox[1])
        w, h = int(abs(bbox[2] - bbox[0])), int(abs(bbox[3] - bbox[1]))
        img_crop = img[y : y + h, x : x + w, :]
        return img_crop

    def detect_violation(self, img, result):
        # Detect helmet violation
        violation_result = []
        for obj in result:
            img_crop = self.image_croping(img, obj[0:4])
            img_padding = self.padding_image(img_crop)
            predict_result = self.predict(img_padding)

            for pred in predict_result:
                if pred[5] == 0:
                    object_id = obj[6]
                    violation_result.append(object_id)
                    break

        return violation_result

    def motorcycle_and_bicycle_filtering(self, result):
        # Filtering result from vehicle detection so only motorcycle and bicycle will be evaluated
        result_filter = np.empty((0, result.shape[1]))
        for obj in result:
            if obj[5] == self.motorcycle_idx or obj[5] == self.bicycle_idx:
                result_filter = np.append(
                    result_filter, obj.reshape(1, result.shape[1]), axis=0
                )

        return result_filter

    def update(self, img, result):
        # Filtering only motorcycle or bicycle object that will be evaluating
        result_filter = self.motorcycle_and_bicycle_filtering(result)

        list_id = []
        # Track age and missing from object
        for object in result_filter:
            id = object[6]
            if id not in self.object_history.keys():
                self.object_history[id] = {
                    "age": 1,
                    "missing": 0,
                    "inference_state": False,
                }
            if id in self.object_history.keys():
                self.object_history[id]["age"] += 1
                self.object_history[id]["missing"] = 0

            list_id.append(id)

        # Update id not in list
        not_in_list_id = list(set(list(self.object_history.keys())) - set(list_id))
        for id in not_in_list_id:
            self.object_history[id]["age"] += 1
            self.object_history[id]["missing"] += 1

        # Check if object appear and age above the threshold of minimum age then inference
        object_inference = np.empty((0, result.shape[1]))
        for object in result_filter:
            id = object[6]
            if (
                self.object_history[id]["age"] > self.min_age
                and self.object_history[id]["age"] % self.detect_interval == 0
            ):
                object_inference = np.append(
                    object_inference, object.reshape(1, 7), axis=0
                )

            # if (
            #     self.object_history[id]["age"] > self.min_age
            #     and self.object_history[id]["inference_state"] != True
            # ):
            #     object_inference = np.append(
            #         object_inference, object.reshape(1, 7), axis=0
            #     )
            #     self.object_history[id]["inference_state"] = True

        # Remove object history if missing above the threshold
        list_delete_id = []
        for id_history in self.object_history.keys():
            if (
                self.object_history[id_history]["missing"]
                >= self.missing_removal_threshold
            ):
                list_delete_id.append(id_history)

        for id_delete in list_delete_id:
            del self.object_history[id_delete]

        # For debuging only.
        # print(f"Result Filter : {result_filter}")
        # print(f"Object History : {self.object_history}")
        # print(f"Object Inference : {object_inference}")

        # Detect violation from list of objects that meet the criteria
        print(f"TEST : {object_inference}")
        violation_result = self.detect_violation(img, object_inference)

        # print(f"Violation Result : {violation_result}")

        return violation_result

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
        )

        # Rescale result to original resolution
        result = result[0]
        result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img0.shape).round()

        return result

    def load_model(self):
        # Check inference device
        if self.device == "gpu":
            if torch.cuda.is_available() == False:
                logging.warning(f"{logging_root}: CUDA not available!, change to CPU")
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

        self.model_loaded_state = True

    def load_parameters(self, parameter: Parameter):
        # Load parameters
        self.device = parameter.device
        self.model_path = parameter.hv_model_path
        self.min_age = parameter.hv_min_age
        self.detect_interval = 10
        self.conf_thres = parameter.hv_conf
        self.iou_thres = parameter.hv_iou
        self.inference_size = (parameter.hv_imgsz, parameter.hv_imgsz)
        self.padding_height_mul = parameter.hv_pad_height_mul
        self.padding_width_mul = parameter.hv_pad_width_mul

    def model_loaded(self):
        # To check if model loaded
        if self.model_loaded_state == True:
            return True
        else:
            return False
