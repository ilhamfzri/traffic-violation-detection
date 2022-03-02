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
import numpy as np
import torch
import torch.nn.functional as F

from tvdr.utils.params import Parameter


class HelmetViolationDetectionClassifier:
    """
    Helmet violation detection using classifier method (EfficientNet)
    """

    def __init__(self, parameter: Parameter):
        self.motorcycle_idx = 1
        self.bicycle_idx = 4
        self.id_tracker = {}
        self.model_path = ""
        self.missing_removal_threshold = 150
        self.index_no_helmet = 1
        self.update_params(parameter)
        self.load_model()

    def update(self, img0, tracking_result):
        filter_result = self.motorcycle_and_bicycle_filtering(tracking_result)
        self.tracker_record_update(filter_result)
        object_inference = self.get_object_inference(filter_result)
        violation_result = self.detect_violation(img0, object_inference)
        return violation_result

    def update_params(self, parameter: Parameter):

        model_path_temp = self.model_path
        self.min_conf = parameter.hv_min_conf
        self.min_age = parameter.hv_min_age
        self.model_path = parameter.hv_model_path
        self.detect_interval = parameter.hv_detect_interval
        self.device = "cpu"
        self.imgsz = 224

        if model_path_temp != self.model_path:
            self.load_model()

        # self.min_conf = (
        #     parameter.hv_min_conf if parameter.hv_min_conf is not None else 0.9
        # )
        # self.min_age = parameter.hv_min_age if parameter.hv_min_age is not None else 0.5
        # self.model_path = parameter.hv_model_path
        # self.detect_interval = (
        #     parameter.hv_detect_interval
        #     if parameter.hv_detect_interval is not None
        #     else 20
        # )
        # self.device = parameter.device
        # self.imgsz = parameter.hv_imgsz if parameter.hv_imgsz is not None else 244

    def motorcycle_and_bicycle_filtering(self, tracking_result):
        # Filtering result from vehicle detection so only motorcycle and bicycle will be evaluated
        result_filter = np.empty((0, tracking_result.shape[1]))
        for obj in tracking_result:
            if obj[5] == self.motorcycle_idx or obj[5] == self.bicycle_idx:
                result_filter = np.append(
                    result_filter, obj.reshape(1, tracking_result.shape[1]), axis=0
                )
        return result_filter

    def load_model(self):
        # Load model
        self.model = torch.load(
            self.model_path, map_location=torch.device(self.device)
        )["model"].float()

    def tracker_record_update(self, result_filter):
        list_id = []
        for object in result_filter:
            id = object[6]
            if id not in self.id_tracker.keys():
                self.id_tracker[id] = {
                    "age": 1,
                    "missing": 0,
                }
            if id in self.id_tracker.keys():
                self.id_tracker[id]["age"] += 1
                self.id_tracker[id]["missing"] = 0

            list_id.append(id)

        not_in_list_id = list(set(list(self.id_tracker.keys())) - set(list_id))
        for id in not_in_list_id:
            self.id_tracker[id]["age"] += 1
            self.id_tracker[id]["missing"] += 1

        list_delete_id = []
        for id in self.id_tracker.keys():
            if self.id_tracker[id]["missing"] >= self.missing_removal_threshold:
                list_delete_id.append(id)

        for id_delete in list_delete_id:
            del self.id_tracker[id_delete]

    def get_object_inference(self, result_filter):
        object_inference = np.empty((0, result_filter.shape[1]))
        print(self.id_tracker)
        for object in result_filter:
            id = object[6]
            if (
                self.id_tracker[id]["age"] > self.min_age
                and self.id_tracker[id]["age"] % self.detect_interval == 0
            ):
                object_inference = np.append(
                    object_inference, object.reshape(1, 7), axis=0
                )
        return object_inference

    def image_croping(self, img, bbox):
        # croping image to get motorcycle and bicycle frame
        x, y = int(bbox[0]), int(bbox[1])
        w, h = int(abs(bbox[2] - bbox[0])), int(abs(bbox[3] - bbox[1]))
        img_crop = img[y : y + h, x : x + w, :]
        return img_crop

    def detect_violation(self, img, object_inference):
        # Detect helmet violation
        violation_result = []
        print("Predict Violation")
        for obj in object_inference:

            # Create Croping Image
            img_crop = self.image_croping(img, obj[0:4])
            predict_result = self.predict(img_crop)

            if predict_result == True:
                object_id = obj[6]
                violation_result.append(object_id)

        return violation_result

    def predict(self, img):
        resize = torch.nn.Upsample(
            size=(self.imgsz, self.imgsz), mode="bilinear", align_corners=False
        )
        normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std

        im = np.ascontiguousarray(np.asarray(img).transpose((2, 0, 1)))
        im = torch.tensor(im).float().unsqueeze(0) / 255.0
        im = resize(normalize(im))

        with torch.no_grad():
            results = self.model(im)
            p = F.softmax(results, dim=1)  # probabilities
            i = p.argmax()  # max index

        if int(i) == self.index_no_helmet and p[0, i] >= self.min_conf:
            return True

        return False
