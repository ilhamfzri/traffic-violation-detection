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
from re import L
from types import coroutine
from typing import Text
from PIL import ImageDraw

import cv2
import torch
import numpy as np


class YOLOModel:
    """
    A class used to access YOLO Model

    ...

    Attributes
    ----------
    device : str
        run the model with 'cpu'/'gpu'
    """

    def __init__(self, device: Text, draw_bounding_boxes: bool = True):
        super().__init__()
        self.device = self.select_device(device)
        self.draw_bounding_boxes = draw_bounding_boxes

    def load_model(self, model_type: Text = "yolov5s"):
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")

    def inference_frame(self, frame_data: np.ndarray):
        print(frame_data.shape)
        self.result = self.model(frame_data)
        self.result_pandas = self.result.pandas().xyxy[0]

        print(self.result_pandas)
        print(len(self.result_pandas))

        arr_coordinates = np.empty((0, 4), dtype=np.float32)

        for i in range(0, len(self.result_pandas)):
            coordinate = self.result_pandas.iloc[i]
            arr_coordinates = np.append(
                arr_coordinates,
                np.array(
                    [
                        [
                            coordinate["xmin"],
                            coordinate["ymin"],
                            coordinate["xmax"],
                            coordinate["ymax"],
                        ]
                    ],
                ),
                axis=0,
            )

        if self.draw_bounding_boxes:
            img_inference = self.create_bounding_boxes(frame_data, arr_coordinates)

        return img_inference

    def select_device(self, device: str):
        if device == "cpu":
            return torch.device("cpu")
        elif device == "gpu":
            return torch.device("gpu")
        else:
            raise Exception("Use 'cpu' or 'gpu' only!")

    def create_bounding_boxes(self, frame: np.ndarray, coordinates: np.ndarray):
        new_frame = frame.copy()
        for i in range(0, len(coordinates)):
            coordinate = coordinates[i]
            new_frame = cv2.rectangle(
                img=new_frame,
                pt1=(int(coordinate[0]), int(coordinate[1])),
                pt2=(int(coordinate[2]), int(coordinate[3])),
                color=(0, 255, 0),
                thickness=1,
            )
        return new_frame

    def combine_human_and_motorcycle_bounding_boxes(self):
        pass
