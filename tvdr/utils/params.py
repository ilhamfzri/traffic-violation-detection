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


class Parameter:
    def __init__(self):
        self.yolo_model_dict = {
            "YOLOV5 Small": "yolov5s",
            "YOLOV5 Medium": "yolov5m",
        }

        self.yolo_imgsz = 250
        self.yolo_stride = "2"
        self.video_path = ""
        self.yolo_conf = 0.25
        self.yolo_iou = 0.45
        self.yolo_classes = [2, 3, 5, 7]
        self.yolo_classss_names = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck",
            4: "Airplane",
        }
        self.yolo_multi_label = False
        self.yolo_max_detection = 500
        self.device = "cpu"
        self.traffic_light_set_view = 0
        self.traffic_light_post_processing = 0
        self.traffic_light_area = [0, 20, 100, 300]
        self.traffic_light_red_light = {
            "h_min": 150,
            "h_max": 180,
            "s_min": 70,
            "s_max": 255,
            "v_min": 50,
            "v_max": 255,
            "threshold": 14,
        }

        self.traffic_light_green_light = {
            "h_min": 36,
            "h_max": 70,
            "s_min": 25,
            "s_max": 255,
            "v_min": 25,
            "v_max": 255,
            "threshold": 13,
        }

        self.draw_bounding_boxes = True
        self.show_label_and_confedence = True
        self.use_tracking = "SORT"  # {"No Tracker", "Deep SORT", "SORT"}

        # Need this if you will use deep_sort
        self.config_deepsort = "configs/deep_sort.yaml"
        self.deepsort_use_cuda = False

        # Sort
