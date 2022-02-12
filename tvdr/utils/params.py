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
        self.device = "cpu"
        self.video_path = ""

        # YOLO Vehicle Detection Params
        self.yolo_model_path = ""
        self.yolo_imgsz = 1080
        self.yolo_stride = "2"
        self.yolo_conf = 0.25
        self.yolo_iou = 0.45
        self.yolo_classes = [2, 3, 5, 7]
        self.yolo_classes_name = {
            0: "Car",
            1: "Motorcycle",
            2: "Bus",
            3: "Truck",
            4: "Bicycle",
        }
        self.yolo_multi_label = False
        self.yolo_max_detection = 500

        # Traffic Light Params
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

        # General
        self.show_bounding_boxes = True
        self.show_label_and_confedence = True
        self.show_detection_area = True
        self.show_stopline = True

        # Downstream task Params
        self.detect_helmet_violation = True
        self.detect_running_redlight_violation = True
        self.detect_wrongway_violation = True

        # Tracking Params
        self.use_tracking = "SORT"  # {"No Tracker", "Deep SORT", "SORT"}

        # DeepSORT Params
        self.deepsort_model_path = "models/ckpt.t7"
        self.deepsort_max_dist = 0.2
        self.deepsort_min_confidence = 0.3
        self.deepsort_max_iou_distance = 0.7
        self.deepsort_max_age = 70
        self.deepsort_n_init = 3
        self.deepsort_nn_budget = 100
        self.deepsort_use_cuda = False

        # SORT Params
        self.sort_min_hits = 3
        self.sort_max_age = 80
        self.sort_iou_threshold = 0.3

        # Detection Area and Stop Line Params
        self.detection_area = []
        self.stopline = []

        # Wrong Way Params
        self.wrongway_direction_degree = 90
        self.wrongway_threshold_degree = 20
        self.wrongway_miss_count = 5  # In second
        self.wrongway_min_value = 50

        # Helmet Violations Params
        self.hv_model_path = ""
        self.hv_imgsz = 128
        self.hv_conf = 0.25
        self.hv_iou = 0.45
        self.hv_min_age = 3
        self.hv_pad_width_mul = 1.5
        self.hv_pad_height_mul = 1.5
        self.hv_detect_interval = 5
