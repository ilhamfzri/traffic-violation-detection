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

from yolov5.utils.datasets import LoadImages
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.general import check_img_size
from yolov5.models.experimental import attempt_load
from pathlib import Path

import cv2
import numpy as nps
import torch
import time


class YOLOModel:
    def __init__(self, weight="yolov5s.pt"):
        # self.load_model(weight)

        # This parameters will update soon
        # self.imgsz = 640
        pass

    def load_model(self, weights):
        self.device = select_device("")
        self.model = attempt_load(weights)
        self.stride = int(self.model.stride.max())
        self.imgsz = check_img_size(self.params.imgsz, s=self.stride)
        self.half = self.device.type != "cpu"
        self.names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )

        if self.half:
            self.model.half()  # to FP16

    def update_parameters(self, conf_thres, iou_thres, imgsz):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz

    def print_yolo_parameters(self):
        pass

    def check_before_inference(self):
        pass

    def inference(self, video_path):
        self.dataset = LoadImages(
            path=video_path, img_size=self.imgsz, stride=self.stride
        )

        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once

        t0 = time.time()

        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = time_sync()
            pred = self.model(img)[0]

            pred = non_max_suppression(
                pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres
            )
            t2 = time_sync()

            for i, det in enumerate(pred):
                p, s, im0, frame = (
                    path,
                    "",
                    im0s.copy(),
                    getattr(self.dataset, "frame", 0),
                )

                p = Path(p)  # to Path
                save_path = str(self.params.output_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / "labels" / p.stem) + (
                    "" if self.dataset.mode == "image" else f"_{frame}"
                )  # img.txt

                s += "%gx%g " % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                # if len(det):
                #     det[:, :4] = scale_coords(
                #         img.shape[2:], det[:, :4], im0.shape
                #     ).round()

                #     for c in det[:, -1].unique():
                #         n = (det[:, -1] == c).sum()  # detections per class
                #         s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #     for *xyxy, conf, cls in reversed(det):
                #         if self.params.save_txt:  # Write to file
                #             xywh = (
                #                 (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                #                 .view(-1)
                #                 .tolist()
                #             )  # normalized xywh
                #             line = (
                #                 (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                #             )  # label format
                #             with open(txt_path + ".txt", "a") as f:
                #                 f.write(("%g " * len(line)).rstrip() % line + "\n")

                #         if save_img or opt.save_crop or view_img:  # Add bbox to image
                #             c = int(cls)  # integer class
                #             label = (
                #                 None
                #                 if opt.hide_labels
                #                 else (
                #                     names[c]
                #                     if opt.hide_conf
                #                     else f"{names[c]} {conf:.2f}"
                #                 )
                #             )

                #             plot_one_box(
                #                 xyxy,
                #                 im0,
                #                 label=label,
                #                 color=colors(c, True),
                #                 line_thickness=opt.line_thickness,
                #             )
                #             if opt.save_crop:
                #                 save_one_box(
                #                     xyxy,
                #                     im0s,
                #                     file=save_dir
                #                     / "crops"
                #                     / names[c]
                #                     / f"{p.stem}.jpg",
                #                     BGR=True,
                #                 )

    def load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, frame = cap.read()
            print(frame)
            #     cv2.imshow("frame", frame)
        #     if cv2.waitKey(1) & 0xFF == ord("q"):
        #         break
        # cap.release()
        # cv2.destroyAllWindows()
