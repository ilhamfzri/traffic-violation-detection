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

from typing import List
import cv2
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2.QtCore import QLine, Qt, QPoint, QRect
from PySide2.QtGui import QPainter, QPen, QBrush, QIcon, QPixmap
from PySide2.QtWidgets import QLabel, QLayout, QMessageBox
from PySide2 import QtGui
import qtawesome as qta

import numpy as np


class PainterArea(QLabel):
    def __init__(
        self,
        parent=None,
    ):
        QLabel.__init__(self, parent)
        self.list_point = []

    def set_ratio_img(self, ratio):
        self.ratio_img = ratio

    def set_point(self, list_point=[]):
        self.list_point = list_point
        self.draw_area()

    def set_frame(self, cv_img):
        self.cv_img = cv_img
        self.ori_shape = cv_img.shape
        self.qimage = self.convert_cv_qt(cv_img)
        self.qpixmap_data = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.qpixmap_data)

    def clear_point(self):
        self.list_point = []
        self.update_point_viewer()

    def draw_area(self):
        if len(self.list_point) > 1:
            points = np.empty((0, 1, 2))
            for i, point in enumerate(self.list_point):
                point_arr = np.array([[point.x(), point.y()]]).reshape(1, 1, 2)
                points = np.append(points, point_arr, axis=0)

            points = points.astype(np.int32)

            shapes = self.cv_img.copy()
            shapes = cv2.drawContours(shapes, [points], -1, (0, 255, 0), 2)

            self.qimage = self.convert_cv_qt(shapes)
            self.qpixmap_data = QtGui.QPixmap.fromImage(self.qimage)

            self.setPixmap(self.qpixmap_data)

    def get_contour(self):
        point_detection_area = []
        height, width, _ = self.ori_shape
        for point in self.list_point:
            point_x = round(point.x() / width, 3)
            point_y = round(point.y() / height, 3)
            point_detection_area.append([point_x, point_y])
        return point_detection_area

    def update_point_viewer(self):
        image_new = self.cv_img.copy()
        for i, point in enumerate(self.list_point):
            image_new = cv2.circle(
                image_new,
                (point.x(), point.y()),
                radius=1,
                color=(0, 0, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

            image_new = cv2.putText(
                image_new,
                f"{i}",
                org=(point.x() - 5, point.y() - 10),
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                color=(0, 0, 255),
                thickness=1,
                fontScale=0.5,
                lineType=cv2.LINE_AA,
            )

        self.qimage = self.convert_cv_qt(image_new)
        self.qpixmap_data = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.qpixmap_data)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data,
            w,
            h,
            bytes_per_line,
            QtGui.QImage.Format_RGB888,
        )

        h_new = h / self.ratio_img
        w_new = w / self.ratio_img

        p = convert_to_Qt_format.scaled(w_new, h_new, Qt.KeepAspectRatio)
        return p

    def mousePressEvent(self, event):
        self.current_pos = event.pos()
        point = event.pos()
        point_new = QtCore.QPoint(
            point.x() * self.ratio_img, point.y() * self.ratio_img
        )
        self.list_point.append(point_new)
        self.update_point_viewer()
        super().mousePressEvent(event)


class DetectionArea(QtWidgets.QDialog):
    def __init__(self, video_path: str, current_detection_area=[]):
        super().__init__()
        main_v_layout = QtWidgets.QVBoxLayout()
        main_h_layout = QtWidgets.QHBoxLayout()

        # Set Window Width
        self.size_window_h = 600
        self.setWindowTitle("Detection Area Configuration")

        # Read Video Data and Information
        self.vid = cv2.VideoCapture(video_path)
        self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        _, self.frame_data = self.vid.read()

        # Current Detection Area
        list_point = []
        height, width, _ = self.frame_data.shape
        for point in current_detection_area:
            point_x, point_y = point
            point_x = int(point_x * width)
            point_y = int(point_y * height)
            list_point.append(QPoint(point_x, point_y))

        # Set Mouse Position Tracking and Video Resolution
        h_layout_video_size_and_pointer = QtWidgets.QHBoxLayout()
        self.frame_size_information = QtWidgets.QLabel(
            f"Resolution : {self.frame_data.shape}"
        )
        self.ratio = self.frame_data.shape[0] / self.size_window_h
        h_layout_video_size_and_pointer.addWidget(self.frame_size_information)

        # Set Slider
        self.slider_video = QtWidgets.QSlider()
        self.slider_video.setMinimum(0)
        self.slider_video.setMaximum(self.frame_count - 1)
        self.slider_video.setSingleStep(1)
        self.slider_video.setOrientation(Qt.Horizontal)
        self.slider_video.valueChanged.connect(self.update_frame)

        # Set Image
        self.image_label = PainterArea(self)
        self.image_label.set_ratio_img(self.ratio)
        self.image_label.set_frame(self.frame_data)
        self.image_label.set_point(list_point)

        # Set Button
        self.set_area_button = QtWidgets.QPushButton(qta.icon("mdi.draw"), "Draw Area")
        self.set_area_button.clicked.connect(self.image_label.draw_area)
        self.clear_area_button = QtWidgets.QPushButton(qta.icon("fa5s.eraser"), "Clear")
        self.clear_area_button.clicked.connect(self.image_label.clear_point)
        self.save_button = QtWidgets.QPushButton(qta.icon("fa.check-circle-o"), "Done")
        self.save_button.clicked.connect(self.save_configuration)

        # Set Horizontal Top Bar Menu
        main_h_layout.addWidget(self.slider_video)
        main_h_layout.addWidget(self.set_area_button)
        main_h_layout.addWidget(self.clear_area_button)
        main_h_layout.addWidget(self.save_button)
        main_h_layout.setSizeConstraint(QLayout.SetFixedSize)

        # Set Main Menu Layout
        main_v_layout.addLayout(main_h_layout)
        main_v_layout.addWidget(self.image_label)
        main_v_layout.addLayout(h_layout_video_size_and_pointer)
        main_v_layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(main_v_layout)

        self.setWindowModality(QtCore.Qt.ApplicationModal)

    def update_frame(self):
        frame_position = self.slider_video.value()

        self.vid.set(1, frame_position)
        _, self.frame_data = self.vid.read()

        self.image_label.set_frame(self.frame_data)
        self.image_label.draw_area()

    def save_configuration(self):
        if len(self.image_label.get_contour()) < 3:
            msg = QMessageBox.critical(
                self,
                "",
                "You need atleast 3 points!",
            )
        else:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Question)
            msg.setText("Do you want to save this detection area ?")
            msg.setDetailedText(
                f"""The details paramaters are as follows:\n
            {self.image_label.get_contour()}
            """
            )

            msg.setStandardButtons(
                QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel
            )
            response = msg.exec_()

            if response == QtWidgets.QMessageBox.Save:
                self.accept()

            elif response == QtWidgets.QMessageBox.Cancel:
                pass

    def get_detection_area(self):
        return self.image_label.get_contour()
