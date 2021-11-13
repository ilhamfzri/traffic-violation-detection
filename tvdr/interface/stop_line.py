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
import logging

import sys

from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2.QtCore import QLine, Qt, QPoint, QRect
from PySide2.QtGui import QPainter, QPen, QBrush, QIcon, QPixmap
from PySide2.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PySide2 import QtGui
import numpy as np

from tvdr.utils import image


class PainterLine(QLabel):
    def __init__(self, parent=None):
        QLabel.__init__(self, parent)
        self.list_point = []
        self.current_pos = QPoint()
        self.setText("This text does not appear")

    def set_frame(self, cv_img):
        self.cv_img = cv_img
        self.qimage = self.convert_cv_qt(cv_img)
        self.qpixmap_data = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.qpixmap_data)

    def set_point(self, list_point=[]):
        self.list_point = list_point
        self.draw_line()

    def clear_point(self):
        self.list_point = []
        self.update_point_viewer()

    def draw_line(self):
        if len(self.list_point) == 2:

            start_point = (self.list_point[0].x(), self.list_point[0].y())
            end_point = (self.list_point[1].x(), self.list_point[1].y())

            image_data = self.cv_img.copy()
            image_data = cv2.line(
                image_data, start_point, end_point, (0, 0, 255), 2, cv2.LINE_AA
            )

            self.qimage = self.convert_cv_qt(image_data)
            self.qpixmap_data = QtGui.QPixmap.fromImage(self.qimage)

            self.setPixmap(self.qpixmap_data)

    def get_stopline(self):
        points = np.empty((0, 1, 2))
        for i, point in enumerate(self.list_point):
            point_arr = np.array([[point.x(), point.y()]]).reshape(1, 1, 2)
            points = np.append(points, point_arr, axis=0)

        points = points.astype(np.int32)
        return points

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
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        return convert_to_Qt_format

    def mousePressEvent(self, event):
        point = event.pos()
        if len(self.list_point) < 2:
            self.list_point.append(point)

        self.update_point_viewer()
        super().mousePressEvent(event)


class StopLine(QtWidgets.QDialog):
    def __init__(self, video_path: str, current_stopline=[]):
        super().__init__()
        main_v_layout = QtWidgets.QVBoxLayout()
        main_h_layout = QtWidgets.QHBoxLayout()

        list_point = []
        for point in current_stopline:
            list_point.append(QPoint(point[0][0], point[0][1]))

        print(list_point)

        # Read Video Data and Information
        self.vid = cv2.VideoCapture(video_path)
        self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        _, self.frame_data = self.vid.read()

        # Set Mouse Position Tracking and Video Resolution
        h_layout_video_size_and_pointer = QtWidgets.QHBoxLayout()
        self.frame_size_information = QtWidgets.QLabel(
            f"Resolution : {self.frame_data.shape}"
        )
        # self.mouse_pos_label = QtWidgets.QLabel(f"Mouse Position : (0,0)")
        h_layout_video_size_and_pointer.addWidget(self.frame_size_information)
        # h_layout_video_size_and_pointer.addWidget(self.mouse_pos_label)

        # Set Slider
        self.slider_video = QtWidgets.QSlider()
        self.slider_video.setMinimum(0)
        self.slider_video.setMaximum(self.frame_count)
        self.slider_video.setSingleStep(1)
        self.slider_video.setOrientation(Qt.Horizontal)
        self.slider_video.valueChanged.connect(self.update_frame)

        # Set Image
        self.image_label = PainterLine(self)
        self.image_label.set_frame(self.frame_data)
        self.image_label.set_point(list_point)

        # Set Button
        self.set_area_button = QtWidgets.QPushButton("Draw Line")
        self.set_area_button.clicked.connect(self.image_label.draw_line)
        self.clear_area_button = QtWidgets.QPushButton("Clear")
        self.clear_area_button.clicked.connect(self.image_label.clear_point)
        self.save_button = QtWidgets.QPushButton("Done")
        self.save_button.clicked.connect(self.save_configuration)

        # Set Horizontal Top Bar Menu
        main_h_layout.addWidget(self.slider_video)
        main_h_layout.addWidget(self.set_area_button)
        main_h_layout.addWidget(self.clear_area_button)
        main_h_layout.addWidget(self.save_button)
        main_h_layout.setSizeConstraint(QLayout.SetFixedSize)

        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.update_mouse_pos)
        # self.timer.start(1000.0 / 30)

        # Set Main Menu Layout
        main_v_layout.addLayout(main_h_layout)
        main_v_layout.addWidget(self.image_label)
        main_v_layout.addLayout(h_layout_video_size_and_pointer)
        main_v_layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(main_v_layout)

        self.setWindowModality(QtCore.Qt.ApplicationModal)

    # def update_mouse_pos(self):
    #     self.mouse_pos_label.setText(
    #         f"Mouse Position : ({self.image_label.current_pos.x()},{self.image_label.current_pos.y()})"
    #     )

    def update_frame(self):
        frame_position = self.slider_video.value()

        self.vid.set(1, frame_position)
        _, self.frame_data = self.vid.read()

        self.image_label.set_frame(self.frame_data)
        self.image_label.draw_line()

    def save_configuration(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText("Do you want to save this stop lines ?")
        msg.setDetailedText(
            f"""The details paramaters are as follows:\n
        {self.image_label.get_stopline()}
        """
        )

        msg.setStandardButtons(
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel
        )
        response = msg.exec_()

        if response == QtWidgets.QMessageBox.Save:
            print("LOG : Save Traffic Lights Parameters")
            self.accept()

        elif response == QtWidgets.QMessageBox.Cancel:
            print("LOG : Cancel")

    def get_stopline(self):
        return self.image_label.get_stopline()
