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


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2.QtCore import QLine, Qt, QPoint, QRect
from PySide2.QtGui import QPainter, QPen, QBrush, QIcon
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
from tvdr.core import TrafficLightDetection, traffic_light

import numpy as np


class TrafficLight(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.main_layout = QtWidgets.QGridLayout()
        self.config_layout = self.configuration_layout()
        self.image_layout = self.image_layout_config()

        self.main_layout.addLayout(self.config_layout, 0, 0, 4, 1, Qt.AlignLeft)
        self.main_layout.addLayout(self.image_layout, 0, 1, 4, 3, Qt.AlignHCenter)
        self.main_layout.setSizeConstraint(QLayout.SetFixedSize)

    def set_init_parameters(self):
        self.checkbox_area_all.setChecked(True)
        self.checkbox_image_rgb.setChecked(True)

        red_light_hsv_params = self.parameter.traffic_light_red_light
        self.red_h_min.setValue(red_light_hsv_params["h_min"])
        self.red_h_max.setValue(red_light_hsv_params["h_max"])
        self.red_s_min.setValue(red_light_hsv_params["s_min"])
        self.red_s_max.setValue(red_light_hsv_params["s_max"])
        self.red_v_min.setValue(red_light_hsv_params["v_min"])
        self.red_v_max.setValue(red_light_hsv_params["v_max"])
        self.red_min_pixel.setValue(red_light_hsv_params["threshold"])

        green_light_hsv_params = self.parameter.traffic_light_green_light
        self.green_h_min.setValue(green_light_hsv_params["h_min"])
        self.green_h_max.setValue(green_light_hsv_params["h_max"])
        self.green_s_min.setValue(green_light_hsv_params["s_min"])
        self.green_s_max.setValue(green_light_hsv_params["s_max"])
        self.green_v_min.setValue(green_light_hsv_params["v_min"])
        self.green_v_max.setValue(green_light_hsv_params["v_max"])
        self.green_min_pixel.setValue(green_light_hsv_params["threshold"])

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        height_target = 650
        divider = height_target / h
        w_new = int(divider * w)
        p = convert_to_Qt_format.scaled(w_new, height_target, Qt.KeepAspectRatio)
        return p

    def show(self, parameter):
        self.parameter = parameter
        self.set_init_parameters()
        self.read_video_init()
        self.tld = TrafficLightDetection(self.parameter)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setLayout(self.main_layout)
        self.exec_()

    def read_video_init(self):
        self.vid = cv2.VideoCapture(self.parameter.video_path)
        frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.slider_frame.setMaximum(int(frame_count))

        ret, frame = self.vid.read()

        image = self.convert_cv_qt(frame)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(image))
        pass

    def update_frame(self):
        frame_position = self.slider_frame.value()

        # get current frame from videos
        self.vid.set(1, frame_position)
        ret, self.current_frame = self.vid.read()

        # get parameter traffic light frame position
        index_crop = self.parameter.traffic_light_area

        # croping frame to get traffic light frame
        self.traffic_light_frame = self.current_frame[
            index_crop[1] : index_crop[3], index_crop[0] : index_crop[2]
        ]
        self.traffic_light_frame = np.ascontiguousarray(self.traffic_light_frame)

        # processing to get color of traffic light
        status = self.tld.detect_state(self.traffic_light_frame)

        # print processing result
        self.segmentation_status_label.setText(
            "Traffic Light Status \t: {}".format(status)
        )
        self.greenlight_label.setText(
            "Green Light Pixel \t\t: {}".format(self.tld.green_light_count)
        )
        self.redlight_label.setText(
            "Red Light Pixel \t\t: {}".format(self.tld.red_light_count)
        )
        if self.parameter.traffic_light_set_view == 1:

            if self.parameter.traffic_light_post_processing == 1:
                if (
                    self.traffic_light_segmentation_combobox.currentText()
                    == "Red Light"
                ):
                    self.image_data_to_show = self.tld.get_red_light_segmentation()
                else:
                    self.image_data_to_show = self.tld.get_green_light_segmentation()

            else:
                self.image_data_to_show = self.traffic_light_frame

        else:
            self.image_data_to_show = self.current_frame

        image = self.convert_cv_qt(self.image_data_to_show)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(image))

    def image_layout_config(self):
        v_layout = QtWidgets.QVBoxLayout()
        self.image_frame = QtWidgets.QLabel()
        self.image = cv2.imread("samples/file-20200803-24-50u91u.jpg")
        self.image = QtGui.QImage(
            self.image.data,
            self.image.shape[1],
            self.image.shape[0],
            QtGui.QImage.Format_RGB888,
        ).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.slider_frame = QtWidgets.QSlider()
        self.slider_frame.setOrientation(Qt.Horizontal)
        self.slider_frame.setMaximum(1)
        self.slider_frame.setMinimum(0)
        self.slider_frame.setTickInterval(1)
        self.slider_frame.valueChanged.connect(self.update_frame)
        v_layout.addWidget(self.image_frame)
        v_layout.addLayout(self.set_segmentation_result())
        v_layout.addWidget(self.slider_frame)
        return v_layout

    def configuration_layout(self):
        v_layout = QtWidgets.QVBoxLayout()
        v_red_layout = QtWidgets.QGridLayout()
        red_light_gb = QtWidgets.QGroupBox("Red Light")
        self.red_h_min = QtWidgets.QDoubleSpinBox()
        self.red_h_min.setMinimum(0)
        self.red_h_min.setMaximum(180)
        self.red_h_min.setSingleStep(1)

        self.red_h_max = QtWidgets.QDoubleSpinBox()
        self.red_h_max.setMinimum(0)
        self.red_h_max.setMaximum(180)
        self.red_h_max.setSingleStep(1)

        self.red_s_min = QtWidgets.QDoubleSpinBox()
        self.red_s_min.setMinimum(0)
        self.red_s_min.setMaximum(255)
        self.red_s_min.setSingleStep(1)

        self.red_s_max = QtWidgets.QDoubleSpinBox()
        self.red_s_max.setMinimum(0)
        self.red_s_max.setMaximum(255)
        self.red_s_max.setSingleStep(1)

        self.red_v_min = QtWidgets.QDoubleSpinBox()
        self.red_v_min.setMinimum(0)
        self.red_v_min.setMaximum(255)
        self.red_v_min.setSingleStep(1)

        self.red_v_max = QtWidgets.QDoubleSpinBox()
        self.red_v_max.setMinimum(0)
        self.red_v_max.setMaximum(255)
        self.red_v_max.setSingleStep(1)

        self.red_min_pixel = QtWidgets.QDoubleSpinBox()
        self.red_min_pixel.setMinimum(0)
        self.red_min_pixel.setMaximum(2500)
        self.red_min_pixel.setSingleStep(1)

        v_red_layout.addWidget(QLabel("H Min"), 0, 0)
        v_red_layout.addWidget(self.red_h_min, 0, 1)
        v_red_layout.addWidget(QLabel("H Max"), 1, 0)
        v_red_layout.addWidget(self.red_h_max, 1, 1)
        v_red_layout.addWidget(QLabel("S Min"), 2, 0)
        v_red_layout.addWidget(self.red_s_min, 2, 1)
        v_red_layout.addWidget(QLabel("S Max"), 3, 0)
        v_red_layout.addWidget(self.red_s_max, 3, 1)
        v_red_layout.addWidget(QLabel("V Min"), 4, 0)
        v_red_layout.addWidget(self.red_v_min, 4, 1)
        v_red_layout.addWidget(QLabel("V Max"), 5, 0)
        v_red_layout.addWidget(self.red_v_max, 5, 1)
        v_red_layout.addWidget(QLabel("Threshold"), 6, 0)
        v_red_layout.addWidget(self.red_min_pixel, 6, 1)

        v_green_layout = QtWidgets.QGridLayout()
        green_light_gb = QtWidgets.QGroupBox("Green Light")
        self.green_h_min = QtWidgets.QDoubleSpinBox()
        self.green_h_min.setMinimum(0)
        self.green_h_min.setMaximum(180)
        self.green_h_min.setSingleStep(1)

        self.green_h_max = QtWidgets.QDoubleSpinBox()
        self.green_h_max.setMinimum(0)
        self.green_h_max.setMaximum(180)
        self.green_h_max.setSingleStep(1)

        self.green_s_min = QtWidgets.QDoubleSpinBox()
        self.green_s_min.setMinimum(0)
        self.green_s_min.setMaximum(255)
        self.green_s_min.setSingleStep(1)

        self.green_s_max = QtWidgets.QDoubleSpinBox()
        self.green_s_max.setMinimum(0)
        self.green_s_max.setMaximum(255)
        self.green_s_max.setSingleStep(1)

        self.green_v_min = QtWidgets.QDoubleSpinBox()
        self.green_v_min.setMinimum(0)
        self.green_v_min.setMaximum(255)
        self.green_v_min.setSingleStep(1)

        self.green_v_max = QtWidgets.QDoubleSpinBox()
        self.green_v_max.setMinimum(0)
        self.green_v_max.setMaximum(255)
        self.green_v_max.setSingleStep(1)

        self.green_min_pixel = QtWidgets.QDoubleSpinBox()
        self.green_min_pixel.setMinimum(0)
        self.green_min_pixel.setMaximum(2500)
        self.green_min_pixel.setSingleStep(1)

        v_green_layout.addWidget(QLabel("H Min"), 0, 0)
        v_green_layout.addWidget(self.green_h_min, 0, 1)
        v_green_layout.addWidget(QLabel("H Max"), 1, 0)
        v_green_layout.addWidget(self.green_h_max, 1, 1)
        v_green_layout.addWidget(QLabel("S Min"), 2, 0)
        v_green_layout.addWidget(self.green_s_min, 2, 1)
        v_green_layout.addWidget(QLabel("S Max"), 3, 0)
        v_green_layout.addWidget(self.green_s_max, 3, 1)
        v_green_layout.addWidget(QLabel("V Min"), 4, 0)
        v_green_layout.addWidget(self.green_v_min, 4, 1)
        v_green_layout.addWidget(QLabel("V Max"), 5, 0)
        v_green_layout.addWidget(self.green_v_max, 5, 1)
        v_green_layout.addWidget(QLabel("Threshold"), 6, 0)
        v_green_layout.addWidget(self.green_min_pixel, 6, 1)

        red_light_gb.setLayout(v_red_layout)
        green_light_gb.setLayout(v_green_layout)

        self.set_area_button = QtWidgets.QPushButton("Set Area")
        self.set_area_button.clicked.connect(self.set_area)

        self.save_button = QtWidgets.QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.save_configuration)

        # Set Image Frame Configuration Area of Traffic Light or Scenes
        h_groupbox_cropped = QGroupBox("Set View")
        h_layout_cropped = QHBoxLayout()
        self.checkbox_area_traffic_light = QCheckBox("Traffic Light Area")
        self.checkbox_area_all = QCheckBox("All Area")

        self.checkbox_area_traffic_light.clicked.connect(
            lambda: self.btnstate_setview(self.checkbox_area_traffic_light)
        )
        self.checkbox_area_all.clicked.connect(
            lambda: self.btnstate_setview(self.checkbox_area_all)
        )

        h_layout_cropped.addWidget(self.checkbox_area_all)
        h_layout_cropped.addWidget(self.checkbox_area_traffic_light)
        h_groupbox_cropped.setLayout(h_layout_cropped)

        # Set Image Processing
        self.traffic_light_segmentation_combobox = QComboBox()
        self.traffic_light_segmentation_combobox.addItem("Red Light")
        self.traffic_light_segmentation_combobox.addItem("Green Light")
        self.traffic_light_segmentation_combobox.currentTextChanged.connect(
            self.apply_parameters
        )

        h_groupbox_image_post_processing = QGroupBox("Set Image")
        v_layout_image_post_processing = QVBoxLayout()
        h_layout_image_post_processing = QHBoxLayout()
        self.checkbox_image_rgb = QCheckBox("RGB")
        self.checkbox_image_segmentation = QCheckBox("Segmentation")

        self.checkbox_image_rgb.clicked.connect(
            lambda: self.btnstate_setimg(self.checkbox_image_rgb)
        )
        self.checkbox_image_segmentation.clicked.connect(
            lambda: self.btnstate_setimg(self.checkbox_image_segmentation)
        )
        h_layout_image_post_processing.addWidget(self.checkbox_image_rgb)
        h_layout_image_post_processing.addWidget(self.checkbox_image_segmentation)
        v_layout_image_post_processing.addLayout(h_layout_image_post_processing)
        v_layout_image_post_processing.addWidget(
            self.traffic_light_segmentation_combobox
        )

        self.traffic_light_segmentation_combobox.hide()
        h_groupbox_image_post_processing.setLayout(v_layout_image_post_processing)

        # Set Button Apply Parameters
        self.button_apply_paramaters = QPushButton("Apply Parameters")
        self.button_apply_paramaters.clicked.connect(self.apply_parameters)

        v_layout.addWidget(red_light_gb)
        v_layout.addWidget(green_light_gb)
        v_layout.addWidget(self.button_apply_paramaters)
        v_layout.addWidget(h_groupbox_cropped)
        v_layout.addWidget(h_groupbox_image_post_processing)
        v_layout.addWidget(self.set_area_button)
        v_layout.addWidget(self.save_button)

        return v_layout

    def btnstate_setview(self, b):
        if b.text() == "All Area":
            self.checkbox_area_all.setChecked(True)
            self.checkbox_area_traffic_light.setChecked(False)
            self.parameter.traffic_light_set_view = 0
        else:
            self.checkbox_area_all.setChecked(False)
            self.checkbox_area_traffic_light.setChecked(True)
            self.parameter.traffic_light_set_view = 1

        self.update_frame()

    def btnstate_setimg(self, b):
        if b.text() == "RGB":
            self.checkbox_image_rgb.setChecked(True)
            self.checkbox_image_segmentation.setChecked(False)
            self.parameter.traffic_light_post_processing = 0
            self.traffic_light_segmentation_combobox.hide()
        else:
            self.checkbox_image_rgb.setChecked(False)
            self.checkbox_image_segmentation.setChecked(True)
            self.parameter.traffic_light_post_processing = 1
            self.traffic_light_segmentation_combobox.show()

        self.update_frame()

    def set_area(self):
        ret, self.current_frame = self.vid.read()
        self.traffic_light_area = TrafficLightArea(self.current_frame)
        self.traffic_light_area.exec_()
        if self.traffic_light_area.result() == 1:
            self.parameter.traffic_light_area = self.traffic_light_area.getArea()
            print(self.parameter.traffic_light_area)
            self.update_frame()

    def apply_parameters(self):
        red_light_params = {}
        red_light_params["h_min"] = self.red_h_min.value()
        red_light_params["h_max"] = self.red_h_max.value()
        red_light_params["s_min"] = self.red_s_min.value()
        red_light_params["s_max"] = self.red_s_max.value()
        red_light_params["v_min"] = self.red_v_min.value()
        red_light_params["v_max"] = self.red_v_max.value()
        red_light_params["threshold"] = self.red_min_pixel.value()

        green_light_params = {}
        green_light_params["h_min"] = self.green_h_min.value()
        green_light_params["h_max"] = self.green_h_max.value()
        green_light_params["s_min"] = self.green_s_min.value()
        green_light_params["s_max"] = self.green_s_max.value()
        green_light_params["v_min"] = self.green_v_min.value()
        green_light_params["v_max"] = self.green_v_max.value()
        green_light_params["threshold"] = self.green_min_pixel.value()

        self.parameter.traffic_light_red_light = red_light_params
        self.parameter.traffic_light_green_light = green_light_params

        self.tld.update_parameters(self.parameter)
        self.update_frame()

    def set_segmentation_result(self):
        segmentation_detail_layout = QVBoxLayout()
        segmentation_detail_layout.setMargin(0)
        segmentation_detail_layout.setSpacing(0)

        self.segmentation_status_label = QLabel("Traffic Light Status \t: Undefined")
        self.greenlight_label = QLabel("Green Light Pixel \t\t: 0")
        self.redlight_label = QLabel("Red Light Pixel \t\t: 0")

        segmentation_detail_layout.addWidget(self.segmentation_status_label)
        segmentation_detail_layout.addWidget(self.greenlight_label)
        segmentation_detail_layout.addWidget(self.redlight_label)

        return segmentation_detail_layout

    def save_configuration(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText("Do you want to save this traffic light parameters ?")
        msg.setDetailedText(
            f"""
        The details paramaters are as follows:
        Red Light
        H Min : {self.parameter.traffic_light_red_light['h_min']}
        H Max : {self.parameter.traffic_light_red_light['h_max']}
        S Min : {self.parameter.traffic_light_red_light['s_min']}
        S_Max : {self.parameter.traffic_light_red_light['s_max']}
        V Min : {self.parameter.traffic_light_red_light['v_min']}
        V Max : {self.parameter.traffic_light_red_light['v_max']}
        Threshold :{self.parameter.traffic_light_red_light['threshold']}

        Green Light:
        H Min : {self.parameter.traffic_light_green_light['h_min']}
        H Max : {self.parameter.traffic_light_green_light['h_max']}
        S Min : {self.parameter.traffic_light_green_light['s_min']}
        S Max : {self.parameter.traffic_light_green_light['s_max']}
        V Min : {self.parameter.traffic_light_green_light['v_min']}
        V Max : {self.parameter.traffic_light_green_light['v_max']}
        Threshold : {self.parameter.traffic_light_green_light['threshold']}

        Traffic Light Area
        X Begin : {self.parameter.traffic_light_area[0]}
        Y Begin : {self.parameter.traffic_light_area[1]}
        X End   : {self.parameter.traffic_light_area[2]}
        Y End   : {self.parameter.traffic_light_area[3]}
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


class TrafficLightArea(QtWidgets.QDialog):
    def __init__(self, current_frame):
        super().__init__()

        self.x_begin = 0
        self.y_begin = 0
        self.x_end = 0
        self.y_end = 0

        self.v_layout = QVBoxLayout()
        self.v_layout.setMargin(0)
        self.v_layout.setSpacing(0)

        self.image_label = QLabel()

        self.current_frame = current_frame

        # self.image = cv2.imread("samples/file-20200803-24-50u91u.jpg")
        self.image = QtGui.QImage(
            self.current_frame,
            self.current_frame.shape[1],
            self.current_frame.shape[0],
            QtGui.QImage.Format_RGB888,
        ).rgbSwapped()

        self.qimage = QtGui.QPixmap.fromImage(self.image)
        self.image_label.setPixmap(self.qimage)

        self.setGeometry(100, 100, 500, 300)
        self.resize(self.image.width(), self.image.height())

        self.begin = QPoint()
        self.end = QPoint()

        self.last_begin = None
        self.last_end = None

        self.rectangles = []

        self.v_layout.addWidget(self.image_label)
        self.setLayout(self.v_layout)

        self.setWindowModality(QtCore.Qt.ApplicationModal)

    def paintEvent(self, event):
        qimage = QtGui.QPixmap.fromImage(self.image)
        self.qimage2 = qimage

        qp = QPainter(self.qimage2)
        qp.setPen(QPen(Qt.red, 1, Qt.SolidLine))

        if not self.begin.isNull() and not self.end.isNull():
            qp.drawRect(QRect(self.begin, self.end).normalized())
            self.last_begin = self.begin
            self.last_end = self.end
            self.image_label.setPixmap(self.qimage2)

    def mousePressEvent(self, event):
        self.begin = self.end = event.pos()
        self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        r = QRect(self.begin, self.end).normalized()
        print(r)
        self.rectangles.append(r)
        # self.begin = self.end = QPoint()
        self.update()
        super().mouseReleaseEvent(event)
        print("Release Rectangles")

        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText("Do you want to save this traffic area?")
        msg.setDetailedText(
            "The details are as follows: \nX Begin= {} \nY Begin = {}, \nX End = {}, \nY End = {}".format(
                self.begin.x(), self.begin.y(), self.end.x(), self.end.y()
            )
        )
        msg.setStandardButtons(
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel
        )
        response = msg.exec_()

        if response == QtWidgets.QMessageBox.Save:
            logging.info("Save Traffic Light Parameters")
            self.x_begin = (
                self.begin.x() if self.begin.x() < self.end.x() else self.end.x()
            )
            self.y_begin = (
                self.begin.y() if self.begin.y() < self.end.y() else self.end.y()
            )

            self.x_end = (
                self.end.x() if self.begin.x() < self.end.x() else self.begin.x()
            )
            self.y_end = (
                self.end.y() if self.begin.y() < self.end.y() else self.begin.y()
            )

            self.accept()

        elif response == QtWidgets.QMessageBox.Cancel:
            logging.info("Cancel Save Traffic Light Parameters")

    def getArea(self):
        return [self.x_begin, self.y_begin, self.x_end, self.y_end]
