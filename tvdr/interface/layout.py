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

from PySide2 import QtCore
import cv2
import qimage2ndarray
import time
import numpy as np

from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import QTimer, QWaitCondition, Qt, Slot
from tvdr.utils import Parameter
from tvdr.core import YOLOModel, TrafficLightDetection
from tvdr.interface.traffic_light import TrafficLight


class MainLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.parameter = Parameter()

        # Initialize YOLO
        self.yolo = YOLOModel(device="cpu")

        # Initialize Traffic Light State Detection
        self.tld = TrafficLightDetection(self.parameter)

        # self.layout = QtWidgets.QFormLayout()
        self.layout = QtWidgets.QGridLayout()
        self.top_layout = self.set_top_layout()
        self.left_layout = self.set_left_layout()
        self.right_layout = self.set_right_layout()

        self.layout.setAlignment(QtGui.Qt.AlignTop)
        self.layout.addLayout(self.top_layout, 0, 0, 1, 2)
        self.layout.addLayout(self.left_layout, 1, 0, Qt.AlignLeft)
        self.layout.addLayout(self.right_layout, 1, 1, 1, 3)
        self.layout.setRowStretch(0, 4)
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.setLayout(self.layout)

    def set_top_layout(self):
        """configuration top layout"""
        QtWidgets.QApplication.setStyle("Fusion")
        styleComboBox = QtWidgets.QComboBox()
        styleComboBox.addItems(QtWidgets.QStyleFactory.keys())

        styleLabel = QtWidgets.QLabel("&Style :")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QtWidgets.QCheckBox(
            "&Use style's standard palette"
        )
        self.useStylePaletteCheckBox.setChecked(True)

        styleComboBox.activated[str].connect(self.changeStyle)
        self.useStylePaletteCheckBox.toggled.connect(self.changePalette)

        topLayout = QtWidgets.QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)

        return topLayout

    def set_inference_information_layout(self):
        information_combobox = QtWidgets.QGroupBox("Information")
        v_layout = QtWidgets.QVBoxLayout()
        self.traffic_light_state_label = QtWidgets.QLabel("Traffic Light State \t: ")
        self.inference_fps_label = QtWidgets.QLabel("Inference FPS \t\t: ")
        self.inference_log = QtWidgets.QPlainTextEdit("Log")
        self.inference_log.setMaximumHeight(75)

        self.inference_progress_slider = QtWidgets.QProgressBar()

        v_layout.addWidget(self.traffic_light_state_label)
        v_layout.addWidget(self.inference_fps_label)
        v_layout.addWidget(self.inference_progress_slider)
        v_layout.addWidget(self.inference_log)

        information_combobox.setLayout(v_layout)
        return information_combobox

    def changeStyle(self, styleName):
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create(styleName))
        self.changePalette()

    def changePalette(self):
        if self.useStylePaletteCheckBox.isChecked():
            QtWidgets.QApplication.setPalette(
                QtWidgets.Application.style().standardPalette()
            )
        else:
            QtWidgets.QApplication.setPalette(self.originalPalette)

    def set_configuration_layout(self):
        self.configuration_layout.addWidget(self.video_configuration())
        self.configuration_layout.addWidget(self.model_yolo_configuration())
        self.configuration_layout.addWidget(self.main_configuration())
        self.configuration_layout.addWidget(self.control_configuration())

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(3000, 500, Qt.KeepAspectRatio)
        return p

    def set_image_layout(self):
        self.image_frame = QtWidgets.QLabel()
        self.image = cv2.imread("samples/file-20200803-24-50u91u.jpg")

        self.image = self.convert_cv_qt(self.image)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
        return self.image_frame

    def set_left_layout(self):
        """set left layout for configuration"""

        v_layout = QtWidgets.QVBoxLayout()
        v_layout.addWidget(self.video_configuration())
        v_layout.addWidget(self.model_yolo_configuration())
        v_layout.addWidget(self.main_configuration())
        v_layout.addWidget(self.control_configuration())
        return v_layout

    def set_right_layout(self):
        inference_information_layout = self.set_inference_information_layout()
        h_layout = QtWidgets.QVBoxLayout()
        h_layout.addWidget(self.set_image_layout())
        h_layout.addWidget(inference_information_layout)
        return h_layout

    def video_configuration(self):
        """set video configuration for loading video"""
        self.groupbox_layout = QtWidgets.QHBoxLayout()

        self.lineedit_video_path = QtWidgets.QLineEdit()
        self.button_video_load = QtWidgets.QPushButton("Load Video")
        self.button_video_load.clicked.connect(self.load_video)

        self.groupbox_layout.addWidget(self.lineedit_video_path, 2)
        self.groupbox_layout.addWidget(self.button_video_load, 1)

        self.groupbox_video = QtWidgets.QGroupBox("Video Data")
        self.groupbox_video.setLayout(self.groupbox_layout)

        return self.groupbox_video

    def model_yolo_configuration(self):
        # Combo Box YOLO
        self.yolomodel_vlayout = QtWidgets.QVBoxLayout()

        self.yolomodel_layout = QtWidgets.QHBoxLayout()
        self.combobox_yolo = QtWidgets.QComboBox()

        for model in self.parameter.yolo_model_dict.keys():
            self.combobox_yolo.addItem(model)

        self.current_yolo_model_label = QtWidgets.QLabel("  Current Model : Not Loaded")

        self.button_model_yolo_load = QtWidgets.QPushButton("Load Model")
        self.button_model_yolo_load.clicked.connect(self.load_model)

        self.yolomodel_layout.addWidget(self.combobox_yolo, 2)
        self.yolomodel_layout.addWidget(self.button_model_yolo_load, 1)

        self.groupbox_yolomodel = QtWidgets.QGroupBox("YOLO Model")

        self.yolomodel_vlayout.addLayout(self.yolomodel_layout)
        self.yolomodel_vlayout.addWidget(self.current_yolo_model_label)

        self.groupbox_yolomodel.setLayout(self.yolomodel_vlayout)

        return self.groupbox_yolomodel

    def main_configuration(self):
        self.main_configuration_layout = QtWidgets.QVBoxLayout()
        self.main_configuration_layout.setAlignment(QtCore.Qt.AlignTop)
        self.main_configuration_layout.setStretch(1, 1)

        self.main_configuration_layout.addLayout(self.object_threshold_configuration())
        self.main_configuration_layout.addLayout(self.iou_threshold_configuration())
        self.main_configuration_layout.addLayout(self.max_detection_configuration())

        # Set Init YOLO Configuration Value
        self.set_init_value_main_configuration()

        self.main_configuration_groupbox = QtWidgets.QGroupBox(
            "Parameter Configuration"
        )

        self.set_button_classes = QtWidgets.QPushButton("Change Classes")
        self.set_button_classes.clicked.connect(self.set_msgbox_classes)

        self.set_traffic_light_area = QtWidgets.QPushButton("Set Traffic Lights Area")
        self.set_traffic_light_area.clicked.connect(self.set_traffic_light)

        self.set_button_update_parameters = QtWidgets.QPushButton("Apply Parameters")
        self.set_button_update_parameters.clicked.connect(self.apply_parameters)

        self.main_configuration_layout.addWidget(self.set_button_classes)
        self.main_configuration_layout.addWidget(self.set_traffic_light_area)
        self.main_configuration_layout.addWidget(self.set_button_update_parameters)

        self.main_configuration_groupbox.setAlignment(QtCore.Qt.AlignTop)
        self.main_configuration_groupbox.setLayout(self.main_configuration_layout)

        return self.main_configuration_groupbox

    def set_traffic_light(self):
        if self.parameter.video_path != "":
            self.traffic_light = TrafficLight()
            self.traffic_light.show(self.parameter)
            if self.traffic_light.result() == 1:
                self.parameter = self.traffic_light.parameter
                self.update_parameter()

    def set_init_value_main_configuration(self):
        self.object_threshold_spinbox.setValue(self.parameter.yolo_conf)
        self.iou_threshold_spinbox.setValue(self.parameter.yolo_iou)
        self.max_detection_spinbox.setValue(self.parameter.yolo_max_detection)

    def object_threshold_configuration(self):
        self.object_threshold_layout = QtWidgets.QHBoxLayout()

        self.object_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.object_threshold_spinbox.setMaximum(1)
        self.object_threshold_spinbox.setMinimum(0)
        self.object_threshold_spinbox.setSingleStep(0.01)

        self.object_threshold_spinbox.setValue(0.5)
        self.object_threshold_layout.addWidget(QtWidgets.QLabel("Object Threshold"))
        self.object_threshold_layout.addWidget(self.object_threshold_spinbox)

        return self.object_threshold_layout

    def iou_threshold_configuration(self):
        self.iou_threshold_layout = QtWidgets.QHBoxLayout()
        self.iou_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.iou_threshold_spinbox.setMaximum(1)
        self.iou_threshold_spinbox.setMinimum(0)
        self.iou_threshold_spinbox.setSingleStep(0.01)

        self.iou_threshold_layout.addWidget(QtWidgets.QLabel("IOU Threshold"))
        self.iou_threshold_layout.addWidget(self.iou_threshold_spinbox)

        return self.iou_threshold_layout

    def max_detection_configuration(self):
        self.max_detection_layout = QtWidgets.QHBoxLayout()
        self.max_detection_spinbox = QtWidgets.QDoubleSpinBox()
        self.max_detection_spinbox.setMinimum(1)
        self.max_detection_spinbox.setMaximum(500)
        self.max_detection_spinbox.setSingleStep(1)

        self.max_detection_layout.addWidget(QtWidgets.QLabel("Max Detection"))
        self.max_detection_layout.addWidget(self.max_detection_spinbox)

        return self.max_detection_layout

    def control_configuration(self):
        self.control_groupbox = QtWidgets.QGroupBox("Control")

        self.button_start = QtWidgets.QPushButton("Start")
        self.button_start.clicked.connect(self.start_inference)

        self.button_stop = QtWidgets.QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop_inference)

        control_v_layout = QtWidgets.QVBoxLayout()
        control_h_layout = QtWidgets.QHBoxLayout()

        control_h_layout.addWidget(self.button_start)
        control_h_layout.addWidget(self.button_stop)

        control_v_layout.addLayout(control_h_layout)

        self.control_groupbox.setLayout(control_v_layout)

        return self.control_groupbox

    def set_msgbox_classes(self):
        print("LOG : Change Classes")
        msg = QtWidgets.QDialog()
        checkbox_layout = QtWidgets.QGridLayout()

        self.checkbox_car = QtWidgets.QCheckBox()
        self.checkbox_motorcycle = QtWidgets.QCheckBox()
        self.checkbox_bus = QtWidgets.QCheckBox()
        self.checkbox_truck = QtWidgets.QCheckBox()

        self.checkbox_save = QtWidgets.QPushButton("Save")
        self.checkbox_save.clicked.connect(self.set_classes)

        self.check_current_classses()

        checkbox_layout.addWidget(QtWidgets.QLabel("Car"), 0, 0)
        checkbox_layout.addWidget(self.checkbox_car, 0, 1)
        checkbox_layout.addWidget(QtWidgets.QLabel("Motorcycle"), 1, 0)
        checkbox_layout.addWidget(self.checkbox_motorcycle, 1, 1)
        checkbox_layout.addWidget(QtWidgets.QLabel("Bus"), 2, 0)
        checkbox_layout.addWidget(self.checkbox_bus, 2, 1)
        checkbox_layout.addWidget(QtWidgets.QLabel("Truck"), 3, 0)
        checkbox_layout.addWidget(self.checkbox_truck, 3, 1)
        checkbox_layout.addWidget(self.checkbox_save, 4, 0, 1, 2)
        Qt.ApplicationModal
        msg.setWindowModality(Qt.ApplicationModal)
        msg.setLayout(checkbox_layout)
        msg.exec_()

    def check_current_classses(self):
        if 2 in self.parameter.yolo_classes:
            self.checkbox_car.setChecked(True)
        if 3 in self.parameter.yolo_classes:
            self.checkbox_motorcycle.setChecked(True)
        if 5 in self.parameter.yolo_classes:
            self.checkbox_bus.setChecked(True)
        if 7 in self.parameter.yolo_classes:
            self.checkbox_truck.setChecked(True)

    @Slot()
    def set_classes(self):
        classes_list = []
        if self.checkbox_car.isChecked():
            classes_list.append(2)
        if self.checkbox_motorcycle.isChecked():
            classes_list.append(3)
        if self.checkbox_bus.isChecked():
            classes_list.append(5)
        if self.checkbox_truck.isChecked():
            classes_list.append(7)
        self.parameter.yolo_classes = classes_list
        print(self.parameter.yolo_classes)

    @Slot()
    def load_video(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Videos Files (*.*)",
            options=options,
        )
        if fileName:
            self.lineedit_video_path.setText(fileName)
            self.parameter.video_path = fileName

    @Slot()
    def load_model(self):
        print("LOG : Loaded Model Clicked")
        print("Model Type : {}".format(self.combobox_yolo.currentText()))

        self.current_yolo_model_label.setText("  Current Model : Loading model...")

        state_model = self.yolo.load_model(
            self.parameter.yolo_model_dict[self.combobox_yolo.currentText()]
        )
        if state_model:
            self.current_yolo_model_label.setText(
                "  Current Model : {}".format(self.combobox_yolo.currentText())
            )
        else:
            self.current_yolo_model_label.setText(
                "  Current Model : Failed to load model!"
            )

    @Slot()
    def apply_parameters(self):
        self.parameter.yolo_conf = self.object_threshold_spinbox.value()
        self.parameter.yolo_iou = self.iou_threshold_spinbox.value()
        self.parameter.yolo_max_detection = int(self.max_detection_spinbox.value())

        self.yolo.update_params(
            conf=self.parameter.yolo_conf,
            iou=self.parameter.yolo_iou,
            classes=self.parameter.yolo_classes,
            multi_label=self.parameter.yolo_multi_label,
            max_detection=self.parameter.yolo_max_detection,
        )

    @Slot()
    def start_inference(self):
        print("start inference")
        if self.parameter.video_path == "":
            print("Please load video first!")

        else:
            print("Video loaded!")
            self.vid = cv2.VideoCapture(self.parameter.video_path)
            if self.vid.isOpened() == False:
                print("Error opening video file!")
            else:
                self.parameter.video_fps = self.vid.get(cv2.CAP_PROP_FPS)
                self.parameter.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

                print(self.parameter.video_fps)
                print(self.parameter.frame_count)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000.0 / 30)

    @Slot()
    def stop_inference(self):
        print("stop inference")
        self.timer.stop()

    def update_frame(self):

        start_time = time.time()

        # get frame from cv2 read
        ret, frame = self.vid.read()

        # get parameter traffic light frame position
        index_crop = self.parameter.traffic_light_area

        # croping frame to get traffic light frame
        self.traffic_light_frame = frame[
            index_crop[1] : index_crop[3], index_crop[0] : index_crop[2]
        ]
        self.traffic_light_frame = np.ascontiguousarray(self.traffic_light_frame)

        # processing to get color of traffic light
        status = self.tld.detect_state(self.traffic_light_frame)

        # resize frame for yolo inference
        frame_resize = cv2.resize(
            frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))
        )

        # inference yolov5 and count processing time
        new_frame = self.yolo.inference_frame(frame_resize)
        end_time = time.time()

        # update inference information
        fps = 1 / (end_time - start_time)
        progress_value = (
            self.vid.get(cv2.CAP_PROP_POS_FRAMES) / self.parameter.frame_count
        ) * 100
        self.traffic_light_state_label.setText(f"Traffic Light State \t: {status}")
        # self.inference_fps_label.setText(f"Inference FPS \t\t: {fps:.2f}")
        self.inference_fps_label.setText(str(self.parameter.video_fps))
        self.inference_progress_slider.setValue(int(progress_value))

        # convert and shows new frame inference in pyqt5 label
        img = self.convert_cv_qt(new_frame)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(img))

    def update_parameter(self):
        self.tld.update_parameters(self.parameter)


class DatabaseLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QFormLayout()
        self.text = QtWidgets.QTextEdit("Test2")
        self.layout.addWidget(self.text)
        self.setLayout(self.layout)
