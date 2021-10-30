import cv2
from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2.QtCore import Qt, QPoint, QRect
from PySide2.QtGui import QPainter, QPen, QBrush, QIcon
from PySide2.QtWidgets import (
    QApplication,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QVBoxLayout,
    QWidget,
)
from PySide2 import QtGui

import numpy as np


class TrafficLight(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.main_layout = QtWidgets.QGridLayout()
        self.config_layout = self.configuration_layout()
        self.image_layout = self.image_layout_config()

        self.main_layout.addLayout(self.config_layout, 0, 0, 4, 1)
        self.main_layout.columnStretch(1)
        self.main_layout.addLayout(self.image_layout, 0, 1, 4, 3)

    def show(self, parameter):
        self.parameter = parameter
        print(self.parameter)

        self.read_video_init()
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.setLayout(self.main_layout)
        self.exec_()

    def read_video_init(self):
        self.vid = cv2.VideoCapture(self.parameter.video_path)
        frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        self.slider_frame.setMaximum(int(frame_count))

        ret, frame = self.vid.read()

        image = QtGui.QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            QtGui.QImage.Format_RGB888,
        ).rgbSwapped()

        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(image))
        pass

    def update_frame(self):
        frame_position = self.slider_frame.value()
        self.vid.set(1, frame_position)

        ret, self.current_frame = self.vid.read()

        print("Image Shape Before Crop : {}".format(self.current_frame.shape))

        if self.parameter.traffic_light_set_view == 1:
            index_crop = self.parameter.traffic_light_area
            crop_frame = self.current_frame[
                index_crop[1] : index_crop[3], index_crop[0] : index_crop[2]
            ]
            self.current_frame = np.ascontiguousarray(crop)
            print("Image Shape After Crop : {}".format(self.current_frame.shape))
            print(self.current_frame.shape)

        image = QtGui.QImage(
            self.current_frame,
            self.current_frame.shape[1],
            self.current_frame.shape[0],
            QtGui.QImage.Format_RGB888,
        ).rgbSwapped()

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

        h_layout_cropped.addWidget(self.checkbox_area_traffic_light)
        h_layout_cropped.addWidget(self.checkbox_area_all)
        h_groupbox_cropped.setLayout(h_layout_cropped)

        # Set Image Processing
        h_groupbox_image_post_processing = QGroupBox("Set Image")
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
        h_groupbox_image_post_processing.setLayout(h_layout_image_post_processing)

        v_layout.addWidget(red_light_gb)
        v_layout.addWidget(green_light_gb)
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
        else:
            self.checkbox_image_rgb.setChecked(False)
            self.checkbox_image_segmentation.setChecked(True)

    def set_area(self):
        ret, self.current_frame = self.vid.read()
        self.traffic_light_area = TrafficLightArea(self.current_frame)
        self.traffic_light_area.exec_()
        if self.traffic_light_area.result() == 1:
            self.parameter.traffic_light_area = self.traffic_light_area.getArea()
            print(self.parameter.traffic_light_area)
            self.update_frame()

    def save_configuration(self):
        pass


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

        self.image = cv2.imread("samples/file-20200803-24-50u91u.jpg")
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
        # self.painter(self.qimage2)
        qp = QPainter(self.qimage2)
        qp.setPen(QPen(Qt.red, 1, Qt.SolidLine))

        if not self.begin.isNull() and not self.end.isNull():
            # self.painter.drawRect(QRect(self.begin, self.end).normalized())
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
        # msg.setInformativeText("Traf")
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
            print("LOG : Save Area Traffic Lights")
            self.x_begin = self.begin.x()
            self.y_begin = self.begin.y()

            self.x_end = self.end.x()
            self.y_end = self.end.y()

            self.accept()

        elif response == QtWidgets.QMessageBox.Cancel:
            print("LOG : Cancel")

    def getArea(self):
        return [self.x_begin, self.y_begin, self.x_end, self.y_end]
