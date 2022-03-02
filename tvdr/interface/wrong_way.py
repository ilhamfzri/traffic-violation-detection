import cv2

from PySide2 import QtWidgets
from PySide2 import QtCore
from PySide2.QtCore import QLine, Qt, QPoint, QRect
from PySide2.QtGui import QIntValidator, QPainter, QPen, QBrush, QIcon, QPixmap
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
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PySide2 import QtGui
from tvdr.core.algorithm import cart2pol, pol2cart
import numpy as np
from tvdr.utils.params import Parameter


class PainterArrow(QLabel):
    def __init__(self, parent=None):
        QLabel.__init__(self, parent)
        self.list_point = []
        self.current_pos = QPoint()
        self.direction = 0
        self.threshold = 0
        self.setText("This text does not appear")

    def update_frame(self, cv_img):
        self.cv_img = cv_img
        self.update_direction(self.direction, self.threshold)

    def update_direction(self, direction: int, threshold: int):

        self.arrow_length = 120
        self.direction = direction
        self.threshold = threshold

        if direction - threshold < 0:
            direction_th1 = 360 - (threshold - direction)
        else:
            direction_th1 = direction - threshold

        if direction + threshold > 360:
            direction_th2 = threshold - (360 - direction)
        else:
            direction_th2 = direction + threshold

        y_center = self.cv_img.shape[0] / 2
        x_center = self.cv_img.shape[1] / 2

        pos_main = pol2cart(self.arrow_length, np.radians(direction))

        new_img = self.cv_img.copy()
        new_img = cv2.arrowedLine(
            new_img,
            (int(x_center), int(y_center)),
            (int(pos_main[0] + x_center), int(y_center - pos_main[1])),
            color=(0, 0, 255),
            thickness=2,
            line_type=cv2.LINE_AA,
            tipLength=0.1,
        )

        pos_th1 = pol2cart(self.arrow_length, np.radians(direction_th1))

        new_img = cv2.line(
            new_img,
            (int(x_center), int(y_center)),
            (int(pos_th1[0] + x_center), int(y_center - pos_th1[1])),
            color=(0, 0, 255),
            thickness=4,
        )

        pos_th2 = pol2cart(self.arrow_length, np.radians(direction_th2))

        new_img = cv2.line(
            new_img,
            (int(x_center), int(y_center)),
            (int(pos_th2[0] + x_center), int(y_center - pos_th2[1])),
            color=(0, 0, 255),
            thickness=4,
        )

        self.qimage = self.convert_cv_qt(new_img)
        self.qpixmap_data = QtGui.QPixmap.fromImage(self.qimage)
        self.setPixmap(self.qpixmap_data)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        height_target = 400
        divider = height_target / h
        w_new = int(divider * w)
        p = convert_to_Qt_format.scaled(w_new, height_target, Qt.KeepAspectRatio)
        return p


class WrongWayConfig(QtWidgets.QDialog):
    def __init__(self, parameter: Parameter):
        super().__init__()
        self.parameter = parameter
        video_path = self.parameter.video_path

        main_v_layout = QtWidgets.QVBoxLayout()
        direction_and_threshold_layout = QtWidgets.QVBoxLayout()
        main_h_layout = QtWidgets.QHBoxLayout()

        # Read Video Data and Information
        self.vid = cv2.VideoCapture(video_path)
        self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        _, self.frame_data = self.vid.read()

        # Set Mouse Position Tracking and Video Resolution
        h_layout_video_size_and_pointer = QtWidgets.QHBoxLayout()
        self.frame_size_information = QtWidgets.QLabel(
            f"Resolution : {self.frame_data.shape}"
        )
        self.mouse_pos_label = QtWidgets.QLabel(f"Mouse Position : (0,0)")
        h_layout_video_size_and_pointer.addWidget(self.frame_size_information)
        h_layout_video_size_and_pointer.addWidget(self.mouse_pos_label)

        # Set Miss Count and Minimum Total dXY
        h_misscount_and_totaldxy_layout = QtWidgets.QHBoxLayout()

        self.misscount_line = QtWidgets.QLineEdit()
        self.onlyFloat = QIntValidator()
        self.misscount_line.setValidator(self.onlyFloat)
        v_misscount = QtWidgets.QVBoxLayout()
        v_misscount.addWidget(QtWidgets.QLabel("Miss Removal Object (in second)"))
        v_misscount.addWidget(self.misscount_line)

        self.threshold_total = QtWidgets.QHBoxLayout()
        self.threshold_total_line = QtWidgets.QLineEdit()
        self.threshold_total_line.setValidator(self.onlyFloat)
        v_threshold_total = QtWidgets.QVBoxLayout()
        v_threshold_total.addWidget(QtWidgets.QLabel("Minimum (dX+dY) Threshold"))
        v_threshold_total.addWidget(self.threshold_total_line)

        h_misscount_and_totaldxy_layout.addLayout(v_threshold_total)
        h_misscount_and_totaldxy_layout.addLayout(v_misscount)

        # Set Slider Video
        self.slider_video = QtWidgets.QSlider()
        self.slider_video.setMinimum(0)
        self.slider_video.setMaximum(self.frame_count)
        self.slider_video.setSingleStep(1)
        self.slider_video.setOrientation(Qt.Horizontal)
        self.slider_video.valueChanged.connect(self.update_frame)

        # Set Slider Direction
        self.slider_direction = QtWidgets.QSlider(Qt.Horizontal)
        self.slider_direction.setMinimum(0)
        self.slider_direction.setMaximum(350)
        self.slider_direction.setValue(0)
        self.slider_direction.setTickInterval(10)
        self.slider_direction.setTickPosition(QSlider.TicksBelow)
        self.slider_direction.valueChanged.connect(self.update)
        # self.slider_direction.setOrientation(Qt.Horizontal)

        # Set Slider Direction Threshold
        self.slider_direction_threshold = QtWidgets.QSlider()
        self.slider_direction_threshold.setMinimum(0)
        self.slider_direction_threshold.setMaximum(60)
        self.slider_direction_threshold.setSingleStep(2)
        self.slider_direction_threshold.setTickInterval(20)
        self.slider_direction_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_direction_threshold.setOrientation(Qt.Horizontal)
        self.slider_direction_threshold.valueChanged.connect(self.update)

        # Set Label
        self.label_direction = QtWidgets.QLabel(f"Direction Value : {0}°")
        self.label_direction_threshold = QtWidgets.QLabel(f"Threshold Value : ±{0}°")

        direction_and_threshold_layout.addWidget(self.label_direction)
        direction_and_threshold_layout.addWidget(self.slider_direction)
        direction_and_threshold_layout.addWidget(self.label_direction_threshold)
        direction_and_threshold_layout.addWidget(self.slider_direction_threshold)
        direction_and_threshold_layout.setSizeConstraint(QLayout.SetFixedSize)

        # Set Slider Threshold

        # Set Image
        self.image_label = PainterArrow(self)
        self.image_label.update_frame(self.frame_data)

        # # Set Button
        self.save_button = QtWidgets.QPushButton("Done")
        self.save_button.clicked.connect(self.save_configuration)

        # # Set Horizontal Top Bar Menu
        main_h_layout.addWidget(self.slider_video)
        main_h_layout.addWidget(self.save_button)
        main_h_layout.setSizeConstraint(QLayout.SetFixedSize)

        # Set Main Menu Layout
        main_v_layout.addLayout(h_misscount_and_totaldxy_layout)
        main_v_layout.addLayout(direction_and_threshold_layout)
        main_v_layout.addWidget(self.image_label)
        main_v_layout.addLayout(main_h_layout)

        # main_v_layout.addLayout(h_layout_video_size_and_pointer)
        main_v_layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(main_v_layout)
        self.init_widget()
        self.setWindowModality(QtCore.Qt.ApplicationModal)

    def init_widget(self):
        self.slider_direction.setValue(self.parameter.wrongway_direction_degree)
        self.slider_direction_threshold.setValue(
            self.parameter.wrongway_threshold_degree
        )
        self.misscount_line.setText(str(self.parameter.wrongway_miss_count))
        self.threshold_total_line.setText(str(self.parameter.wrongway_min_value))

    def update_frame(self):
        frame_position = self.slider_video.value()

        self.vid.set(1, frame_position)
        _, self.frame_data = self.vid.read()

        self.image_label.update_frame(self.frame_data)

    def update(self):
        self.label_direction.setText(
            f"Direction Value : {self.slider_direction.value()}°"
        )

        self.label_direction_threshold.setText(
            f"Threshold Value : ±{self.slider_direction_threshold.value()}°"
        )

        self.image_label.update_direction(
            self.slider_direction.value(), self.slider_direction_threshold.value()
        )

    def get_params(self):
        params_dict = {}
        params_dict["direction_degree"] = self.slider_direction.value()
        params_dict["threshold_degree"] = self.slider_direction_threshold.value()
        params_dict["miss_count"] = int(self.misscount_line.text())
        params_dict["min_value"] = int(self.threshold_total_line.text())
        return params_dict

    def save_configuration(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText("Do you want to save this wrong way parameter?")
        msg.setDetailedText(
            f"""
        The details paramaters are as follows:
        {self.get_params()}
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
