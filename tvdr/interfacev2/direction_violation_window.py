import cv2
import numpy as np

from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from tvdr.core import WrongWayDetectionConfig
from copy import deepcopy
from tvdr.core.algorithm import cart2pol, pol2cart


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

        self.arrow_length = 180
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
            thickness=4,
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
        self.qpixmap_data = QPixmap.fromImage(self.qimage)
        self.setPixmap(self.qpixmap_data)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        height_target = 400
        divider = height_target / h
        w_new = int(divider * w)
        p = convert_to_Qt_format.scaled(w_new, height_target, Qt.KeepAspectRatio)
        return p


class DirectionViolationInterface(QDialog):
    def __init__(self, config: WrongWayDetectionConfig, video_path: str):
        super().__init__()
        self.config = deepcopy(config)

        # Read Video Data and Information
        self.vid = cv2.VideoCapture(video_path)
        self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
        _, self.frame_data = self.vid.read()

        # Set Slider Video
        self.slider_video = QSlider()
        self.slider_video.setMinimum(0)
        self.slider_video.setMaximum(self.frame_count)
        self.slider_video.setSingleStep(1)
        self.slider_video.setOrientation(Qt.Horizontal)
        self.slider_video.valueChanged.connect(self.update_frame)

        # Set Slider Direction
        self.slider_direction = QSlider(Qt.Horizontal)
        self.slider_direction.setMinimum(0)
        self.slider_direction.setMaximum(350)
        self.slider_direction.setValue(0)
        self.slider_direction.setTickInterval(10)
        self.slider_direction.setTickPosition(QSlider.TicksBelow)
        self.slider_direction.valueChanged.connect(self.update_value)
        # self.slider_direction.setOrientation(Qt.Horizontal)

        # Set Slider Direction Threshold
        self.slider_direction_threshold = QSlider()
        self.slider_direction_threshold.setMinimum(0)
        self.slider_direction_threshold.setMaximum(60)
        self.slider_direction_threshold.setSingleStep(2)
        self.slider_direction_threshold.setTickInterval(20)
        self.slider_direction_threshold.setTickPosition(QSlider.TicksBelow)
        self.slider_direction_threshold.setOrientation(Qt.Horizontal)
        self.slider_direction_threshold.valueChanged.connect(self.update_value)

        # Set Label
        self.label_direction = QLabel(f"Direction Value : {0}°")
        self.label_direction_threshold = QLabel(f"Threshold Value : ±{0}°")

        # Set Image
        self.image_label = PainterArrow(self)
        self.image_label.update_frame(self.frame_data)

        # # Set Button
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_configuration)

        # Setup Top Layout
        top_layout = QVBoxLayout()
        top_layout.addWidget(self.label_direction)
        top_layout.addWidget(self.slider_direction)
        top_layout.addWidget(self.label_direction_threshold)
        top_layout.addWidget(self.slider_direction_threshold)
        top_layout.setSizeConstraint(QLayout.SetFixedSize)

        # Setup Bottom layout
        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.slider_video)
        bottom_layout.addWidget(self.save_button)

        # Combine Layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)
        self.init_widget()
        self.setWindowModality(Qt.ApplicationModal)

    def init_widget(self):
        self.slider_direction.setValue(self.config.direction_violation)
        self.slider_direction_threshold.setValue(self.config.direction_violation_thr)

    def save_configuration(self):
        msg = QMessageBox()
        msg.setText("Do you want to save this parameter?")
        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        response = msg.exec_()

        if response == QMessageBox.Save:
            self.config.direction_violation = self.slider_direction.value()
            self.config.direction_violation_thr = (
                self.slider_direction_threshold.value()
            )
            self.accept()

        elif response == QMessageBox.Cancel:
            pass

    def update_frame(self):
        frame_position = self.slider_video.value()
        self.vid.set(1, frame_position)
        _, self.frame_data = self.vid.read()
        self.image_label.update_frame(self.frame_data)

    def update_value(self):
        self.label_direction.setText(
            f"Direction Value : {self.slider_direction.value()}°"
        )

        self.label_direction_threshold.setText(
            f"Threshold Value : ±{self.slider_direction_threshold.value()}°"
        )

        self.image_label.update_direction(
            self.slider_direction.value(), self.slider_direction_threshold.value()
        )
