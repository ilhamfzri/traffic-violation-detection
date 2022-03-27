from tokenize import group
import cv2
import qtawesome as qta

from copy import deepcopy
from tvdr.core import RunningRedLightConfig, RunningRedLight
from PySide2.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QHBoxLayout,
    QVBoxLayout,
    QSlider,
    QGroupBox,
    QPushButton,
    QMessageBox,
)
from PySide2 import QtCore
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap, QImage
from .range_slider import RangeSlider


class ColorSegmentation(QDialog):
    def __init__(
        self, config: RunningRedLightConfig, video_path: str, light_type="red"
    ):
        super().__init__()
        self.config = deepcopy(config)
        self.video_path = video_path
        self.type = light_type
        self.rrl = RunningRedLight(self.config)

        mainLayout = QVBoxLayout()

        self.apply_configuration_button = QPushButton(
            qta.icon("fa5s.check"), "Apply Configuration"
        )
        self.apply_configuration_button.clicked.connect(self.apply_config)

        self.save_configuration_button = QPushButton(
            qta.icon("mdi6.content-save-check"), "Close and Save Configuration"
        )
        self.save_configuration_button.clicked.connect(self.save_config)

        mainLayout.addLayout(self.set_image_layout())
        mainLayout.addWidget(self.set_hsv_layout())
        mainLayout.addWidget(self.apply_configuration_button)
        mainLayout.addWidget(self.save_configuration_button)

        self.setLayout(mainLayout)
        self.setWindowModality(QtCore.Qt.ApplicationModal)
        self.init_layout()

    def apply_config(self):
        self.rrl.config = self.config
        self.update_frame()

    def save_config(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Save Red Light Configuration?")

        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        response = msg.exec_()

        if response == QMessageBox.Save:
            self.accept()

        elif response == QMessageBox.Cancel:
            pass

    def set_image_layout(self):
        v_layout = QVBoxLayout()
        h_image = QHBoxLayout()

        groupbox_area = QGroupBox("")
        h_layout = QHBoxLayout()
        self.label_area = QLabel(f"Current Area Size : {0} %")
        h_layout.addWidget(self.label_area)
        groupbox_area.setLayout(h_layout)

        self.image_seg = QLabel()
        self.image = QLabel()
        self.video_slider = QSlider()
        self.video_slider.setOrientation(Qt.Horizontal)
        self.video_slider.valueChanged.connect(self.update_frame)

        h_image.addWidget(self.image_seg, 1)
        h_image.addWidget(self.image, 1)

        v_layout.addLayout(h_image)
        v_layout.addWidget(groupbox_area)
        v_layout.addWidget(self.video_slider)

        return v_layout

    def set_hsv_layout(self):
        v_layout = QVBoxLayout()

        groupbox_layout = QGroupBox("HSV Color Range")
        groupbox_layout.setAlignment(Qt.AlignCenter)

        self.label_h = QLabel(f"Hue : ( {0} - {0} )")
        self.slider_h = RangeSlider(0, 179)
        self.slider_h.valueChanged.connect(self.update_slider)

        self.label_s = QLabel(f"Saturation : ( {0} - {0} )")
        self.slider_s = RangeSlider(0, 255)
        self.slider_s.valueChanged.connect(self.update_slider)

        self.label_v = QLabel(f"Value : ( {0} - {0} )")
        self.slider_v = RangeSlider(0, 255)
        self.slider_v.valueChanged.connect(self.update_slider)

        self.label_min_area = QLabel(f"Min Area Thr: {0} %")
        self.slider_min_area = QSlider()
        self.slider_min_area.setOrientation(Qt.Horizontal)
        self.slider_min_area.setMinimum(1)
        self.slider_min_area.setMaximum(100)
        self.slider_min_area.valueChanged.connect(self.update_slider)

        v_layout.addWidget(self.label_h)
        v_layout.addWidget(self.slider_h)

        v_layout.addWidget(self.label_s)
        v_layout.addWidget(self.slider_s)

        v_layout.addWidget(self.label_v)
        v_layout.addWidget(self.slider_v)

        v_layout.addWidget(self.label_min_area)
        v_layout.addWidget(self.slider_min_area)

        groupbox_layout.setLayout(v_layout)
        return groupbox_layout

    def update_frame(self):
        frame_idx = self.video_slider.value()
        self.vid.set(1, frame_idx)
        _, self.current_img = self.vid.read()

        self.rrl.detect_state(self.current_img)
        cropped_img = self.rrl.cropped
        cropped_img_seg = getattr(self.rrl, f"{self.type}_seg")
        current_size = getattr(self.rrl, f"{self.type}_area")

        qt_img = self.convert_cv_qt(cropped_img)
        qt_img_seg = self.convert_cv_qt(cropped_img_seg)

        self.image_seg.setPixmap(QPixmap.fromImage(qt_img))
        self.image.setPixmap(QPixmap.fromImage(qt_img_seg))
        self.label_area.setText(f"Current Size Area : {current_size}%")

    def update_slider(self):
        self.label_h.setText(
            f"Hue : ( {self.slider_h.first_position} - {self.slider_h.second_position} )"
        )
        self.label_s.setText(
            f"Saturation : ( {self.slider_s.first_position} - {self.slider_s.second_position} "
        )
        self.label_v.setText(
            f"Value : ( {self.slider_v.first_position} - {self.slider_v.second_position} )"
        )

        self.label_min_area.setText(f"Min Area Thr : {self.slider_min_area.value()} %")

        new_hsv_min = [
            self.slider_h.first_position,
            self.slider_s.first_position,
            self.slider_v.first_position,
        ]

        new_hsv_max = [
            self.slider_h.second_position,
            self.slider_s.second_position,
            self.slider_v.second_position,
        ]

        setattr(self.config, f"{self.type}_hsv_min", new_hsv_min)
        setattr(self.config, f"{self.type}_hsv_max", new_hsv_max)
        setattr(self.config, f"{self.type}_min_area", self.slider_min_area.value())

    def init_layout(self):
        self.vid = cv2.VideoCapture(self.video_path)

        if self.vid.isOpened() == False:
            print("Error opening video file!")
        else:
            total_frame = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_slider.setMaximum(total_frame - 1)

        hsv_min = getattr(self.config, f"{self.type}_hsv_min")
        hsv_max = getattr(self.config, f"{self.type}_hsv_max")

        self.slider_h.setRange(hsv_min[0], hsv_max[0])
        self.slider_s.setRange(hsv_min[1], hsv_max[1])
        self.slider_v.setRange(hsv_min[2], hsv_max[2])
        self.slider_min_area.setValue(getattr(self.config, f"{self.type}_min_area"))

        self.label_h.setText(f"Hue : ( {hsv_min[0]} - {hsv_max[0]} )")
        self.label_s.setText(f"Saturation : ( {hsv_min[1]} - {hsv_max[1]} )")
        self.label_v.setText(f"Value : ( {hsv_min[2]} - {hsv_max[2]} )")

        self.update_frame()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(10000, 100, Qt.KeepAspectRatio)
        return p
