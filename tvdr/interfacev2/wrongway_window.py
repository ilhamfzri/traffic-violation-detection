import qtawesome as qta
import cv2

from copy import deepcopy
from tvdr.core import (
    WrongWayDetectionConfig,
    WrongWayDetection,
    VehicleDetectionConfig,
    VehicleDetection,
    PipelineConfig,
)
from tvdr.utils import Annotator
from tvdr.interfacev2 import DirectionViolationInterface

from PySide2.QtWidgets import *
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import *


# from .direction_violation import DirectionViolation


class WrongWayInterface(QDialog):
    def __init__(
        self,
        config: WrongWayDetectionConfig,
        vd_config: VehicleDetectionConfig,
        video_path: str,
    ):
        super().__init__()
        self.setWindowTitle("Wrong Way Detection Configuration")

        self.config = deepcopy(config)
        self.vd_config = vd_config
        self.video_path = video_path

        # Initialize Vehicle Detection and WrongWay
        self.vd = VehicleDetection(vd_config)
        self.ww = WrongWayDetection(config)

        # Initialize Annotator
        self.annotator_config = PipelineConfig()
        self.annotator_config.vd_config = vd_config
        self.annotator = Annotator(self.annotator_config)
        self.annotate_state = False

        self.layout = QGridLayout()
        self.layout.addLayout(self.set_left_layout(), 0, 0, 1, 2, Qt.AlignLeft)
        self.layout.addLayout(self.set_right_layout(), 0, 2, 1, 6, Qt.AlignCenter)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)

        self.setLayout(self.layout)
        self.config_init()
        self.setWindowModality(Qt.ApplicationModal)

    def set_left_layout(self):
        v_left_layout = QVBoxLayout()
        groupbox_left = QGroupBox("Configuration")

        self.direction_violation_label = QLabel(f"Direction Violation \t:  {0}°")
        self.direction_violation_thr_label = QLabel(
            f"Direction Violation Thr \t: ±{0}°"
        )
        self.direction_violation_button = QPushButton("Set Direction Violation")
        self.direction_violation_button.clicked.connect(self.set_direction_violation)

        self.direction_sigma_dy_dx = QLabel(f"Minimum Total (dY+dX)")
        self.direction_sigma_dy_dx_spin_box = QSpinBox()

        grid_left_layout = QGridLayout()
        grid_left_layout.addWidget(self.direction_violation_label, 0, 0, 1, 2)
        grid_left_layout.addWidget(self.direction_violation_thr_label, 1, 0, 1, 2)
        grid_left_layout.addWidget(self.direction_violation_button, 2, 0, 1, 2)
        grid_left_layout.addWidget(self.direction_sigma_dy_dx, 3, 0)
        grid_left_layout.addWidget(self.direction_sigma_dy_dx_spin_box, 3, 1)
        groupbox_left.setLayout(grid_left_layout)

        # self.apply_configuration_button = QPushButton(
        #     qta.icon("fa5s.check"), "Apply Configuration"
        # )
        # self.apply_configuration_button.clicked.connect(self.apply_config)

        self.save_configuration_button = QPushButton(
            qta.icon("mdi6.content-save-check"), "Close and Save Configuration"
        )
        self.save_configuration_button.clicked.connect(self.save_config)

        v_left_layout.addWidget(groupbox_left)
        # v_left_layout.addWidget(self.apply_configuration_button)
        v_left_layout.addWidget(self.save_configuration_button)
        v_left_layout.addStretch(1)

        return v_left_layout

    def set_image_layout(self):
        self.image_frame = QLabel()
        self.image = cv2.imread("samples/meong.jpg")

        self.image = self.convert_cv_qt(self.image)
        self.image_frame.setPixmap(QPixmap.fromImage(self.image))
        return self.image_frame

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(10000, 500, Qt.KeepAspectRatio)
        return p

    def set_right_layout(self):
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.set_image_layout())
        v_layout.addLayout(self.set_slider_and_inference())

        return v_layout

    def set_slider_and_inference(self):
        slider_and_inference_layout = QGridLayout()

        self.video_slider = QSlider()
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(1)
        self.video_slider.setSingleStep(1)
        self.video_slider.setOrientation(Qt.Horizontal)
        self.video_slider.valueChanged.connect(self.update_frame)

        self.frame_label = QLabel("Total Frame")
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setMinimum(5)
        self.frame_spinbox.setMaximum(100)
        self.frame_spinbox.setSingleStep(1)

        self.inference_button = QPushButton("Inference Frame")
        self.inference_button.clicked.connect(self.inference_image)

        slider_and_inference_layout.addWidget(self.video_slider, 0, 0, 2, 8)
        slider_and_inference_layout.addWidget(self.frame_label, 0, 8, 1, 1)
        slider_and_inference_layout.addWidget(self.frame_spinbox, 0, 9, 1, 1)
        slider_and_inference_layout.addWidget(self.inference_button, 1, 8, 1, 2)

        return slider_and_inference_layout

    def update_frame(self):
        frame_idx = self.video_slider.value()
        if self.annotate_state == False:
            self.vid.set(1, frame_idx)
            _, self.current_img = self.vid.read()
            img = self.convert_cv_qt(self.current_img)
            self.image_frame.setPixmap(QPixmap.fromImage(img))
        self.annotate_state = False

    def config_init(self):
        # Video Initialize
        self.vid = cv2.VideoCapture(self.video_path)
        _, img = self.vid.read()

        if self.vid.isOpened() == False:
            print("Error opening video file!")
        else:
            self.total_frames = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_slider.setMaximum(self.total_frames - 1)

        img = self.convert_cv_qt(img)
        self.image_frame.setPixmap(QPixmap.fromImage(img))

        self.direction_violation_label.setText(
            f"Direction Violation \t: {self.config.direction_violation}°"
        )
        self.direction_violation_thr_label.setText(
            f"Direction Violation Thr \t: ±{self.config.direction_violation_thr}°"
        )
        self.direction_sigma_dy_dx_spin_box.setValue(
            self.config.min_sigma_dy_dx_violation
        )

    def apply_config(self):
        pass

    def save_config(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Save Wrong-way Configuration?")

        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        response = msg.exec_()

        if response == QMessageBox.Save:
            self.accept()

        elif response == QMessageBox.Cancel:
            pass

    def set_direction_violation(self):
        self.dvi = DirectionViolationInterface(
            config=self.config, video_path=self.video_path
        )
        self.dvi.exec_()

        if self.dvi.result() == 1:
            self.config = self.dvi.config
            self.direction_violation_label.setText(
                f"Direction Violation \t:  {self.config.direction_violation}°"
            )
            self.direction_violation_thr_label.setText(
                f"Direction Violation Thr \t: ±{self.config.direction_violation_thr}°"
            )

    def inference_image(self):
        current_idx_frame = self.video_slider.value()
        count_frame = self.frame_spinbox.value()

        self.ref_idx_frame = current_idx_frame + count_frame
        if self.ref_idx_frame > self.total_frames - 1:
            self.ref_idx_frame = self.total_frames - 1

        self.ww.config = self.config
        self.ww.reset_object_tracker()
        self.vd.reset_tracker()

        self.vid.set(1, current_idx_frame)

        self.timer = QTimer()
        self.timer.timeout.connect(self.process)
        self.timer.start(1000 / 30)

    def process(self):
        _, frame = self.vid.read()

        preds = self.vd.update(frame)
        annotate_img = self.annotator.vehicle_detection(frame, preds)

        direction_data, violation = self.ww.update(preds)
        annotate_img = self.annotator.wrongway_detection(
            annotate_img, preds, direction_data, violation
        )

        self.annotate_state = True
        current_idx = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
        img = self.convert_cv_qt(annotate_img)
        self.image_frame.setPixmap(QPixmap.fromImage(img))

        self.video_slider.setValue(current_idx)

        if current_idx > self.ref_idx_frame:
            self.timer.stop()
