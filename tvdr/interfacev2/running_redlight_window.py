import cv2

from PySide2.QtWidgets import (
    QDialog,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QDoubleSpinBox,
    QLabel,
    QGroupBox,
    QLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QSlider,
    QMessageBox,
    QStyle,
)
from PySide2.QtCore import Qt
from PySide2 import QtGui
from PySide2.QtGui import QPixmap, QImage
from copy import deepcopy

from tvdr.core import VehicleDetection, VehicleDetectionConfig
from tvdr.core import RunningRedLightConfig, RunningRedLight
from tvdr.utils import Annotator
from tvdr.core import PipelineConfig

from .detection_area import DetectionArea
from .color_segmentation import ColorSegmentation
from .stop_line import StopLine

import qtawesome as qta


class RunningRedLightInterface(QDialog):
    """Vehicle Detection Interface"""

    def __init__(
        self,
        config: RunningRedLightConfig,
        config_vd: VehicleDetectionConfig,
        video_path,
    ):
        super().__init__()
        self.setWindowTitle("Running Red Light")

        self.config = deepcopy(config)
        self.video_path = video_path

        # Initialize Vehicle Detection
        self.vd = VehicleDetection(config_vd)

        # Initialize Running Red Light
        self.rrl = RunningRedLight(config)

        cfg_annotator = PipelineConfig()
        self.annotator = Annotator(cfg_annotator)
        self.annotator.vd_config = config_vd
        self.annotator.rrl_config = config

        self.layout = QGridLayout()

        self.layout.addLayout(self.set_left_layout(), 0, 0, 1, 2, Qt.AlignLeft)
        self.layout.addLayout(self.set_right_layout(), 0, 2, 1, 6, Qt.AlignCenter)
        self.layout.setSizeConstraint(QLayout.SetFixedSize)
        self.setLayout(self.layout)

        self.config_init()
        self.setWindowModality(Qt.ApplicationModal)

    def config_init(self):
        # Video Initialize
        self.vid = cv2.VideoCapture(self.video_path)
        print(int(self.vid.get(cv2.CAP_PROP_POS_FRAMES)))
        _, img = self.vid.read()

        if self.vid.isOpened() == False:
            print("Error opening video file!")
        else:
            total_frame = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_slider.setMaximum(total_frame - 1)

        img = self.convert_cv_qt(img)
        self.image_frame.setPixmap(QPixmap.fromImage(img))

    #     # Detection config
    #     self.model_line.setText(self.config.model_path)
    #     self.inference_size_spin_box.setValue(self.config.imgsz)
    #     self.min_conf_spin_box.setValue(self.config.conf_thres)
    #     self.min_iou_spin_box.setValue(self.config.iou_thres)

    #     # Tracker config
    #     self.iou_thres_spinbox.setValue(self.config.sort_iou_thres)
    #     self.min_hits_spinbox.setValue(self.config.sort_min_hits)
    #     self.max_age_spinbox.setValue(self.config.sort_max_age)

    #     # Video Slider config
    #     total_frame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)

    def set_right_layout(self):
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.set_image_layout())
        v_layout.addLayout(self.set_slider_and_inference())
        return v_layout

    def set_left_layout(self):
        v_left_layout = QVBoxLayout()

        self.stopline_button = QPushButton("Stop Line")
        self.stopline_button.clicked.connect(self.set_stop_line)

        self.apply_configuration_button = QPushButton(
            qta.icon("fa5s.check"), "Apply Configuration"
        )
        self.apply_configuration_button.clicked.connect(self.apply_config)
        self.save_configuration_button = QPushButton(
            qta.icon("mdi6.content-save-check"), "Close and Save Configuration"
        )
        self.save_configuration_button.clicked.connect(self.save_config)

        v_left_layout.addWidget(self.set_traffic_light_configuration_layout())
        v_left_layout.addWidget(self.stopline_button)
        v_left_layout.addStretch(1)
        v_left_layout.addWidget(self.apply_configuration_button)
        v_left_layout.addWidget(self.save_configuration_button)
        v_left_layout.addStretch(1)

        return v_left_layout

    def set_traffic_light_configuration_layout(self):
        tlc_layout = QVBoxLayout()
        tlc_groupbox = QGroupBox("Traffic Light Configuration")

        self.tlc_area = QPushButton("Traffic Light Area")
        self.tlc_area.clicked.connect(self.set_traffic_light_area)

        self.tlc_red_button = QPushButton("Red Light")
        self.tlc_red_button.clicked.connect(self.set_red_light_config)

        self.tlc_yellow_button = QPushButton("Yellow Light")
        self.tlc_yellow_button.clicked.connect(self.set_yellow_light_config)

        self.tlc_green_button = QPushButton("Green Light")
        self.tlc_green_button.clicked.connect(self.set_green_light_config)

        tlc_layout.addWidget(self.tlc_area)
        tlc_layout.addWidget(self.tlc_red_button)
        tlc_layout.addWidget(self.tlc_yellow_button)
        tlc_layout.addWidget(self.tlc_green_button)

        tlc_groupbox.setLayout(tlc_layout)

        return tlc_groupbox

    def set_stop_line(self):
        current_stopline = self.config.stop_line
        self.stop_line = StopLine(self.video_path, current_stopline)
        self.stop_line.exec_()

        if self.stop_line.result() == 1:
            self.config.stop_line = self.stop_line.get_stopline()
            print(self.config.stop_line)

    def set_traffic_light_area(self):
        self.detection_area = DetectionArea(
            self.video_path, self.config.traffic_light_area
        )
        self.detection_area.exec_()

        if self.detection_area.result() == 1:
            self.config.traffic_light_area = self.detection_area.get_detection_area()

    def set_red_light_config(self):
        self.red_config = ColorSegmentation(
            self.config, self.video_path, light_type="red"
        )
        self.red_config.exec_()

        if self.red_config.result() == 1:
            self.config = self.red_config.config

    def set_green_light_config(self):
        self.green_config = ColorSegmentation(
            self.config, self.video_path, light_type="green"
        )
        self.green_config.exec_()

        if self.green_config.result() == 1:
            self.config = self.green_config.config

    def set_yellow_light_config(self):
        self.yellow_config = ColorSegmentation(
            self.config, self.video_path, light_type="yellow"
        )
        self.yellow_config.exec_()

        if self.yellow_config.result() == 1:
            self.config = self.yellow_config.config

    # def set_tracker_layout(self):
    #     group_tracker = QGroupBox("Vehicle Tracker (SORT)")
    #     tracker_layout = QGridLayout()

    #     self.iou_thres_spinbox = QDoubleSpinBox()
    #     self.iou_thres_spinbox.setMaximum(1)
    #     self.iou_thres_spinbox.setMinimum(0)
    #     self.iou_thres_spinbox.setSingleStep(0.01)

    #     self.min_hits_spinbox = QDoubleSpinBox()
    #     self.min_hits_spinbox.setMaximum(100)
    #     self.min_hits_spinbox.setMinimum(1)
    #     self.min_hits_spinbox.setSingleStep(1)

    #     self.max_age_spinbox = QDoubleSpinBox()
    #     self.max_age_spinbox.setMaximum(200)
    #     self.max_age_spinbox.setMinimum(1)
    #     self.max_age_spinbox.setSingleStep(1)

    #     tracker_layout.addWidget(QLabel("IOU Thres"), 0, 0)
    #     tracker_layout.addWidget(self.iou_thres_spinbox, 0, 1)
    #     tracker_layout.addWidget(QLabel("Min Hits"), 1, 0)
    #     tracker_layout.addWidget(self.min_hits_spinbox, 1, 1)
    #     tracker_layout.addWidget(QLabel("Max Age"), 2, 0)
    #     tracker_layout.addWidget(self.max_age_spinbox, 2, 1)

    #     tracker_layout.setRowStretch(3, 1)

    #     group_tracker.setLayout(tracker_layout)

    #     return group_tracker

    def set_image_layout(self):
        self.image_frame = QLabel()
        self.image = cv2.imread("samples/huggingface.png")

        self.image = self.convert_cv_qt(self.image)
        self.image_frame.setPixmap(QPixmap.fromImage(self.image))
        return self.image_frame

    def convert_cv_qt(self, cv_img):
        print(cv_img.shape)
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(10000, 500, Qt.KeepAspectRatio)
        return p

    def set_slider_and_inference(self):
        h_layout = QHBoxLayout()

        self.video_slider = QSlider()
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(1)
        self.video_slider.setSingleStep(1)
        self.video_slider.setOrientation(Qt.Horizontal)
        self.video_slider.valueChanged.connect(self.update_frame)

        self.inference_button = QPushButton("Inference Frame")
        self.inference_button.clicked.connect(self.inference_image)

        h_layout.addWidget(self.video_slider, 8)
        h_layout.addWidget(self.inference_button)

        return h_layout

    def update_frame(self):
        frame_idx = self.video_slider.value()
        self.vid.set(1, frame_idx)
        _, self.current_img = self.vid.read()
        img = self.convert_cv_qt(self.current_img)
        self.image_frame.setPixmap(QPixmap.fromImage(img))

    def apply_config(self):
        self.rrl.config = self.config
        self.annotator.rrl_config = self.config

    def save_config(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Save Running Red Light Configuration ?")

        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        response = msg.exec_()

        if response == QMessageBox.Save:
            self.accept()

        elif response == QMessageBox.Cancel:
            pass

    def set_detection_area(self):
        self.detection_area = DetectionArea(self.video_path, self.config.detection_area)
        self.detection_area.exec_()

        if self.detection_area.result() == 1:
            self.config.detection_area = self.detection_area.get_detection_area()

    def inference_image(self):
        # Vehicle Detection Task
        self.vd.reset_tracker()
        result = self.vd.update(self.current_img)

        # Running Red Light Task
        violate = self.rrl.update(result, self.current_img)
        state = self.rrl.state

        # Annotate Image
        img_annotate = self.annotator.vehicle_detection(self.current_img, result)
        img_annotate = self.annotator.running_red_light(
            img_annotate, result, violate, state
        )

        img = self.convert_cv_qt(img_annotate)
        self.image_frame.setPixmap(QPixmap.fromImage(img))
