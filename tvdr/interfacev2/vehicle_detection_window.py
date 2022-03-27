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

from tvdr.core import VehicleDetectionConfig, VehicleDetection
from tvdr.utils import Annotator
from tvdr.core import PipelineConfig

from .detection_area import DetectionArea

import qtawesome as qta


class VehicleDetectionInterface(QDialog):
    """Vehicle Detection Interface"""

    def __init__(self, config: VehicleDetectionConfig, video_path: str):
        super().__init__()
        self.setWindowTitle("Vehicle Detection and Tracker Configuration")

        self.config = deepcopy(config)
        self.video_path = video_path

        self.vd = VehicleDetection(config)

        cfg_annotator = PipelineConfig()
        self.annotator = Annotator(cfg_annotator)
        self.annotator.vd_config = config

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

        # Detection config
        self.model_line.setText(self.config.model_path)
        self.inference_size_spin_box.setValue(self.config.imgsz)
        self.min_conf_spin_box.setValue(self.config.conf_thres)
        self.min_iou_spin_box.setValue(self.config.iou_thres)

        # Tracker config
        self.iou_thres_spinbox.setValue(self.config.sort_iou_thres)
        self.min_hits_spinbox.setValue(self.config.sort_min_hits)
        self.max_age_spinbox.setValue(self.config.sort_max_age)

        # Video Slider config
        total_frame = self.vid.get(cv2.CAP_PROP_POS_FRAMES)

    def set_right_layout(self):
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.set_image_layout())
        v_layout.addLayout(self.set_slider_and_inference())
        return v_layout

    def set_left_layout(self):
        v_left_layout = QVBoxLayout()
        v_left_layout.addWidget(self.set_detection_layout())
        v_left_layout.addWidget(self.set_tracker_layout())

        self.apply_configuration_button = QPushButton(
            qta.icon("fa5s.check"), "Apply Configuration"
        )
        self.apply_configuration_button.clicked.connect(self.apply_config)

        self.save_configuration_button = QPushButton(
            qta.icon("mdi6.content-save-check"), "Close and Save Configuration"
        )
        self.save_configuration_button.clicked.connect(self.save_config)

        v_left_layout.addWidget(self.apply_configuration_button)
        v_left_layout.addWidget(self.save_configuration_button)
        v_left_layout.addStretch(1)

        return v_left_layout

    def set_detection_layout(self):
        group_detection = QGroupBox("Vehicle Detection")
        detection_layout = QGridLayout()

        self.model_line = QLineEdit()
        self.model_load_button = QPushButton("Load Model")
        self.model_load_button.clicked.connect(self.load_model)
        detection_layout.addWidget(self.model_line, 0, 0)
        detection_layout.addWidget(self.model_load_button, 0, 1)

        self.inference_size_spin_box = QDoubleSpinBox()
        self.inference_size_spin_box.setMinimum(200)
        self.inference_size_spin_box.setMaximum(1000)
        self.inference_size_spin_box.setSingleStep(100)
        detection_layout.addWidget(QLabel("Inference Size"), 1, 0)
        detection_layout.addWidget(self.inference_size_spin_box, 1, 1)

        self.min_conf_spin_box = QDoubleSpinBox()
        self.min_conf_spin_box.setMaximum(1)
        self.min_conf_spin_box.setMinimum(0)
        self.min_conf_spin_box.setSingleStep(0.01)
        detection_layout.addWidget(QLabel("Min Confidence"), 2, 0)
        detection_layout.addWidget(self.min_conf_spin_box, 2, 1)

        self.min_iou_spin_box = QDoubleSpinBox()
        self.min_iou_spin_box.setMaximum(1)
        self.min_iou_spin_box.setMinimum(0)
        self.min_iou_spin_box.setSingleStep(0.01)
        detection_layout.addWidget(QLabel("Min IOU"), 3, 0)
        detection_layout.addWidget(self.min_iou_spin_box, 3, 1)

        self.detection_area_button = QPushButton(
            qta.icon("fa5s.draw-polygon"), "Detection Area"
        )
        self.detection_area_button.clicked.connect(self.set_detection_area)
        detection_layout.addWidget(self.detection_area_button, 4, 0, 1, 2)

        detection_layout.setRowStretch(5, 1)

        group_detection.setLayout(detection_layout)

        return group_detection

    def set_tracker_layout(self):
        group_tracker = QGroupBox("Vehicle Tracker (SORT)")
        tracker_layout = QGridLayout()

        self.iou_thres_spinbox = QDoubleSpinBox()
        self.iou_thres_spinbox.setMaximum(1)
        self.iou_thres_spinbox.setMinimum(0)
        self.iou_thres_spinbox.setSingleStep(0.01)

        self.min_hits_spinbox = QDoubleSpinBox()
        self.min_hits_spinbox.setMaximum(100)
        self.min_hits_spinbox.setMinimum(1)
        self.min_hits_spinbox.setSingleStep(1)

        self.max_age_spinbox = QDoubleSpinBox()
        self.max_age_spinbox.setMaximum(200)
        self.max_age_spinbox.setMinimum(1)
        self.max_age_spinbox.setSingleStep(1)

        tracker_layout.addWidget(QLabel("IOU Thres"), 0, 0)
        tracker_layout.addWidget(self.iou_thres_spinbox, 0, 1)
        tracker_layout.addWidget(QLabel("Min Hits"), 1, 0)
        tracker_layout.addWidget(self.min_hits_spinbox, 1, 1)
        tracker_layout.addWidget(QLabel("Max Age"), 2, 0)
        tracker_layout.addWidget(self.max_age_spinbox, 2, 1)

        tracker_layout.setRowStretch(3, 1)

        group_tracker.setLayout(tracker_layout)

        return group_tracker

    def set_image_layout(self):
        self.image_frame = QLabel()
        self.image = cv2.imread("samples/huggingface.png")

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
        self.config.imgsz = int(self.inference_size_spin_box.value())
        self.config.conf_thres = self.min_conf_spin_box.value()
        self.config.iou_thres = self.min_iou_spin_box.value()
        self.config.sort_iou_thres = self.iou_thres_spinbox.value()
        self.config.sort_max_age = self.max_age_spinbox.value()
        self.config.sort_min_hits = self.min_hits_spinbox.value()
        self.vd.config = self.config

        if self.config.model_path != self.model_line.text():
            self.config.model_path = self.model_line.text()
            self.vd.config = self.config
            self.vd.load_model()

        self.annotator.vd_config = self.config

    def save_config(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Save Vehicle Detection And Tracker Configuration ?")

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
        self.vd.reset_tracker()
        result = self.vd.update(self.current_img)
        img_annotate = self.annotator.vehicle_detection(self.current_img, result)

        img = self.convert_cv_qt(img_annotate)
        self.image_frame.setPixmap(QPixmap.fromImage(img))

    def load_model(self):
        # Open window for select helmet violation model
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Vehicle Detection Model",
            "",
            "Model (*.pt)",
            options=options,
        )
        if fileName:
            self.model_line.setText(fileName)
