from re import A
import qtawesome as qta
import cv2

from copy import deepcopy
from PySide2 import QtWidgets
from PySide2.QtWidgets import (
    QGroupBox,
    QVBoxLayout,
    QGridLayout,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QDoubleSpinBox,
    QLabel,
    QLayout,
    QDialog,
    QHBoxLayout,
    QSlider,
    QFileDialog,
    QMessageBox,
)
from PySide2.QtCore import Qt
from PySide2.QtGui import QPixmap, QImage

from tvdr.core import (
    HelmetViolation,
    HelmetViolationConfig,
    VehicleDetectionConfig,
    VehicleDetection,
    PipelineConfig,
)

from tvdr.utils import Annotator


class HelmetViolationInterface(QDialog):
    def __init__(
        self,
        config: HelmetViolationConfig,
        config_vd: VehicleDetectionConfig,
        video_path: str,
    ):
        super().__init__()
        self.setWindowTitle("Helmet Violation Configuration")

        self.config = deepcopy(config)
        self.config_vd = config_vd
        self.video_path = video_path

        self.hv = HelmetViolation(self.config)
        self.vd = VehicleDetection(self.config_vd)

        # Initialize Annotator
        self.annotator_config = PipelineConfig()
        self.annotator_config.vd_config = config_vd
        self.annotator = Annotator(self.annotator_config)

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
        grid_left_layout = QGridLayout()

        self.model_line = QLineEdit()
        self.model_load_button = QPushButton("Load Model")
        self.model_load_button.clicked.connect(self.load_model)

        inference_size_label = QLabel("Inference Size")
        self.inference_size_spin_box = QDoubleSpinBox()
        self.inference_size_spin_box.setMinimum(200)
        self.inference_size_spin_box.setMaximum(1000)
        self.inference_size_spin_box.setSingleStep(10)

        min_conf_label = QLabel("Min Conf Thres")
        self.min_conf_spin_box = QDoubleSpinBox()
        self.min_conf_spin_box.setMaximum(1)
        self.min_conf_spin_box.setMinimum(0.1)
        self.min_conf_spin_box.setSingleStep(0.01)

        min_age_label = QLabel("Min Age")
        self.min_age_spin_box = QDoubleSpinBox()
        self.min_age_spin_box.setMaximum(180)
        self.min_age_spin_box.setMinimum(1)
        self.min_age_spin_box.setSingleStep(1)

        detect_interval_label = QLabel("Detect Interval")
        self.detect_interval_spin_box = QDoubleSpinBox()
        self.detect_interval_spin_box.setMaximum(60)
        self.detect_interval_spin_box.setMinimum(1)
        self.detect_interval_spin_box.setSingleStep(1)

        grid_left_layout.addWidget(self.model_line, 0, 0)
        grid_left_layout.addWidget(self.model_load_button, 0, 1)

        grid_left_layout.addWidget(inference_size_label, 1, 0)
        grid_left_layout.addWidget(self.inference_size_spin_box, 1, 1)
        grid_left_layout.addWidget(min_conf_label, 2, 0)
        grid_left_layout.addWidget(self.min_conf_spin_box, 2, 1)
        grid_left_layout.addWidget(min_age_label, 3, 0)
        grid_left_layout.addWidget(self.min_age_spin_box, 3, 1)
        grid_left_layout.addWidget(min_age_label, 4, 0)
        grid_left_layout.addWidget(self.min_age_spin_box, 4, 1)
        grid_left_layout.addWidget(detect_interval_label, 5, 0)
        grid_left_layout.addWidget(self.detect_interval_spin_box, 5, 1)
        grid_left_layout.setRowStretch(6, 1)
        groupbox_left.setLayout(grid_left_layout)

        self.apply_configuration_button = QPushButton(
            qta.icon("fa5s.check"), "Apply Configuration"
        )
        self.apply_configuration_button.clicked.connect(self.apply_config)

        self.save_configuration_button = QPushButton(
            qta.icon("mdi6.content-save-check"), "Close and Save Configuration"
        )
        self.save_configuration_button.clicked.connect(self.save_config)

        v_left_layout.addWidget(groupbox_left)
        v_left_layout.addWidget(self.apply_configuration_button)
        v_left_layout.addWidget(self.save_configuration_button)
        v_left_layout.addStretch(1)

        return v_left_layout

    def set_right_layout(self):
        v_layout = QVBoxLayout()
        v_layout.addWidget(self.set_image_layout())
        v_layout.addLayout(self.set_slider_and_inference())
        return v_layout

    def config_init(self):
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

        self.model_line.setText(self.config.model_path)
        self.inference_size_spin_box.setValue(self.config.imgsz)
        self.min_conf_spin_box.setValue(self.config.conf_thres)
        self.min_age_spin_box.setValue(self.config.min_age)
        self.detect_interval_spin_box.setValue(self.config.detect_interval)

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

    def apply_config(self):
        self.config.imgsz = int(self.inference_size_spin_box.value())
        self.config.conf_thres = self.min_conf_spin_box.value()
        self.config.min_age = int(self.min_age_spin_box.value())
        self.config.detect_interval = int(self.detect_interval_spin_box.value())

        if self.config.model_path != self.model_line.text():
            self.config.model_path = self.model_line.text()
            self.hv.config = self.config
            self.hv.load_model()

    def save_config(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Save Helmet Violation Configuration ?")

        msg.setStandardButtons(QMessageBox.Save | QMessageBox.Cancel)
        response = msg.exec_()

        if response == QMessageBox.Save:
            self.accept()

        elif response == QMessageBox.Cancel:
            pass

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

    def inference_image(self):

        # Vehicle Detection Task
        self.vd.reset_tracker()
        result = self.vd.update(self.current_img)

        # Helmet Violation Detection Task
        filter_vehicle = self.hv.motorcycle_and_bicycle_filtering(result)
        violate = self.hv.detect_violation(self.current_img, filter_vehicle)

        img_annotate = self.annotator.vehicle_detection(self.current_img, result)
        img_annotate = self.annotator.helmet_violation(img_annotate, result, violate)

        img = self.convert_cv_qt(img_annotate)
        self.image_frame.setPixmap(QPixmap.fromImage(img))

        print(violate)
