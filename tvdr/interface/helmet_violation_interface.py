import logging
from turtle import color
import cv2
import numpy as np

from tvdr.utils.params import Parameter
from tvdr.core.vehicle_detection import VehicleDetection
from tvdr.core.violation_recorder_class import ViolationRecorderMain
from tvdr.core.helmet_violation_classifier import HelmetViolationDetectionClassifier
from PySide2 import QtWidgets, QtCore, QtGui

logging_root = "Helmet Violation Interface"
logging.basicConfig(level=logging.INFO)


class HelmetViolationInterface(QtWidgets.QDialog):
    """Helmet Violation Configuration Interface"""

    def __init__(self, parameter: Parameter):
        super().__init__()
        self.parameter = parameter

        # Initialize Vehicle Detection
        self.vd = VehicleDetection(parameter)

        # Initialize Violation Recorder
        # Need it for visualize detection result
        self.vr = ViolationRecorderMain(parameter)

        # Initialize Helmet Violation Detection
        self.hvd = HelmetViolationDetectionClassifier(parameter)

        self.size_window_h = 600

        config_layout = QtWidgets.QVBoxLayout()
        config_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        config_layout.addWidget(self.set_model_layout())
        config_layout.addWidget(self.set_main_config_layout())
        config_layout.addWidget(self.set_control_layout())

        self.close_and_save_button = QtWidgets.QPushButton("Save Parameters")
        self.close_and_save_button.clicked.connect(self.save_configuration)

        config_layout.addWidget(self.close_and_save_button)
        config_layout.addStretch()

        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.main_layout.addLayout(config_layout)
        self.main_layout.addLayout(self.set_inference_layout())

        self.update_parameter_layout()
        self.setLayout(self.main_layout)
        self.setWindowModality(QtCore.Qt.ApplicationModal)

    def convert_cv_qt(self, cv_img):
        # Convert cv_img to qt pixmap
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        ratio_img = cv_img.shape[0] / self.size_window_h
        h_new = h / ratio_img
        w_new = w / ratio_img
        p = convert_to_Qt_format.scaled(w_new, h_new, QtCore.Qt.KeepAspectRatio)

        return QtGui.QPixmap.fromImage(p)

    def set_control_layout(self):
        # Control layout
        self.control_inference = QtWidgets.QPushButton("Inference")
        self.control_reset = QtWidgets.QPushButton("Reset")

        self.control_inference.clicked.connect(self.predict_image)
        self.control_reset.clicked.connect(self.reset_image)

        self.control_h_layout = QtWidgets.QHBoxLayout()
        self.control_h_layout.addWidget(self.control_inference)
        self.control_h_layout.addWidget(self.control_reset)

        self.control_groupbox = QtWidgets.QGroupBox("Control")
        self.control_groupbox.setLayout(self.control_h_layout)
        return self.control_groupbox

    def set_inference_layout(self):
        # Read video
        self.vid = cv2.VideoCapture(self.parameter.video_path)
        _, self.frame_data = self.vid.read()

        # Frame count
        self.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

        # Set video slider
        self.slider_video = QtWidgets.QSlider()
        self.slider_video.setMinimum(0)
        self.slider_video.setMaximum(self.frame_count)
        self.slider_video.setSingleStep(1)
        self.slider_video.setOrientation(QtCore.Qt.Horizontal)
        self.slider_video.valueChanged.connect(self.update_frame)

        # Set image viewer
        self.image_label = QtWidgets.QLabel()
        self.image_label.setPixmap(self.convert_cv_qt(self.frame_data))

        self.inference_v_layout = QtWidgets.QVBoxLayout()
        self.inference_v_layout.addWidget(self.image_label)
        self.inference_v_layout.addWidget(self.slider_video)

        return self.inference_v_layout

    def update_frame(self):
        # Get current frame data from selected slider position
        frame_position = self.slider_video.value()
        self.vid.set(1, frame_position)
        _, self.frame_data = self.vid.read()
        self.image_label.setPixmap(self.convert_cv_qt(self.frame_data))

    def set_model_layout(self):
        # Model Path
        self.model_lineedit = QtWidgets.QLineEdit()
        self.model_load_button = QtWidgets.QPushButton("Load Model")
        self.model_load_button.clicked.connect(self.load_model_path)

        self.model_hlayout = QtWidgets.QHBoxLayout()
        self.model_hlayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.model_hlayout.addWidget(self.model_lineedit)
        self.model_hlayout.addWidget(self.model_load_button)

        self.model_groupbox = QtWidgets.QGroupBox("Model Path")
        self.model_groupbox.setLayout(self.model_hlayout)

        return self.model_groupbox

    def load_model_path(self):
        # Open window for select helmet violation model
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Model (*.pt)",
            options=options,
        )
        if fileName:
            self.model_lineedit.setText(fileName)
            self.parameter.hv_model_path = fileName

    def set_main_config_layout(self):
        # Image Size
        self.inference_size_layout = QtWidgets.QHBoxLayout()
        self.inference_size_spin_box = QtWidgets.QDoubleSpinBox()
        self.inference_size_spin_box.setMinimum(0)
        self.inference_size_spin_box.setMaximum(1000)
        self.inference_size_layout.addWidget(QtWidgets.QLabel("Inference Size"))
        self.inference_size_layout.addWidget(self.inference_size_spin_box)

        # Minimum Confidence Threshold
        self.minimum_confidence_layout = QtWidgets.QHBoxLayout()
        self.minimum_confidence_spinbox = QtWidgets.QDoubleSpinBox()
        self.minimum_confidence_spinbox.setMaximum(1)
        self.minimum_confidence_spinbox.setMinimum(0)
        self.minimum_confidence_spinbox.setSingleStep(0.01)

        self.minimum_confidence_spinbox.setValue(0.5)
        self.minimum_confidence_layout.addWidget(QtWidgets.QLabel("Minimum Confidence"))
        self.minimum_confidence_layout.addWidget(self.minimum_confidence_spinbox)

        # Detect Interval
        self.detect_interval_layout = QtWidgets.QHBoxLayout()
        self.detect_interval_spinbox = QtWidgets.QDoubleSpinBox()
        self.detect_interval_spinbox.setMaximum(1)
        self.detect_interval_spinbox.setMinimum(0)
        self.detect_interval_spinbox.setSingleStep(0.01)
        self.detect_interval_layout.addWidget(QtWidgets.QLabel("Detect Interval"))
        self.detect_interval_layout.addWidget(self.detect_interval_spinbox)

        # Minimum Age
        self.minimum_age_layout = QtWidgets.QHBoxLayout()
        self.minimum_age_spinbox = QtWidgets.QDoubleSpinBox()
        self.minimum_age_spinbox.setMaximum(100)
        self.minimum_age_spinbox.setMinimum(0)
        self.minimum_age_spinbox.setSingleStep(1)
        self.minimum_age_layout.addWidget(QtWidgets.QLabel("Minimum Age"))
        self.minimum_age_layout.addWidget(self.minimum_age_spinbox)

        # Apply Parameters
        self.apply_parameter_button = QtWidgets.QPushButton("Apply Parameters")
        self.apply_parameter_button.clicked.connect(self.set_apply_parameter)

        # Combine widgets
        self.main_config_layout = QtWidgets.QVBoxLayout()
        self.main_config = QtWidgets.QGroupBox("Parameters Configuration")

        self.main_config_layout.addLayout(self.inference_size_layout)
        self.main_config_layout.addLayout(self.minimum_confidence_layout)
        self.main_config_layout.addLayout(self.minimum_confidence_layout)
        self.main_config_layout.addLayout(self.detect_interval_layout)
        self.main_config_layout.addLayout(self.minimum_age_layout)
        self.main_config_layout.addWidget(self.apply_parameter_button)

        self.main_config.setLayout(self.main_config_layout)

        return self.main_config

    def set_apply_parameter(self):
        # Apply parameters
        self.parameter.hv_imgsz = int(self.inference_size_spin_box.value())
        self.parameter.hv_min_conf = self.minimum_confidence_spinbox.value()
        self.parameter.hv_detect_interval = self.detect_interval_spinbox.value()
        self.parameter.hv_min_age = int(self.minimum_age_spinbox.value())

        self.update_parameter_layout()
        self.hvd.update_params(self.parameter)

    def update_parameter_layout(self):
        # To update layout visualization value
        self.model_lineedit.setText(self.parameter.hv_model_path)
        self.inference_size_spin_box.setValue(self.parameter.hv_imgsz)
        self.minimum_confidence_spinbox.setValue(self.parameter.hv_min_conf)
        self.detect_interval_spinbox.setValue(self.parameter.hv_detect_interval)
        self.minimum_age_spinbox.setValue(self.parameter.hv_min_age)

    def save_configuration(self):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setText("Do you want to save this helmet violation parameters?")
        msg.setDetailedText(
            f"""The details of paramaters are as follow:\n
        Model Path \t: {self.parameter.hv_model_path}\n
        Inference Size \t: {self.parameter.hv_imgsz}\n
        Object Threshold \t: {self.parameter.hv_conf}\n
        IOU Threshold \t: {self.parameter.hv_iou}\n
        Min Age Threshold \t: {self.parameter.hv_min_age}\n
        Padding Width Multiplier \t: {self.parameter.hv_pad_width_mul}\n
        Padding Height Multiplier \t: {self.parameter.hv_pad_height_mul}"
        """
        )

        msg.setStandardButtons(
            QtWidgets.QMessageBox.Save | QtWidgets.QMessageBox.Cancel
        )
        response = msg.exec_()

        if response == QtWidgets.QMessageBox.Save:
            logging.info(f"{logging_root}: Saved Parameters")
            self.accept()

        elif response == QtWidgets.QMessageBox.Cancel:
            logging.info(f"{logging_root} : Cancel Save Parameters")

    def predict_image(self):
        # Load vehicle detection model if not loaded
        if self.vd.model_loaded() != True:
            logging.info(f"{logging_root}: Load Vehicle Detection Model")
            self.vd.load_model()

        # Inference frame data to detect vehicle
        result = self.vd.predict(self.frame_data)

        print(result)

        # Add dummy ID for consistency
        # result_new = np.empty((0, result.shape[1] + 1))
        # id = 0
        # for obj in result:
        #     obj = np.append(obj, [id], axis=0)
        #     result_new = np.append(result_new, obj.reshape(1, 7), axis=0)
        #     id += 1

        # Filtering
        result_filter = self.hvd.motorcycle_and_bicycle_filtering(result)

        # Detect helmet violation
        result_violation = self.hvd.detect_violation(self.frame_data, result_filter)

        result_final = self.vr.detection_combiner(
            result_filter, helmet_violation_result=result_violation
        )

        # Annotate frame data
        img_annotate = self.vr.annotate_result(self.frame_data, result_final)

        self.image_label.setPixmap(self.convert_cv_qt(img_annotate))

    def reset_image(self):
        pass
