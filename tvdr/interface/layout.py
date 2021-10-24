import cv2

from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import QWaitCondition, Slot
from tvdr.utils import Parameter
from tvdr.core import YOLOModel


class MainLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QFormLayout()
        self.parameter = Parameter()

        self.configuration_layout = QtWidgets.QVBoxLayout()
        self.set_configuration_layout()

        self.image_layout = QtWidgets.QVBoxLayout()

        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.addLayout(self.configuration_layout)
        self.h_layout.addLayout(self.image_layout)

        self.setLayout(self.h_layout)

        # Initialize YOLO Model
        self.yolo = YOLOModel()
        self.yolo.update_parameters(0.0, 0.0, 640)

    def set_configuration_layout(self):
        self.configuration_layout.addWidget(self.video_configuration())
        self.configuration_layout.addWidget(self.model_yolo_configuration())
        self.configuration_layout.addWidget(self.main_configuration())
        self.configuration_layout.addWidget(self.control_configuration())

    def set_image_layout(self):
        self.image_frame = QtWidgets.QLabel()
        self.image = cv2.imread(
            "/media/hamz/Alpha/Semester7/Skripsi/TrafficViolationDetection/data/nabila.jpeg"
        )
        self.image = QtGui.QImage(
            self.image.data,
            self.image.shape[1],
            self.image.shape[0],
            QtGui.QImage.Format_RGB888,
        ).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

        self.image_layout.addWidget(self.image_frame)

    def video_configuration(self):
        self.groupbox_layout = QtWidgets.QHBoxLayout()

        self.lineedit_video_path = QtWidgets.QLineEdit()
        self.button_video_load = QtWidgets.QPushButton("Load Video")
        self.button_video_load.clicked.connect(self.load_video)

        self.groupbox_layout.addWidget(self.lineedit_video_path)
        self.groupbox_layout.addWidget(self.button_video_load)

        self.groupbox_video = QtWidgets.QGroupBox("Video Data")
        self.groupbox_video.setLayout(self.groupbox_layout)

        return self.groupbox_video

    def model_yolo_configuration(self):
        self.yolomodel_layout = QtWidgets.QHBoxLayout()

        self.combobox_yolo = QtWidgets.QComboBox()

        for model in self.parameter.yolo_model_dict.keys():
            self.combobox_yolo.addItem(model)

        self.button_model_yolo_load = QtWidgets.QPushButton("Load Model")
        self.button_model_yolo_load.clicked.connect(self.load_model)

        self.yolomodel_layout.addWidget(self.combobox_yolo)
        self.yolomodel_layout.addWidget(self.button_model_yolo_load)

        self.groupbox_yolomodel = QtWidgets.QGroupBox("Load YOLO Model")
        self.groupbox_yolomodel.setLayout(self.yolomodel_layout)

        return self.groupbox_yolomodel

    def main_configuration(self):
        self.main_configuration_layout = QtWidgets.QVBoxLayout()
        self.main_configuration_layout.addLayout(self.object_threshold_configuration())
        self.main_configuration_layout.addLayout(self.iou_threshold_configuration())

        self.main_configuration_groupbox = QtWidgets.QGroupBox("Configuration")
        self.main_configuration_groupbox.setLayout(self.main_configuration_layout)

        return self.main_configuration_groupbox

    def object_threshold_configuration(self):
        self.object_threshold_layout = QtWidgets.QHBoxLayout()
        self.object_threshold_spinbox = QtWidgets.QDoubleSpinBox()
        self.object_threshold_spinbox.setMaximum(1)
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

    def control_configuration(self):
        self.control_groupbox = QtWidgets.QGroupBox("Control")

        self.button_update_parameter = QtWidgets.QPushButton("Update Parameters")
        self.button_update_parameter.clicked.connect(self.update_parameters)

        self.button_start = QtWidgets.QPushButton("Start")
        self.button_start.clicked.connect(self.start_inference)

        self.button_stop = QtWidgets.QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop_inference)

        control_v_layout = QtWidgets.QVBoxLayout()
        control_h_layout = QtWidgets.QHBoxLayout()

        control_h_layout.addWidget(self.button_start)
        control_h_layout.addWidget(self.button_stop)

        control_v_layout.addWidget(self.button_update_parameter)
        control_v_layout.addLayout(control_h_layout)

        self.control_groupbox.setLayout(control_v_layout)

        return self.control_groupbox

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
            print(fileName)

    @Slot()
    def load_model(self):
        print("model")

    @Slot()
    def update_parameters(self):
        print("\nUpdating Parameters")
        print("video_path : {}".format(self.lineedit_video_path.text()))
        print("iou_threshold : {}".format(self.iou_threshold_spinbox.value()))
        print("object_threshold : {}".format(self.object_threshold_spinbox.value()))

    @Slot()
    def start_inference(self):
        print("start inference")

    @Slot()
    def stop_inference(self):
        print("stop inference")


class DatabaseLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QFormLayout()
        self.text = QtWidgets.QTextEdit("Test2")
        self.layout.addWidget(self.text)
        self.setLayout(self.layout)
