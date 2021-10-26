import cv2
import qimage2ndarray
import time

from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import QTimer, QWaitCondition, Qt, Slot
from tvdr.utils import Parameter
from tvdr.core import YOLOModel, yolo


class MainLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QtWidgets.QFormLayout()
        self.parameter = Parameter()
        self.configuration_layout = QtWidgets.QVBoxLayout()
        self.set_configuration_layout()

        self.image_layout = QtWidgets.QVBoxLayout()
        self.set_image_layout()

        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.addLayout(self.configuration_layout)
        self.h_layout.addLayout(self.image_layout)

        self.setLayout(self.h_layout)

        # Initialize YOLO Model
        self.yolo = YOLOModel(device="cpu")
        # self.yolo.update_parameters(0.0, 0.0, 640)

    def set_configuration_layout(self):
        self.configuration_layout.addWidget(self.video_configuration())
        self.configuration_layout.addWidget(self.model_yolo_configuration())
        self.configuration_layout.addWidget(self.main_configuration())
        self.configuration_layout.addWidget(self.control_configuration())

    def set_image_layout(self):
        self.image_frame = QtWidgets.QLabel()
        self.image = cv2.imread("samples/file-20200803-24-50u91u.jpg")
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
        self.lineedit_video_path.setContentsMargins(0, 0, 0, 0)
        self.button_video_load = QtWidgets.QPushButton("Load Video")
        self.button_video_load.clicked.connect(self.load_video)

        self.groupbox_layout.addWidget(self.lineedit_video_path)
        self.groupbox_layout.addWidget(self.button_video_load)

        self.groupbox_video = QtWidgets.QGroupBox("Video Data")
        self.groupbox_video.setContentsMargins(0, 0, 0, 0)
        self.groupbox_video.setMaximumHeight(80)

        self.groupbox_video.setLayout(self.groupbox_layout)

        return self.groupbox_video

    def model_yolo_configuration(self):
        # Combo Box YOLO
        self.yolomodel_vlayout = QtWidgets.QVBoxLayout()
        self.yolomodel_vlayout.setSpacing(0)
        self.yolomodel_vlayout.setMargin(0)
        self.yolomodel_vlayout.setContentsMargins(0, 0, 0, 0)

        self.yolomodel_layout = QtWidgets.QHBoxLayout()
        self.yolomodel_layout.setSpacing(0)
        self.yolomodel_layout.setMargin(0)
        self.yolomodel_layout.setContentsMargins(0, 0, 0, 0)

        self.combobox_yolo = QtWidgets.QComboBox()

        for model in self.parameter.yolo_model_dict.keys():
            self.combobox_yolo.addItem(model)

        self.current_yolo_model_label = QtWidgets.QLabel("  Current Model : Not Loaded")

        self.button_model_yolo_load = QtWidgets.QPushButton("Load Model")
        self.button_model_yolo_load.clicked.connect(self.load_model)

        self.yolomodel_layout.addWidget(self.combobox_yolo)
        self.yolomodel_layout.addWidget(self.button_model_yolo_load)

        self.groupbox_yolomodel = QtWidgets.QGroupBox("YOLO Model")
        self.groupbox_yolomodel.setMaximumHeight(80)

        self.yolomodel_vlayout.addLayout(self.yolomodel_layout)
        self.yolomodel_vlayout.addWidget(self.current_yolo_model_label)

        self.groupbox_yolomodel.setLayout(self.yolomodel_vlayout)

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
            self.parameter.video_path = fileName

    @Slot()
    def load_model(self):
        print("LOG : Loaded Model Clicked")
        print("Model Type : {}".format(self.combobox_yolo.currentText()))
        self.yolo.load_model(
            self.parameter.yolo_model_dict[self.combobox_yolo.currentText()]
        )

    @Slot()
    def update_parameters(self):
        print("\nUpdating Parameters")
        print("video_path : {}".format(self.lineedit_video_path.text()))
        print("iou_threshold : {}".format(self.iou_threshold_spinbox.value()))
        print("object_threshold : {}".format(self.object_threshold_spinbox.value()))

    @Slot()
    def start_inference(self):
        print("start inference")
        if self.parameter.video_path == "":
            print("Please load video first!")

        else:
            print("Video loaded!")
            self.vid = cv2.VideoCapture(self.parameter.video_path)
            if self.vid.isOpened() == False:
                print("Error opening video file!")
            else:
                self.parameter.video_fps = self.vid.get(cv2.CAP_PROP_FPS)
                self.parameter.frame_count = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

                print(self.parameter.video_fps)
                print(self.parameter.frame_count)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000.0 / 30)

    @Slot()
    def stop_inference(self):
        print("stop inference")
        self.timer.stop()

    def update_frame(self):
        ret, frame = self.vid.read()
        # cv2.cv.CV_B
        # frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        frame_resize = cv2.resize(
            frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5))
        )
        new_frame = self.yolo.inference_frame(frame_resize)
        img = QtGui.QImage(
            new_frame,
            new_frame.shape[1],
            new_frame.shape[0],
            QtGui.QImage.Format_RGB888,
        )
        # image = qimage2ndarray.array2qimage(frame)
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(img))
        # self.image_layout.addWidget(self.image_frame)


class DatabaseLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QFormLayout()
        self.text = QtWidgets.QTextEdit("Test2")
        self.layout.addWidget(self.text)
        self.setLayout(self.layout)
