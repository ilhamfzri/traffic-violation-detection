import cv2

from PySide2 import QtWidgets, QtGui
from PySide2.QtCore import QWaitCondition, Slot


class MainLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QFormLayout()

        self.configuration_layout = QtWidgets.QVBoxLayout()
        self.set_configuration_layout()

        self.image_layout = QtWidgets.QVBoxLayout()
        # self.set_image_layout()

        self.h_layout = QtWidgets.QHBoxLayout()
        self.h_layout.addLayout(self.configuration_layout)
        self.h_layout.addLayout(self.image_layout)

        self.setLayout(self.h_layout)

    def set_configuration_layout(self):
        self.groupbox_layout = QtWidgets.QHBoxLayout()

        self.lineedit_video_path = QtWidgets.QLineEdit()
        self.button_video_load = QtWidgets.QPushButton("Load Video")
        self.button_video_load.clicked.connect(self.load_video)

        self.groupbox_layout.addWidget(self.lineedit_video_path)
        self.groupbox_layout.addWidget(self.button_video_load)

        self.groupbox_video = QtWidgets.QGroupBox("Video Data")
        self.groupbox_video.setLayout(self.groupbox_layout)

        self.configuration_layout.addWidget(self.groupbox_video)

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

    @Slot()
    def load_video(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Videos Files (*.mp4)",
            options=options,
        )
        if fileName:
            print(fileName)


class DatabaseLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QtWidgets.QFormLayout()
        self.text = QtWidgets.QTextEdit("Test2")
        self.layout.addWidget(self.text)
        self.setLayout(self.layout)
