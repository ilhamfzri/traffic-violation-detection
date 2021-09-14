from PyQt5 import QtGui, QtCore, QtWidgets
import cv2
import sys


class DisplayImageWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.button = QtWidgets.QPushButton("Show picture")
        # self.button.clicked.connect(self.show_image)
        self.image_frame = QtWidgets.QLabel()

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.image_frame)

    #     self.setLayout(self.layout)

    # @QtCore.pyqtSlot()
    # def show_image(self):
    #     self.image = cv2.imread(
    #         "/media/hamz/Alpha/Semester7/Skripsi/TrafficViolationDetection/data/nabila.jpeg"
    #     )
    #     self.image = QtGui.QImage(
    #         self.image.data,
    #         self.image.shape[1],
    #         self.image.shape[0],
    #         QtGui.QImage.Format_RGB888,
    #     ).rgbSwapped()
    #     self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))
