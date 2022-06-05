import sys

from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from .main_layout import MainLayout


class MainWindows:
    def __init__(self):
        self.app = QApplication([])
        QApplication.setStyle("Fusion")

        self.current_layout = QHBoxLayout()
        self.current_layout.addWidget(MainLayout())
        self.widget = QWidget()
        self.widget.setWindowTitle("TVDR : Traffic Violation Detection and Recognition")
        self.widget.setLayout(self.current_layout)
        self.widget.show()

        sys.exit(self.app.exec_())
