import sys
import random

from tvdr.interface.widget import Widget
from PySide2 import QtCore, QtWidgets, QtGui


class MainWindows:
    def __init__(self):
        self.app = QtWidgets.QApplication([])
        self.widget = Widget()

        self.widget.setMaximumWidth(1500)
        self.widget.showMaximized()
        self.widget.show()

        sys.exit(self.app.exec_())
