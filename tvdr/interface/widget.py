from PySide2 import QtWidgets
from tvdr.interface import menu_bar
from tvdr.interface.menu_bar import MenuBar


class Widget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Traffic Violation Detection and Recognition")
        self.menuBar = MenuBar()
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout.addLayout(self.menuBar)

        self.setLayout(self.mainLayout)
