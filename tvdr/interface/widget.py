from PySide2 import QtWidgets
from tvdr.interface import menu_bar
from tvdr.interface.menu_bar import MenuBar


class Widget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("UGM 2021 : Traffic Violation Detection and Recognition")
        self.menuBar = MenuBar()
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.mainLayout.addLayout(self.menuBar)
        self.mainLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.setLayout(self.mainLayout)
