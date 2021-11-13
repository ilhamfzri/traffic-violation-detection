#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 UGM

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Ilham Fazri - ilhamfazri3rd@gmail.com

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
