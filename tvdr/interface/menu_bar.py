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
from tvdr.interface.layout import MainLayout, DatabaseLayout


class MenuBar(QtWidgets.QHBoxLayout):
    def __init__(self):
        super().__init__()

        self.mainLayout = MainLayout()
        self.databaseLayout = DatabaseLayout()

        self.tabBar = QtWidgets.QTabWidget()

        self.tabBar.addTab(self.mainLayout, "Main Menu")
        self.tabBar.addTab(self.databaseLayout, "Database")
        self.addWidget(self.tabBar)
        self.tabBar.setCurrentIndex(0)

        self.tab_changed()

    def current_index(self):
        return self.tabBar.currentIndex

    def tab_changed(self):
        print(self.tabBar.currentChanged)
