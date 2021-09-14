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
