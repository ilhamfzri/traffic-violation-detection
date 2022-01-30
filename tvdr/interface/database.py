import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns

from PySide2 import QtWidgets, QtCore, QtGui
from PySide2.QtCore import Slot
from dataset.kitti_dataset_distribution import show_values
from tvdr.interface.video_viewer import VideoViewer
from subprocess import Popen


class DatabaseLayout(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.db_data = {}

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addLayout(self.top_bar_menu())
        h_layout_main = QtWidgets.QHBoxLayout()
        h_layout_main.addWidget(self.database_viewer_layout())
        h_layout_main.addLayout(self.bottom_layout())
        self.layout.addLayout(h_layout_main)
        self.layout.addLayout(self.data_visualize_layout())
        # self.layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.setLayout(self.layout)

        # For troubleshooting only
        self.db_path_file = "output/result.json"
        self.database_reader(self.db_path_file)

    def sorting_db(self):
        id_data = list(self.db_data.keys())
        timestamp = []
        for key in self.db_data.keys():
            timestamp.append(self.db_data[key]["timestamp"])

        new_data = []
        index_sort = np.argsort(timestamp)
        for index in index_sort:
            new_data.append(id_data[index])

        print(new_data)

        self.db_key_viewer = new_data

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(700, 500, QtCore.Qt.KeepAspectRatio)
        return p

    def viewClicked(self, clickedIndex):
        row = clickedIndex.row()
        key_db = self.db_key_viewer[row]
        data = self.db_data[key_db]
        self.update_viewer_violation(data, key_db)

    def database_viewer_layout(self):
        row_size = len(self.db_data)
        column_size = 4

        self.table_column_label = "ID", "Timestamp", "Vehicle", "Violation Type"
        self.table_widget = QtWidgets.QTableWidget(row_size, column_size)
        self.table_widget.clicked.connect(self.viewClicked)
        self.table_widget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        hheader = QtWidgets.QHeaderView(QtCore.Qt.Orientation.Horizontal)
        self.table_widget.setHorizontalHeader(hheader)
        self.table_widget.setHorizontalHeaderLabels(self.table_column_label)
        return self.table_widget

    def database_viewer_update(self):
        row_size = len(self.db_data)
        self.table_widget.setRowCount(row_size)
        self.table_widget.setHorizontalHeaderLabels(self.table_column_label)

        for index in range(0, len(self.db_key_viewer)):
            data_item = self.db_data[self.db_key_viewer[index]]

            vehicle_item = QtWidgets.QTableWidgetItem(data_item["vehicle_type"])
            vehicle_item.setFlags(QtCore.Qt.ItemIsEnabled)

            timestamp_item = QtWidgets.QTableWidgetItem(data_item["timestamp"])
            timestamp_item.setFlags(QtCore.Qt.ItemIsEnabled)

            violation_item = QtWidgets.QTableWidgetItem(data_item["violation_type"])
            violation_item.setFlags(QtCore.Qt.ItemIsEnabled)

            id_item = QtWidgets.QTableWidgetItem(self.db_key_viewer[index])
            id_item.setFlags(QtCore.Qt.ItemIsEnabled)

            data_table_item = [
                id_item,
                timestamp_item,
                vehicle_item,
                violation_item,
            ]

            for index_column, item_column in enumerate(data_table_item):
                self.table_widget.setItem(index, index_column, item_column)

    def data_visualize_layout(self):
        self.button_inference_result = QtWidgets.QPushButton("Inference Video Result")
        self.button_inference_result.clicked.connect(self.set_video_viewer)

        self.button_vehicle = QtWidgets.QPushButton("Vehicle Violation Distribution")
        self.button_vehicle.clicked.connect(self.plot_vehicle_distribution)

        self.button_violation = QtWidgets.QPushButton("Violation Type Distribution")
        self.button_violation.clicked.connect(self.plot_violation_distribution)

        v_layout = QtWidgets.QVBoxLayout()

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(self.button_inference_result)
        h_layout.addWidget(self.button_vehicle)
        h_layout.addWidget(self.button_violation)

        v_layout.addStretch(1)
        v_layout.addLayout(h_layout)
        v_layout.addStretch(1)

        return v_layout

    def set_video_viewer(self):
        video_path_abs = os.path.abspath(self.db_path_file)
        video_path = f"{video_path_abs[:-5]}.avi"
        video_viewer = VideoViewer(video_path)
        video_viewer.exec_()

    def bottom_layout(self):
        violation_group = QtWidgets.QGroupBox("Record Information")

        self.violation_id = QtWidgets.QLabel()
        self.violation_timestamp = QtWidgets.QLabel()
        self.violation_vehicle = QtWidgets.QLabel()
        self.violation_type = QtWidgets.QLabel()

        v1_layout = QtWidgets.QVBoxLayout()
        v1_layout.addWidget(self.violation_id)
        v1_layout.addWidget(self.violation_timestamp)

        v2_layout = QtWidgets.QVBoxLayout()
        v2_layout.addWidget(self.violation_vehicle)
        v2_layout.addWidget(self.violation_type)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addLayout(v1_layout)
        h_layout.addLayout(v2_layout)

        self.image_label = QtWidgets.QLabel()
        self.update_image("samples/file-20200803-24-50u91u.jpg")

        violation_group.setLayout(h_layout)
        main_bottom_layout = QtWidgets.QVBoxLayout()
        main_bottom_layout.addWidget(self.image_label)
        main_bottom_layout.addWidget(violation_group)

        main_bottom_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

        return main_bottom_layout

    def update_viewer_violation(self, data, id):
        timestamp = data["timestamp"]
        vehicle = data["vehicle_type"]
        violation_type = data["violation_type"]

        self.violation_id.setText(f"ID\t\t: {str(id)}")
        self.violation_timestamp.setText(f"Timestamp\t: {timestamp}")
        self.violation_vehicle.setText(f"Vehicle \t\t: {vehicle}")
        self.violation_type.setText(f"Violation Type\t: {violation_type}")
        self.update_image(data["img_proof"])

    def update_image(self, path_img):
        img = cv2.imread(path_img)
        qimage = self.convert_cv_qt(img)
        qpixmap_data = QtGui.QPixmap.fromImage(qimage)
        self.image_label.setPixmap(qpixmap_data)

    def top_bar_menu(self):
        group_db_sort = QtWidgets.QGroupBox("Sort Database")
        self.db_sort_combobox = QtWidgets.QComboBox()
        self.db_sort_combobox.addItem("Vehicle")
        self.db_sort_combobox.addItem("Timestamp")
        self.db_sort_combobox.addItem("Violance Type")
        self.db_sort_apply_button = QtWidgets.QPushButton("Apply")

        h_layout_sort = QtWidgets.QHBoxLayout()
        h_layout_sort.addWidget(self.db_sort_combobox)
        h_layout_sort.addWidget(self.db_sort_apply_button)
        group_db_sort.setLayout(h_layout_sort)

        group_db_data = QtWidgets.QGroupBox("Database Data")
        self.db_path_line = QtWidgets.QLineEdit("")
        self.db_load_button = QtWidgets.QPushButton("Load Database")
        self.db_load_button.clicked.connect(self.load_database)

        h_layout = QtWidgets.QHBoxLayout()
        h_layout.addWidget(self.db_path_line)
        h_layout.addWidget(self.db_load_button)

        # h_layout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        group_db_data.setLayout(h_layout)

        h_combine = QtWidgets.QHBoxLayout()
        h_combine.addWidget(group_db_data)
        # h_combine.addWidget(group_db_sort)

        return h_combine

    @Slot()
    def load_database(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "QFileDialog.getOpenFileName()",
            "",
            "Database File (*.json)",
            options=options,
        )
        if fileName:
            self.db_path_line.setText(fileName)
            self.db_path_file = fileName
            self.database_reader(self.db_path_file)

    def database_reader(self, file_path):
        # Generate Image Analysis
        cmd = f"python tvdr/utils/data_statistics_generator.py --json_path '{os.path.abspath(file_path)}'"
        os.system(cmd)

        with open(file_path, "r", encoding="utf-8") as db_file:
            db_json_data = json.load(db_file)
        self.db_data = {}

        for violation_type in db_json_data.keys():
            violation_data = db_json_data[violation_type]
            for object_id in violation_data.keys():
                print(violation_type)
                id_data = violation_data[object_id]["img_proof"][-14:-4]

                if violation_type == "running_red_light":
                    print("here")
                    id_data = id_data + "R"
                    violation_type_new = "Running Red Light"

                elif violation_type == "wrong_way ":
                    id_data = id_data + "W"
                    violation_type_new = "Wrong Way"

                self.db_data[id_data] = {
                    "vehicle_type": violation_data[object_id]["vehicle_type"],
                    "violation_type": violation_type_new,
                    "img_proof": violation_data[object_id]["img_proof"],
                    "timestamp": violation_data[object_id]["timestamp"],
                }
        self.sorting_db()
        self.database_viewer_update()

    def plot_vehicle_distribution(self):
        dir_path = os.path.dirname(self.db_path_file)
        dialog = QtWidgets.QDialog()
        image = QtWidgets.QLabel()

        img_path = os.path.join(dir_path, "vehicle_distribution_plot.png")
        pixmap = QtGui.QPixmap(img_path)
        x = pixmap.scaled(
            700, 1200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        image.setPixmap(x)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(image)
        dialog.setLayout(layout)

        dialog.exec_()

    def plot_violation_distribution(self):
        print("here")
        dir_path = os.path.dirname(self.db_path_file)
        dialog = QtWidgets.QDialog()
        image = QtWidgets.QLabel()

        img_path = os.path.join(dir_path, "violation_distribution_plot.png")
        pixmap = QtGui.QPixmap(img_path)
        x = pixmap.scaled(
            700, 1200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )

        image.setPixmap(x)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(image)
        dialog.setLayout(layout)

        dialog.exec_()
