from concurrent.futures import process
import cv2
import time

from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *

from tvdr.core import PipelineConfig, Pipeline
from tvdr.interfacev2 import (
    VehicleDetectionInterface,
    RunningRedLightInterface,
    HelmetViolationInterface,
    WrongWayInterface,
)


class MainLayout(QWidget):
    def __init__(self):
        super().__init__()

        # Pipeline Initialize
        self.pipeline_cfg = PipelineConfig()
        self.pipeline = Pipeline(self.pipeline_cfg)
        self.ready_status = False

        # Layout Initialize
        self.g_layout = QGridLayout()
        self.g_layout.addLayout(self.set_configuration_layout(), 0, 0, 1, 2)
        self.g_layout.addLayout(self.set_image_viewer_layout(), 0, 3, 1, 6)
        self.setLayout(self.g_layout)

    def set_control_layout(self):
        """Control Layout"""

        control_group_box = QGroupBox("Control")
        control_group_box.setAlignment(Qt.AlignCenter)

        self.ready_check_button = QPushButton("Ready Check")
        self.ready_check_button.clicked.connect(self.ready_check)

        self.start_process_button = QPushButton("Start")
        self.start_process_button.clicked.connect(self.start)

        self.stop_process_button = QPushButton("Stop")
        self.stop_process_button.clicked.connect(self.stop)

        h_control_layout = QHBoxLayout()
        h_control_layout.addWidget(self.start_process_button)
        h_control_layout.addWidget(self.stop_process_button)

        v_control_layout = QVBoxLayout()
        v_control_layout.addWidget(self.ready_check_button)
        v_control_layout.addLayout(h_control_layout)

        control_group_box.setLayout(v_control_layout)
        return control_group_box

    def set_configuration_layout(self):
        """Configuration Layout"""

        # Configuration Loader
        config_group_box = QGroupBox("Configuration")
        config_group_box.setAlignment(Qt.AlignCenter)
        self.load_config_button = QPushButton("Load")
        self.load_config_button.clicked.connect(self.load_config)
        self.save_config_button = QPushButton("Save")
        self.save_config_button.clicked.connect(self.save_config)

        h_config_layout = QHBoxLayout()
        h_config_layout.addWidget(self.load_config_button)
        h_config_layout.addWidget(self.save_config_button)
        config_group_box.setLayout(h_config_layout)

        # Video Loader
        video_group_box = QGroupBox("Video")
        video_group_box.setAlignment(Qt.AlignCenter)

        self.video_lineedit = QLineEdit("")
        self.video_load_button = QPushButton("Load")
        self.video_load_button.clicked.connect(self.load_video)

        video_config_layout = QHBoxLayout()
        video_config_layout.addWidget(self.video_lineedit)
        video_config_layout.addWidget(self.video_load_button)
        video_group_box.setLayout(video_config_layout)

        # Vehicle Detection and Tracker Configuration
        vd_group_box = QGroupBox("Vehicle Detection And Tracker")
        vd_group_box.setAlignment(Qt.AlignCenter)
        self.vd_config_button = QPushButton("Configuration")
        self.vd_config_button.clicked.connect(self.load_vd_config)
        self.vd_status = QLabel("Status : Not Ready")

        vd_config_layout = QVBoxLayout()
        vd_config_layout.addWidget(self.vd_config_button)
        vd_config_layout.addWidget(self.vd_status)
        vd_group_box.setLayout(vd_config_layout)

        # Running Red Light Configuration
        rrld_group_box = QGroupBox("Running Red Light")
        rrld_group_box.setAlignment(Qt.AlignCenter)
        self.rrld_config_button = QPushButton("Configuration")
        self.rrld_config_button.clicked.connect(self.load_rrld_config)
        self.rrld_status = QLabel("Status : Not Ready")

        rrld_config_layout = QVBoxLayout()
        rrld_config_layout.addWidget(self.rrld_config_button)
        rrld_config_layout.addWidget(self.rrld_status)
        rrld_group_box.setLayout(rrld_config_layout)

        # Helmet Violation Configuration
        hv_group_box = QGroupBox("Helmet Violation")
        hv_group_box.setAlignment(Qt.AlignCenter)
        self.hv_config_button = QPushButton("Configuration")
        self.hv_config_button.clicked.connect(self.load_hv_config)
        self.hv_status = QLabel("Status : Not Ready")

        hv_config_layout = QVBoxLayout()
        hv_config_layout.addWidget(self.hv_config_button)
        hv_config_layout.addWidget(self.hv_status)
        hv_group_box.setLayout(hv_config_layout)

        # Wrong-way Detection
        ww_group_box = QGroupBox("Wrong-way Detection")
        ww_group_box.setAlignment(Qt.AlignCenter)
        self.ww_config_button = QPushButton("Configuration")
        self.ww_config_button.clicked.connect(self.load_ww_config)
        self.ww_status = QLabel("Status : Not Ready")

        ww_config_layout = QVBoxLayout()
        ww_config_layout.addWidget(self.ww_config_button)
        ww_config_layout.addWidget(self.ww_status)
        ww_group_box.setLayout(ww_config_layout)

        v_layout_config = QVBoxLayout()
        v_layout_config.addWidget(config_group_box)
        v_layout_config.addWidget(video_group_box)
        v_layout_config.addWidget(vd_group_box)
        v_layout_config.addWidget(rrld_group_box)
        v_layout_config.addWidget(hv_group_box)
        v_layout_config.addWidget(ww_group_box)
        v_layout_config.addWidget(self.set_control_layout())
        v_layout_config.addStretch(1)

        return v_layout_config

    def load_config(self):
        """Load Configuration From File"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Config File",
            "",
            "Config File (*.json)",
            options=options,
        )

        if fileName:
            self.pipeline_cfg.load_config(fileName)
            self.video_lineedit.setText(self.pipeline_cfg.video_path)
            self.vid = cv2.VideoCapture(self.pipeline_cfg.video_path)
            ret, frame = self.vid.read()
            image = self.convert_cv_qt(frame)
            self.image_frame.setPixmap(QPixmap.fromImage(image))
            self.frame_total = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)

    def save_config(self):
        """Save Configuration To File"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "Save Config File",
            "",
            "Config File (*.json)",
            options=options,
        )

        if fileName:
            if fileName.endswith(".json") == False:
                fileName = f"{fileName}.json"
            self.pipeline_cfg.save_config(fileName)
            self.video_lineedit.setText(fileName)

    def load_video(self):
        """Load Video From File"""
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video File (*.mp4)",
            options=options,
        )

        if fileName:
            self.pipeline_cfg.video_path = fileName
            self.video_lineedit.setText(fileName)
            self.vid = cv2.VideoCapture(self.pipeline_cfg.video_path)
            ret, frame = self.vid.read()
            image = self.convert_cv_qt(frame)
            self.image_frame.setPixmap(QPixmap.fromImage(image))
            self.frame_total = self.vid.get(cv2.CAP_PROP_FRAME_COUNT)


    
    def ready_check(self):
        """Ready Check Before Start"""
        self.pipeline.config = self.pipeline_cfg
        self.pipeline.reload_config()
        self.ready_status = self.pipeline.ready_check() 

        if self.pipeline.vd_ready_status == True:
            self.vd_status.setText("Status : Ready")
        else:
            self.vd_status.setText("Status : Not Ready")

        if self.pipeline.rrl_ready_status == True:
            self.rrld_status.setText("Status : Ready")
        else:
            self.rrld_status.setText("Status : Not Ready")

        if self.pipeline.hv_ready_status == True:
            self.hv_status.setText("Status : Ready")
        else:
            self.hv_status.setText("Status : Not Ready")

        if self.pipeline.ww_ready_status == True:
            self.ww_status.setText("Status : Ready")
        else:
            self.ww_status.setText("Status : Not Ready")


    def start(self):
        """Start Processing Video"""
        if self.ready_status == False:
            msg = QMessageBox()
            msg.setText("Make sure everything is ready!")
            msg.exec_()
            return 
            
        self.vid = cv2.VideoCapture(self.pipeline_cfg.video_path)
        self.pipeline.video_recorder_init(self.pipeline_cfg.video_path)
    
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000.0 / 30)
        self.disable_button()


    def stop(self):
        """Stop Processing Video"""
        self.timer.stop()
        self.enable_button()
        video_output_path = self.pipeline.video_recorder_close()
        msg = QMessageBox()
        msg.setText(f"Video saved in this path :\n {video_output_path}")
        msg.exec_()
        

    def update_frame(self):
        start_time = time.time()
        ret, frame = self.vid.read()
        result_frame = self.pipeline.update(frame)
        self.pipeline.video_recorder_update(result_frame)
        image = self.convert_cv_qt(result_frame)
        self.image_frame.setPixmap(QPixmap.fromImage(image))
        processing_time = time.time() - start_time
        frame_idx = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
        progress_value = (frame_idx / self.frame_total) * 100
        
        fps = 1/processing_time
        self.fps_label.setText(f"FPS : {fps}")
        self.progress_bar.setValue(progress_value)

        if frame_idx == self.frame_total- 1:
            self.timer.stop()
            self.stop()
    
    def load_vd_config(self):
        """Load Vehicle Detection & Tracker Config Window"""
        self.vdi = VehicleDetectionInterface(
            config=self.pipeline_cfg.vd_config, video_path=self.pipeline_cfg.video_path
        )
        self.vdi.exec_()
        if self.vdi.result() == 1:
            self.pipeline_cfg.vd_config = self.vdi.config
            self.vd_status.setText("Status : Not Ready")
            self.ready_status = False

    def load_rrld_config(self):
        """Load Running Red Light Config Window"""
        self.rrli = RunningRedLightInterface(
            config=self.pipeline_cfg.rrl_config,
            config_vd=self.pipeline_cfg.vd_config,
            video_path=self.pipeline_cfg.video_path,
        )
        self.rrli.exec_()

        if self.rrli.result() == 1:
            self.pipeline_cfg.rrl_config = self.rrli.config
            self.rrld_status.setText("Status : Not Ready")
            self.ready_status = False

    def load_hv_config(self):
        """Load Helmet Violation Config Window"""
        self.hvi = HelmetViolationInterface(
            config=self.pipeline_cfg.hv_config,
            config_vd=self.pipeline_cfg.vd_config,
            video_path=self.pipeline_cfg.video_path,
        )
        self.hvi.exec_()

        if self.hvi.result() == 1:
            self.pipeline_cfg.hv_config = self.hvi.config
            self.hv_status.setText("Status : Not Ready")
            self.ready_status = False

    def load_ww_config(self):
        """Load Wrong-way Detection Config Window"""
        self.wwi = WrongWayInterface(
            config=self.pipeline_cfg.ww_config,
            vd_config=self.pipeline_cfg.vd_config,
            video_path=self.pipeline_cfg.video_path,
        )
        self.wwi.exec_()

        if self.wwi.result() == 1:
            self.pipeline_cfg.ww_config = self.wwi.config
            self.ww_status.setText("Satus : Not Ready")
            self.ready_status = False

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(
            rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        p = convert_to_Qt_format.scaled(10000, 600, Qt.KeepAspectRatio)
        return p

    def set_image_viewer_layout(self):
        # Frame Viewer
        self.image_frame = QLabel()
        self.image = cv2.imread("img/tvdr.png")
        self.image = self.convert_cv_qt(self.image)
        self.image_frame.setPixmap(QPixmap.fromImage(self.image))

        # Progress Bar
        self.progress_bar = QProgressBar()

        # Additional Information
        information_groupbox = QGroupBox("Information")
        self.video_duration_label = QLabel("Video Duration : ")
        self.fps_label = QLabel("FPS : ")

        v_information_layout = QVBoxLayout()
        v_information_layout.addWidget(self.video_duration_label)
        v_information_layout.addWidget(self.fps_label)
        information_groupbox.setLayout(v_information_layout)

        v_image_viewer_layout = QVBoxLayout()
        v_image_viewer_layout.addWidget(self.image_frame)
        v_image_viewer_layout.addWidget(self.progress_bar)
        v_image_viewer_layout.addWidget(information_groupbox)

        return v_image_viewer_layout

    def set_image_layout(self):
        return self.image_frame

    def disable_button(self):
        self.ready_check_button.setDisabled(True)
        self.start_process_button.setDisabled(True)
        self.stop_process_button.setDisabled(False)
        self.load_config_button.setDisabled(True)
        self.save_config_button.setDisabled(True)
        self.video_load_button.setDisabled(True)
        self.vd_config_button.setDisabled(True)
        self.rrld_config_button.setDisabled(True)
        self.hv_config_button.setDisabled(True)
        self.ww_config_button.setDisabled(True)

    def enable_button(self):
        self.ready_check_button.setDisabled(False)
        self.start_process_button.setDisabled(False)
        self.stop_process_button.setDisabled(True)
        self.load_config_button.setDisabled(False)
        self.save_config_button.setDisabled(False)
        self.video_load_button.setDisabled(False)
        self.vd_config_button.setDisabled(False)
        self.rrld_config_button.setDisabled(False)
        self.hv_config_button.setDisabled(False)
        self.ww_config_button.setDisabled(False)


