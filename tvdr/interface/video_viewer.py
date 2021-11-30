import sys
import cv2
import datetime

from PySide2 import QtWidgets
from PySide2.QtCore import Qt, QUrl
from PySide2.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlayerControl
from PySide2.QtMultimediaWidgets import QVideoWidget
from PySide2.QtWidgets import (
    QApplication,
    QMainWindow,
    QFrame,
    QTabWidget,
    QWidget,
    QHBoxLayout,
    QDialog,
    QVBoxLayout,
)


class VideoViewer(QDialog):
    def __init__(self, video_path):
        super().__init__()

        # video_path = "/Users/hamz/Documents/Kuliah/Semester 7/Skripsi Bismillah/Environment/TrafficViolationDetection/output/result.avi"

        self.setWindowTitle("Violation Video")
        self.resize(1200, 800)
        self.setStyleSheet("Player {background: #000;}")

        # Set Video Player
        videoWidget = QVideoWidget()
        video_url = QUrl.fromLocalFile(video_path)
        video_content = QMediaContent(video_url)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.error.connect(self.handleError)
        self.mediaPlayer.positionChanged.connect(self.update_timestamp_and_slider)
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.setMedia(video_content)

        # Calculate Video Duration
        vid = cv2.VideoCapture(video_path)
        frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = vid.get(cv2.CAP_PROP_FPS)
        duration = int(frame_count / fps)

        # Set Slider and Timestamp
        h_layout_1 = QtWidgets.QHBoxLayout()
        self.video_slider = QtWidgets.QSlider()
        self.video_slider.setOrientation(Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.setMaximum(duration)
        self.video_slider.setSingleStep(1)
        self.video_slider.sliderPressed.connect(self.pause_video)
        self.video_slider.sliderReleased.connect(self.update_slider)
        self.video_timestamp = QtWidgets.QLabel("0:00:00")
        h_layout_1.addWidget(self.video_slider, 9)
        h_layout_1.addWidget(self.video_timestamp, 1)

        # Set Button Control
        h_layout_2 = QtWidgets.QHBoxLayout()
        self.button_start = QtWidgets.QPushButton("Start")
        self.button_start.clicked.connect(self.start_video)

        self.button_pause = QtWidgets.QPushButton("Pause")
        self.button_pause.clicked.connect(self.pause_video)

        self.button_stop = QtWidgets.QPushButton("Stop")
        self.button_stop.clicked.connect(self.stop_video)

        h_layout_1.addWidget(self.button_start)
        h_layout_1.addWidget(self.button_pause)
        h_layout_1.addWidget(self.button_stop)

        h_total = QtWidgets.QHBoxLayout()
        h_total.addLayout(h_layout_2)
        h_total.addLayout(h_layout_1)

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(h_total)

        self.mediaPlayer.play()
        self.setLayout(layout)

    def update_timestamp_and_slider(self, position):
        seconds = int(position / 1000)
        timestamp = datetime.timedelta(seconds=seconds)
        self.video_timestamp.setText(str(timestamp))
        self.video_slider.setValue(seconds)

    def update_slider(self):
        position_slider = self.video_slider.value() * 1000
        self.update_timestamp_and_slider(position_slider)
        self.mediaPlayer.setPosition(position_slider)

    def start_video(self):
        self.mediaPlayer.play()

    def pause_video(self):
        self.mediaPlayer.pause()

    def stop_video(self):
        self.mediaPlayer.stop()

    def handleError(self):
        print("Error: " + self.mediaPlayer.errorString())
