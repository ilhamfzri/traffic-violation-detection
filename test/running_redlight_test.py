from tvdr.core import RunningRedLightConfig, RunningRedLight
import unittest
import cv2


class RunningRedLightTest(unittest.TestCase):
    def setUp(self):
        self.rrl_config = RunningRedLightConfig()
        self.rrl = RunningRedLight(self.rrl_config)
        self.img = cv2.imread("test/assets/traffic_light.jpg")

    def test_red_state(self):
        self.rrl.config.traffic_light_area = [
            [0.45, 0.15],
            [0.6, 0.15],
            [0.6, 0.85],
            [0.45, 0.85],
        ]
        state = self.rrl.detect_state(self.img.copy())
        self.assertEqual(state, "red") 

    def test_green_state(self):
        self.rrl.config.traffic_light_area = [
            [0.75, 0.15],
            [0.9, 0.15],
            [0.9, 0.85],
            [0.75, 0.85],
        ]
        state = self.rrl.detect_state(self.img.copy())
        self.assertEqual(state, "green")

    def test_yellow_state(self):
        self.rrl.config.traffic_light_area = [
            [0.1, 0.15],
            [0.25, 0.15],
            [0.25, 0.85],
            [0.1, 0.85],
        ]
        state = self.rrl.detect_state(self.img.copy())
        self.assertEqual(state, "yellow")

        