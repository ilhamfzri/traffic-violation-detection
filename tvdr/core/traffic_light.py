import numpy as np
import cv2

from tvdr.utils import Parameter


class TrafficLightDetection:
    def __init__(self, parameter: Parameter):
        super().__init__()
        self.update_parameters(parameter)

    def update_parameters(self, parameter: Parameter):
        self.parameter = parameter

        self.red_light = self.parameter.traffic_light_red_light
        self.red_min = np.array(
            [self.red_light["h_min"], self.red_light["s_min"], self.red_light["v_min"]]
        )
        self.red_max = np.array(
            [self.red_light["h_max"], self.red_light["s_max"], self.red_light["v_max"]]
        )

        self.green_light = self.parameter.traffic_light_green_light
        self.green_min = np.array(
            [
                self.green_light["h_min"],
                self.green_light["s_min"],
                self.green_light["v_min"],
            ]
        )

        self.green_max = np.array(
            [
                self.green_light["h_max"],
                self.green_light["s_max"],
                self.green_light["v_max"],
            ]
        )

    def detect_state(self, image: np.ndarray):

        # convert image from rgb to hsv colorspace
        image_csv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # segmentation colors of image based on hsv min and hsv max values
        self.image_red_light = cv2.inRange(image_csv, self.red_min, self.red_max)
        self.image_green_light = cv2.inRange(image_csv, self.green_min, self.green_max)

        # count total pixel each traffic light colors
        self.red_light_count = np.sum(self.image_red_light == 255)
        self.green_light_count = np.sum(self.image_green_light == 255)

        # return traffic light state based on number of pixel threshold
        if self.red_light_count >= self.red_light["threshold"]:
            return "Red"

        elif self.green_light_count >= self.green_light["threshold"]:
            return "Green"

        else:
            return "Undefined"

    def get_red_light_segmentation(self):
        return self.image_red_light

    def get_green_light_segmentation(self):
        return self.image_green_light
