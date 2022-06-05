import cv2
import numpy as np


class Annotator:
    def __init__(self, config):
        self.config = config
        self.vd_config = config.vd_config
        self.rrl_config = config.rrl_config

        self.color_bbox = (0, 240, 0)
        self.text_font = cv2.FONT_HERSHEY_DUPLEX
        self.text_color = (0, 0, 0)
        self.text_thickness = 1
        self.text_fontscale = 0.5

        self.color_detection_area = (0, 255, 255)
        self.line_thickness_detection_area = 2

        self.rrl_traffic_light_thickness = 2
        self.rrl_bbox_color = (0, 0, 255)
        self.rrl_stopline_color = (0, 0, 255)
        self.rrl_stopline_thickness = 2

        self.hv_text_fontscale = 1
        self.hv_text_color = (0, 0, 0)
        self.hv_text_thickness = 1
        self.hv_text_b_color = (0, 0, 255)
        self.hv_text_font = cv2.FONT_HERSHEY_DUPLEX

    def vehicle_detection(self, img, preds):
        img_new = img.copy()

        # draw detection area
        img_new = self.draw_detection_area(img_new, self.vd_config.detection_area)

        # create bounding boxes
        for obj in preds:
            # add bounding box
            img_new = self.draw_bounding_box(
                img_new, bbox=obj[0:4], color=self.color_bbox
            )
            # add text
            img_new = self.put_text_info(img_new, obj)

        return img_new

    def running_red_light(self, img, preds, violation_list, state=None):
        # create violation vehicle bbox
        img_new = img.copy()
        for obj in preds:
            if obj[6] in violation_list:
                # add bounding box
                img_new = self.draw_bounding_box(
                    img_new, bbox=obj[0:4], color=self.rrl_bbox_color
                )

        # draw stopline
        img_new = self.draw_stop_line(img_new, self.rrl_config.stop_line)

        # draw traffic light
        img_new = self.draw_traffic_light(
            img_new, self.rrl_config.traffic_light_area, state
        )
        return img_new

    def helmet_violation(self, img, preds, violation_list):
        img_new = img.copy()
        for obj in preds:
            if obj[6] in violation_list:
                # add helmet violation box
                img_new = self.put_no_helmet(img_new, obj)

        return img_new

    def wrongway_detection(self, img, preds, direction_data, violation):
        img_new = img.copy()

        for obj in preds:
            x_min, y_min, x_max, y_max = obj[0:4]
            object_id = obj[6]

            width = abs(x_max - x_min)
            height = abs(y_max - y_min)

            x_center = x_min + width / 2
            y_center = y_min + height / 2

            if direction_data[object_id] == None:
                continue

            degree_direction = int(direction_data[object_id])
            rad_direction = np.radians(degree_direction)

            x = (0.2 * np.cos(rad_direction)) * width
            y = (0.2 * np.sin(rad_direction)) * height

            pos1 = (int(x_center - x), int(y_center + y))
            pos2 = (int(x_center), int(y_center))

            color_arrow = (0, 0, 255) if object_id in violation else (0, 255, 0)

            img_new = cv2.arrowedLine(
                img_new,
                pos1,
                pos2,
                color_arrow,
                3,
                cv2.LINE_AA,
                tipLength=0.5,
            )

        return img_new

    def put_no_helmet(self, img0, obj):
        bbox = obj[0:4]

        # add text
        text_bbox = "No Helmet"
        text_size, _ = cv2.getTextSize(
            text_bbox, self.hv_text_font, self.hv_text_fontscale, self.hv_text_thickness
        )
        text_w, text_h = text_size

        img_new = img0.copy()
        img_new = cv2.rectangle(
            img_new,
            pt1=(int(bbox[0]), int(bbox[3])),
            pt2=(int(bbox[0] + text_w), int(bbox[3] + text_h)),
            color=self.hv_text_b_color,
            thickness=cv2.FILLED,
        )

        img_new = cv2.putText(
            img_new,
            text_bbox,
            org=(int(bbox[0]), int(bbox[3] + text_h)),
            fontFace=self.hv_text_font,
            fontScale=self.hv_text_fontscale,
            color=self.hv_text_color,
            thickness=self.hv_text_thickness,
            lineType=cv2.LINE_AA,
        )

        return img_new

    def put_text_info(self, img0, obj):
        bbox = obj[0:4]
        label_idx = obj[5]
        object_id = obj[6]

        # add text
        text_bbox = f"{self.vd_config.classes_names[int(label_idx)]} ({object_id})"
        text_size, _ = cv2.getTextSize(
            text_bbox, self.text_font, self.text_fontscale, self.text_thickness
        )
        text_w, text_h = text_size

        # create background color for text
        img_new = img0.copy()
        img_new = cv2.rectangle(
            img_new,
            pt1=(int(bbox[0]), int(bbox[1])),
            pt2=(int(bbox[0] + text_w), int(bbox[1] - text_h)),
            color=self.color_bbox,
            thickness=cv2.FILLED,
        )
        img_new = cv2.putText(
            img_new,
            text_bbox,
            org=(int(bbox[0]), int(bbox[1])),
            fontFace=self.text_font,
            fontScale=self.text_fontscale,
            color=self.text_color,
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )
        return img_new

    def draw_bounding_box(self, img0, bbox, color, thickness=2, r=1, d=0.4):

        bbox = [int(p) for p in bbox]
        x1, y1 = bbox[0], bbox[1]
        x2, y2 = bbox[2], bbox[3]
        img = img0.copy()

        width = abs(x2 - x1)
        height = abs(y2 - y1)

        d_x = int(width * d / 2)
        d_y = int(height * d / 2)

        d = 1

        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d_x, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d_y), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d_x, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d_y), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d_x, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d_y), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d_x, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d_y), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        return img

    def draw_detection_area(self, img0, detection_area):
        height, width, _ = img0.shape
        detection_area = [[int(x * width), int(y * height)] for x, y in detection_area]
        detection_area = np.array(detection_area)

        img = cv2.drawContours(
            img0,
            [detection_area],
            -1,
            self.color_detection_area,
            self.line_thickness_detection_area,
        )

        return img

    def draw_stop_line(self, img0, stop_line):
        height, width, _ = img0.shape
        stop_line = [[int(x * width), int(y * height)] for x, y in stop_line]
        stop_line = np.array(stop_line)

        img = cv2.drawContours(
            img0,
            [stop_line],
            -1,
            self.rrl_stopline_color,
            self.rrl_stopline_thickness,
        )

        return img

    def draw_traffic_light(self, img0, traffic_light_area, state):
        color_state = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
        }

        height, width, _ = img0.shape
        traffic_light_area = [
            [int(x * width), int(y * height)] for x, y in traffic_light_area
        ]
        traffic_light_area = np.array(traffic_light_area)

        img = cv2.drawContours(
            img0,
            [traffic_light_area],
            -1,
            color_state[state] if state in color_state.keys() else (0, 0, 0),
            self.rrl_traffic_light_thickness,
        )
        return img
