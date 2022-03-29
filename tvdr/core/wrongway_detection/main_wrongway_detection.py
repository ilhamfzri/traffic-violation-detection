import numpy as np
from .configuration_wrongway_detection import WrongWayDetectionConfig


class WrongWayDetection:
    def __init__(self, config: WrongWayDetectionConfig):
        self.config = config
        self.object_tracker = {}

    def update(self, preds):
        id_show_in_frame = []

        for obj in preds:
            id = int(obj[6])
            centroid = self.calculate_centroid_bbox(bbox=obj[0:4])
            self.update_object(centroid, id)
            id_show_in_frame.append(id)

        self.update_miss_object(id_show_in_frame)
        violate = self.check_wrongway(id_show_in_frame)
        direction_id_show = self.get_object_direction(id_show_in_frame)

        return direction_id_show, violate

    def calculate_centroid_bbox(self, bbox):
        x_min, y_min = bbox[0], bbox[1]
        x_max, y_max = bbox[2], bbox[3]
        x_center = x_min + (x_max - x_min) / 2
        y_center = y_min + (y_max - y_min) / 2
        return (x_center, y_center)

    def update_object(self, centroid, id):
        x_center, y_center = centroid
        if id not in self.object_tracker.keys():
            self.object_tracker[id] = {
                "total_dx": 0,
                "total_dy": 0,
                "last_x": x_center,
                "last_y": y_center,
                "direction": None,
                "miss_count": 0,
            }
        else:
            last_total_dx = self.object_tracker[id]["total_dx"]
            last_total_dy = self.object_tracker[id]["total_dy"]
            last_x = self.object_tracker[id]["last_x"]
            last_y = self.object_tracker[id]["last_y"]

            new_total_dx = last_total_dx + (x_center - last_x)
            new_total_dy = last_total_dy + (y_center - last_y)

            direction = self.calculate_direction(new_total_dx, new_total_dy)

            self.object_tracker[id] = {
                "total_dx": new_total_dx,
                "total_dy": new_total_dy,
                "last_x": x_center,
                "last_y": y_center,
                "direction": direction,
                "miss_count": 0,
            }

    def calculate_direction(self, total_dx, total_dy):
        phi = np.arctan2(total_dy, total_dx)
        degree = np.degrees(phi)
        return degree

    def update_miss_object(self, id_show_in_frame):
        all_id_in_tracker = list(self.object_tracker.keys())
        id_not_show_in_frame = list(set(all_id_in_tracker) - set(id_show_in_frame))
        for id in id_not_show_in_frame:
            miss_count = self.object_tracker[id]["miss_count"] + 1
            self.object_tracker[id] = {"miss_count": miss_count}
            if miss_count >= self.config.removal_miss_count:
                self.object_tracker.pop(id)

    def check_wrongway(self, list_id):
        wrong_way_id = []
        for id in list_id:
            total_dx = self.object_tracker[id]["total_dx"]
            total_dy = self.object_tracker[id]["total_dy"]
            sigma_dy_dx = abs(total_dx + total_dy)
            if sigma_dy_dx >= self.config.min_sigma_dy_dx_violation:
                direction = self.object_tracker[id]["direction"]
                if self.detect_violation(direction):
                    wrong_way_id.append(id)
        return wrong_way_id

    def get_object_direction(self, list_id):
        direction_id = {}
        for id in list_id:
            direction_id[id] = self.object_tracker[id]["direction"]
        return direction_id

    def detect_violation(self, degree):
        if self.config.direction_violation + self.config.direction_violation_thr > 360:
            th1_degrees = self.config.direction_violation_thr - (
                360 - self.config.direction_violation
            )
        else:
            th1_degrees = (
                self.config.direction_violation + self.config.direction_violation_thr
            )

        if self.config.direction_violation - self.config.direction_violation_thr < 0:
            th2_degrees = 360 - (
                self.config.direction_violation_thr - self.config.direction_violation
            )
        else:
            th2_degrees = (
                self.config.direction_violation - self.config.direction_violation_thr
            )

        if th1_degrees < th2_degrees:
            if degree <= th1_degrees or degree >= th2_degrees:
                return True
            else:
                return False
        else:
            if degree <= th1_degrees and degree >= th2_degrees:
                return True
            else:
                return False

    def reset_object_tracker(self):
        self.object_tracker = {}
