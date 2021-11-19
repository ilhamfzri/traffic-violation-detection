from tvdr.core.algorithm import calculate_center_of_box
from tvdr.core.algorithm import cart2pol
from tvdr.utils.params import Parameter
import numpy as np
import cv2


class WrongWayDetection:
    def __init__(self, parameter: Parameter):
        self.update_params(parameter)

    def update_params(self, parameter: Parameter):
        # Get Video FPS
        vid = cv2.VideoCapture(parameter.video_path)
        self.fps = vid.get(cv2.CAP_PROP_FPS)

        # Set Wrong Way Params
        self.miss_count = parameter.wrongway_miss_count * self.fps
        self.min_value = parameter.wrongway_min_value
        self.direction_violation = parameter.wrongway_direction_degree
        self.direction_threshold = parameter.wrongway_threshold_degree
        self.data_dict = {}

    def update(self, result):
        id_show = []
        for data in result:
            id = data[6]
            pos = calculate_center_of_box(data[0:4])

            # Update Direction For Each Object
            if id not in self.data_dict.keys():
                id_dict = {}
                id_dict["age"] = 1
                id_dict["total_gx"] = 0
                id_dict["total_gy"] = 0
                id_dict["last_x"] = pos[0]
                id_dict["last_y"] = pos[1]
                id_dict["show_count"] = 1
                id_dict["miss_count_repeat"] = 0
                id_dict["direction"] = None
                self.data_dict[id] = id_dict

            else:
                self.data_dict[id]["age"] += 1
                self.data_dict[id]["total_gx"] += pos[0] - self.data_dict[id]["last_x"]
                self.data_dict[id]["total_gy"] += pos[1] - self.data_dict[id]["last_y"]
                self.data_dict[id]["last_x"] = pos[0]
                self.data_dict[id]["last_y"] = pos[1]
                self.data_dict[id]["show_count"] += 1
                self.data_dict[id]["miss_count_repeat"] = 0
                direction = np.degrees(
                    cart2pol(
                        self.data_dict[id]["total_gx"],
                        -self.data_dict[id]["total_gy"],
                    )[1]
                )
                if direction < 0:
                    direction = 360 + direction

                self.data_dict[id]["direction"] = direction
            id_show.append(id)

        all_id = list(self.data_dict.keys())
        id_not_shown = list(set(all_id) - set(id_show))

        for id in id_not_shown:
            id = int(id)
            self.data_dict[id]["age"] += 1
            self.data_dict[id]["miss_count_repeat"] += 1

        # For Troubleshoot
        # print(f"Key : Direction | Total dXY | Miss")
        # for key in self.data_dict.keys():
        #     total = abs(self.data_dict[key]["total_gx"]) + abs(
        #         self.data_dict[key]["total_gy"]
        #     )
        #     direction = self.data_dict[key]["direction"]
        #     miss_count = self.data_dict[key]["miss_count_repeat"]
        #     print(f"{key} : {direction} | {total} | {miss_count}")
        #     print("\n")
        # print(
        #     f"Wrong Way Direction : {self.direction_violation}   |  Threshold Direction : {self.direction_threshold}"
        # )

        self.remove_object_miss()
        self.wrong_way_list = self.check_wrong_way()
        print(self.wrong_way_list)

    def get_wrong_way_list(self):
        return self.wrong_way_list

    def check_wrong_way(self):
        wrong_way_id = []
        for key in self.data_dict.keys():
            total = abs(self.data_dict[key]["total_gx"]) + abs(
                self.data_dict[key]["total_gy"]
            )
            if total >= self.min_value:
                object_direction = self.data_dict[key]["direction"]
                if self.check_degrees(object_direction):
                    wrong_way_id.append(key)

        return wrong_way_id

    def remove_object_miss(self):
        print(self.miss_count)
        for key in self.data_dict.copy():
            if self.data_dict[key]["miss_count_repeat"] >= self.miss_count:
                self.data_dict.pop(key)

    def check_degrees(self, degree):
        if self.direction_violation + self.direction_threshold > 360:
            th1_degrees = self.direction_threshold - (360 - self.direction_violation)
        else:
            th1_degrees = self.direction_violation + self.direction_threshold

        if self.direction_violation - self.direction_threshold < 0:
            th2_degrees = 360 - (self.direction_threshold - self.direction_violation)
        else:
            th2_degrees = self.direction_violation - self.direction_threshold

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
