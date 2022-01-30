import json
from tvdr.core import yolo
from tvdr.utils import general
from tvdr.utils.params import Parameter


class ConfigLoader(Parameter):
    def __init__(self):
        super().__init__()
        pass

    def load_parser(self, json_path: str):
        with open(json_path, "r") as json_file:
            json_data = json.load(json_file)

            # Read YOLO (Vehicle Detection) Params
            yolo_params = json_data["yolo_vehicle_detection"]
            self.yolo_model_path = yolo_params["model_path"]
            self.yolo_imgsz = yolo_params["imgsz"]
            self.yolo_conf = yolo_params["conf_threshold"]
            self.yolo_iou = yolo_params["iou_threshold"]
            self.yolo_classes = yolo_params["classes"]
            self.yolo_classes_name = yolo_params["classes_name"]
            self.yolo_multi_label = yolo_params["multi_label"]
            self.yolo_max_detection = yolo_params["max_detection"]

            # Read Traffic Light Recognition Params
            traffic_light_params = json_data["traffic_light"]
            self.traffic_light_area = traffic_light_params["area"]
            self.traffic_light_red_light = traffic_light_params["red_light"]
            self.traffic_light_green_light = traffic_light_params["green_light"]

            # Read Tracking Params
            tracking_params = json_data["tracking"]
            self.use_tracking = tracking_params["use_tracking"]

            sort_params = tracking_params["sort"]
            self.sort_min_hits = sort_params["min_hits"]
            self.sort_max_age = sort_params["max_age"]
            self.sort_iou_threshold = sort_params["iou_threshold"]

            deepsort_params = tracking_params["deepsort"]
            self.deepsort_model_path = deepsort_params["model_path"]
            self.deepsort_max_dist = deepsort_params["max_dist"]
            self.deepsort_min_confidence = deepsort_params["min_confidence"]
            self.deepsort_max_iou_distance = deepsort_params["max_iou_distance"]
            self.deepsort_max_age = deepsort_params["max_age"]
            self.deepsort_n_init = deepsort_params["n_init"]
            self.deepsort_nn_budget = deepsort_params["nn_budget"]
            self.deepsort_use_cuda = deepsort_params["use_cuda"]

            # Read Wrong Way Params
            wrongway_params = json_data["wrong_way"]
            self.wrongway_direction_degree = wrongway_params["direction_degree"]
            self.wrongway_threshold_degree = wrongway_params["threshold_degree"]
            self.wrongway_miss_count = wrongway_params["miss_count"]
            self.wrongway_min_value = wrongway_params["min_value_threshold"]

            # Read Helmet Violation Detection Params
            hvd_params = json_data["helmet_violation_detection"]
            self.hv_model_path = hvd_params["model_path"]
            self.hv_imgsz = hvd_params["imgsz"]
            self.hv_conf = hvd_params["conf_threshold"]
            self.hv_iou = hvd_params["iou_threshold"]
            self.hv_min_age = hvd_params["min_age"]
            self.hv_pad_height_mul = hvd_params["padding_height_multiplier"]
            self.hv_pad_width_mul = hvd_params["padding_width_multiplier"]

            # Read General Params
            general_params = json_data["general"]
            self.device = general_params["device"]
            self.video_path = general_params["video_path"]
            self.detection_area = general_params["detection_area"]
            self.stopline = general_params["stopline"]

            self.detect_helmet_violation = general_params["detect_helmet_violation"]
            self.detect_running_redlight_violation = general_params[
                "detect_running_redlight_violation"
            ]
            self.detect_wrongway_violation = general_params["detect_wrongway_violation"]

            self.show_bounding_boxes = general_params["show_bounding_boxes"]
            self.show_label_and_confedence = general_params["show_label_and_confedence"]
            self.show_detection_area = general_params["show_detection_area"]
            self.show_stopline = general_params["show_stopline"]

            return self

    def save_config(self, json_path: str, parameter: Parameter):

        self = parameter
        json_data = {}

        # Set YOLO Params
        yolo_params = {}
        yolo_params["model_path"] = self.yolo_model_path
        yolo_params["imgsz"] = self.yolo_imgsz
        yolo_params["conf_threshold"] = self.yolo_conf
        yolo_params["iou_threshold"] = self.yolo_iou
        yolo_params["classes"] = self.yolo_classes
        yolo_params["classes_name"] = self.yolo_classes_name
        yolo_params["multi_label"] = self.yolo_multi_label
        yolo_params["max_detection"] = self.yolo_max_detection

        # Set Traffic Light Recognition Params
        traffic_light_params = {}
        traffic_light_params["area"] = self.traffic_light_area
        traffic_light_params["red_light"] = self.traffic_light_red_light
        traffic_light_params["green_light"] = self.traffic_light_green_light

        # Set Tracking Params
        hvd_params = {}
        hvd_params["model_path"] = self.hv_model_path
        hvd_params["imgsz"] = self.hv_imgsz
        hvd_params["conf_threshold"] = self.hv_conf
        hvd_params["iou_threshold"] = self.hv_iou
        hvd_params["min_age"] = self.hv_min_age

        ## Set SORT Params
        sort_params = {}
        sort_params["min_hits"] = self.sort_min_hits
        sort_params["max_age"] = self.sort_max_age
        sort_params["iou_threshold"] = self.sort_iou_threshold

        ## Set DeepSORT Params
        deepsort_params = {}
        deepsort_params["model_path"] = self.deepsort_model_path
        deepsort_params["max_dist"] = self.deepsort_max_dist
        deepsort_params["min_confidence"] = self.deepsort_min_confidence
        deepsort_params["max_iou_distance"] = self.deepsort_max_iou_distance
        deepsort_params["max_age"] = self.deepsort_max_age
        deepsort_params["n_init"] = self.deepsort_n_init
        deepsort_params["nn_budget"] = self.deepsort_nn_budget
        deepsort_params["use_cuda"] = self.deepsort_use_cuda

        tracking_params = {}
        tracking_params["use_tracking"] = self.use_tracking
        tracking_params["sort"] = sort_params
        tracking_params["deepsort"] = deepsort_params

        # Wrong Way Params
        wrongway_params = {}
        wrongway_params["direction_degree"] = self.wrongway_direction_degree
        wrongway_params["threshold_degree"] = self.wrongway_threshold_degree
        wrongway_params["miss_count"] = self.wrongway_miss_count
        wrongway_params["min_value_threshold"] = self.wrongway_min_value

        # Helmet Violation Detection  Params
        hvd_params = {}
        hvd_params["model_path"] = self.hv_model_path
        hvd_params["imgsz"] = self.hv_imgsz
        hvd_params["conf_threshold"] = self.hv_conf
        hvd_params["iou_threshold"] = self.hv_iou
        hvd_params["min_age"] = self.hv_min_age
        hvd_params["padding_height_multiplier"] = self.hv_pad_height_mul
        hvd_params["padding_width_multiplier"] = self.hv_pad_width_mul

        # General Params
        general_params = {}
        general_params["device"] = self.device
        general_params["video_path"] = self.video_path
        general_params["detection_area"] = self.detection_area
        general_params["stopline"] = self.stopline

        general_params["detect_helmet_violation"] = self.detect_helmet_violation
        general_params[
            "detect_running_redlight_violation"
        ] = self.detect_running_redlight_violation
        general_params["detect_wrongway_violation"] = self.detect_wrongway_violation

        general_params["show_bounding_boxes"] = self.show_bounding_boxes
        general_params["show_label_and_confedence"] = self.show_label_and_confedence
        general_params["show_detection_area"] = self.show_detection_area
        general_params["show_stopline"] = self.show_stopline

        json_data["general"] = general_params
        json_data["yolo_vehicle_detection"] = yolo_params
        json_data["wrong_way"] = wrongway_params
        json_data["traffic_light"] = traffic_light_params
        json_data["helmet_violation_detection"] = hvd_params
        json_data["tracking"] = tracking_params

        with open(json_path, "w") as json_file:
            json.dump(json_data, json_file, indent=3)
