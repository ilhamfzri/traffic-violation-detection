class Parameter:
    def __init__(self):
        self.yolo_model_dict = {
            "YOLOV5 Small": "yolov5s",
            "YOLOV5 Medium": "yolov5m",
        }
        self.video_path = ""
        self.yolo_conf = 0.25
        self.yolo_iou = 0.45
        self.yolo_classes = [2, 3, 5, 7]
        self.yolo_multi_label = False
        self.yolo_max_detection = 500

        # 0 = all area, 1 = traffic light area
        self.traffic_light_set_view = 0

        # 0 = RGB, 1 = Segmentation
        self.traffic_light_post_processing = 0

        # traffic ligth area : [x_begin, y_begin, x_end, y_end]
        self.traffic_light_area = [0, 20, 100, 300]

        self.traffic_light_red_light = {
            "h_min": 150,
            "h_max": 180,
            "s_min": 70,
            "s_max": 255,
            "v_min": 50,
            "v_max": 255,
            "threshold": 14,
        }

        self.traffic_light_green_light = {
            "h_min": 36,
            "h_max": 70,
            "s_min": 25,
            "s_max": 255,
            "v_min": 25,
            "v_max": 255,
            "threshold": 13,
        }
