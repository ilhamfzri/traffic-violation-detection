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
