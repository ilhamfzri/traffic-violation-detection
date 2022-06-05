import torch
import cv2
import numpy as np
import logging

from tvdr.utils.general import (
    check_img_size,
    letterbox,
    non_max_suppression,
    scale_coords,
    combine_yolo_sort_result,
)
from .configuration_vehicle_detection import VehicleDetectionConfig
from ..sort import Sort

class VehicleDetection:
    def __init__(self, config: VehicleDetectionConfig):
        self.config = config
        self.load_model()
        self.load_tracker()

    def load_model(self):
        try:
            # Load vehicle detection model (yolo5 architecture)
            self.model = torch.hub.load(
                "ultralytics/yolov5", "custom", path=self.config.model_path
            )
            # Calculate best imgsz based on model stride
            self.imgsz = check_img_size(self.config.imgsz, self.model.stride)

            # Warmup model if device is 'gpu'
            if self.config.device == "gpu":
                dummy_im = torch.zeros((1, 3, self.imgsz, self.imgsz)).to(
                    self.config.device
                )
                self.model.forward(dummy_im)
            return True

        except Exception as e:
            logging.info(e)
            return False

    def load_tracker(self):
        # Load tracker
        # Currently only support forSORT, maybe in the future i will add DeepSORT or another good tracker method
        try:
            if self.config.tracker == "SORT":
                self.tracker = Sort(
                    self.config.sort_max_age,
                    self.config.sort_min_hits,
                    self.config.sort_iou_thres,
                )
            return True
        except:
            return False

    def update(self, img):
        # Detect vehicle object
        preds = self.predict(img)

        # Track vehicle object
        preds = self.track(preds)

        # Post-processing (Remove object outside detection area)
        img_size = img.shape
        preds = self.post_processing(preds, img_size)

        return preds

    def reset_tracker(self):
        if self.config.tracker == "SORT":
            self.tracker.reset_count()

    def predict(self, img) -> np.ndarray:
        # Padding image
        im = letterbox(
            img,
            new_shape=self.imgsz,
            stride=self.model.stride,
            auto=True,
            scaleFill=True,
        )[0]

        # Convert cv2 colorspace from bgr to rgb
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.config.device).float()

        # Normalize
        im = im / 255

        # if inference is single image size then add dimension
        if len(im.shape) == 3:
            im = im[None]

        # Inference img
        preds = self.model(im)

        # Non Max Suppression
        preds = non_max_suppression(
            prediction=preds,
            conf_thres=self.config.conf_thres,
            iou_thres=self.config.iou_thres,
            max_det=self.config.max_detection,
        )

        # Rescale result to original resolution
        preds = preds[0]
        preds[:, :4] = scale_coords(im.shape[2:], preds[:, :4], img.shape).round()

        # Convert to tensor cpu if tensor cuda
        if preds.is_cuda:
            preds = preds.cpu()

        return preds.numpy()

    def track(self, preds: np.ndarray):
        if self.config.tracker == "SORT":
            bbox = preds[:, 0:5]
            sort_result = self.tracker.update(bbox)
            tracker_result = combine_yolo_sort_result(preds, sort_result)

            return tracker_result

    def post_processing(self, preds, img_shape):
        # remove object outside detection area
        # https://stackoverflow.com/questions/217578/how-can-i-determine-whether-a-2d-point-is-within-a-polygon

        if len(self.config.detection_area) < 3:
            raise ValueError(
                f"Num of points in detection area below 3, current num of points is {len(self.config.detection_area)}"
            )

        result_post = np.empty((0, preds.shape[1]))
        for object in preds:

            # calculate centroid for object
            x_center = object[0] + abs(object[2] - object[0]) / 2
            y_center = object[1] + abs(object[3] - object[1]) / 2

            # Rescale x_center, y_center to range (0.0 - 1.0)
            x_center = x_center / img_shape[1]
            y_center = y_center / img_shape[0]

            # check if centroid inside polygon points of the detection area
            np_detection_area = np.array(self.config.detection_area)

            x_min_da = np.min(np_detection_area[:, 0])
            y_min_da = np.min(np_detection_area[:, 1])

            x_max_da = np.max(np_detection_area[:, 0])
            y_max_da = np.max(np_detection_area[:, 1])

            if (
                x_center < x_min_da
                or x_center > x_max_da
                or y_center < y_min_da
                or y_center > y_max_da
            ):
                continue

            inside = False

            j = np_detection_area.shape[0] - 1

            for i in range(0, np_detection_area.shape[0]):
                xi_da, yi_da = np_detection_area[i]
                xj_da, yj_da = np_detection_area[j]

                state_1 = (yi_da > y_center) != (yj_da > y_center)

                divisor = yj_da - yi_da

                if divisor == 0:
                    divisor = 0.00000001

                state_2 = x_center < (
                    (xj_da - xi_da) * (y_center - yi_da) / divisor + +xi_da
                )

                if state_1 and state_2:
                    inside = not inside

                j = i

            if inside:
                result_post = np.append(
                    result_post, object.reshape(1, preds.shape[1]), axis=0
                )

        return result_post

    def ready_check(self) -> bool:
        """Ready Check"""
        if self.config.model_path == "":
            return False

        if len(self.config.detection_area) < 3:
            return False

        # Force Reload Model and Tracker
        if self.load_model() == False:
            logging.info("Failed to initialize VD Model")
            return False

        if self.load_tracker() == False:
            return False

        return True
