import torch
import numpy as np
import torch.nn.functional as F

from .configuration_helmet_violation import HelmetViolationConfig


class HelmetViolation:
    def __init__(self, config: HelmetViolationConfig):
        self.config = config
        self.missing_removal_thres = 150
        self.id_tracker = {}
        self.load_model()

    def load_model(self) -> bool:
        self.model = torch.load(
            self.config.model_path, map_location=torch.device(self.config.device)
        )["model"].float()
        return True

    def update(self, img, preds):
        filter_vehicle = self.motorcycle_and_bicycle_filtering(preds)
        self.tracker_record_update(filter_vehicle)
        list_object_inference = self.get_object_inference(filter_vehicle)
        violation_result = self.detect_violation(img, list_object_inference)
        return violation_result

    def motorcycle_and_bicycle_filtering(self, preds):
        # Get data from vehicle detection and return only 2 wheel vehicle (motorcycle and bicycle)
        result_filter = np.empty((0, preds.shape[1]))
        for obj in preds:
            vehicle_idx = obj[5]
            if vehicle_idx in self.config.vehicle_idx:
                result_filter = np.append(
                    result_filter, obj.reshape(1, preds.shape[1]), axis=0
                )
        return result_filter

    def tracker_record_update(self, result_filter):
        list_id = []
        for object in result_filter:
            id = object[6]
            if id not in self.id_tracker.keys():
                self.id_tracker[id] = {
                    "age": 1,
                    "missing": 0,
                }
            if id in self.id_tracker.keys():
                self.id_tracker[id]["age"] += 1
                self.id_tracker[id]["missing"] = 0
            list_id.append(id)

        not_in_list_id = list(set(list(self.id_tracker.keys())) - set(list_id))
        for id in not_in_list_id:
            self.id_tracker[id]["age"] += 1
            self.id_tracker[id]["missing"] += 1

        list_delete_id = []
        for id in self.id_tracker.keys():
            if self.id_tracker[id]["missing"] >= self.missing_removal_thres:
                list_delete_id.append(id)

        for id_delete in list_delete_id:
            del self.id_tracker[id_delete]

    def get_object_inference(self, filter_preds):
        object_inference = np.empty((0, filter_preds.shape[1]))
        for object in filter_preds:
            id = object[6]
            if (
                self.id_tracker[id]["age"] > self.config.min_age
                and self.id_tracker[id]["age"] % self.config.detect_interval == 0
            ):
                object_inference = np.append(
                    object_inference, object.reshape(1, 7), axis=0
                )
        return object_inference

    def cropping_img(self, img, bbox):
        # croping image to get motorcycle and bicycle frame
        x, y = int(bbox[0]), int(bbox[1])
        w, h = int(abs(bbox[2] - bbox[0])), int(abs(bbox[3] - bbox[1]))
        img_crop = img[y : y + h, x : x + w, :]
        return img_crop

    def detect_violation(self, img, list_object):
        violation_result = []

        for obj in list_object:
            bbox = obj[0:4]
            img_crop = self.cropping_img(img, bbox)
            predict_result = self.predict(img_crop)

            if predict_result == True:
                object_id = obj[6]
                violation_result.append(object_id)

        return violation_result

    def predict(self, img):
        resize = torch.nn.Upsample(
            size=(self.config.imgsz, self.config.imgsz),
            mode="bilinear",
            align_corners=False,
        )
        normalize = lambda x, mean=0.5, std=0.25: (x - mean) / std

        im = np.ascontiguousarray(np.asarray(img).transpose((2, 0, 1)))
        im = torch.tensor(im).float().unsqueeze(0) / 255.0
        im = resize(normalize(im))

        print(im.shape)

        with torch.no_grad():
            results = self.model(im)
            p = F.softmax(results, dim=1)  # probabilities
            i = p.argmax()  # max index

        if int(i) == self.config.violation_idx and p[0, i] >= self.config.conf_thres:
            return True

        return False
