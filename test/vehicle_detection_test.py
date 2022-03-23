import unittest
import os
import numpy as np
import cv2

from tvdr.core import VehicleDetection, VehicleDetectionConfig


class VehicleDetectionTest(unittest.TestCase):
    def setUp(self):
        # Initialize Vehicle Detection Config
        self.vd_config = VehicleDetectionConfig()
        self.vd_config.model_path = "models/vehicle_detection/best.pt"

        # Initialize Vehicle Detection
        self.vd = VehicleDetection(self.vd_config)

        self.img_path = "test/assets/img_1.jpg"

    def test_load_model(self):
        # Load model test
        state = self.vd.load_model()
        self.assertTrue(state)

    def test_predict(self):
        # Testing prediction with sample image
        img = cv2.imread(self.img_path)
        preds = self.vd.predict(img)
        self.preds = preds

        self.assertIsInstance(preds, np.ndarray)
        self.assertGreater(preds.shape[0], 0)
        self.assertEqual(preds.shape[1], 6)

    def test_tracker(self):
        img = cv2.imread(self.img_path)
        preds = self.vd.predict(img)
        preds = self.vd.track(preds)

        self.assertIsInstance(preds, np.ndarray)
        self.assertGreater(preds.shape[0], 0)
        self.assertEqual(preds.shape[1], 7)

    def test_postprocessing(self):
        img = cv2.imread(self.img_path)
        preds = self.vd.predict(img)
        preds = self.vd.track(preds)
        preds = self.vd.post_processing(preds, img.shape)

        self.assertIsInstance(preds, np.ndarray)
        self.assertGreater(preds.shape[0], 0)
