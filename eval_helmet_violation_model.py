#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 UGM

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Ilham Fazri - ilhamfazri3rd@gmail.com

"""
Evaluate a Helmet Violation Model
"""


MODEL_PATH = ""

from tqdm import tqdm

import torchvision
import torchvision.transforms as T
import torch
import sys
import torch.nn.functional as F

from torchmetrics import Precision, Recall, Accuracy, MetricCollection

sys.path.append("yolov5_repo")

VAL_DIR = "dataset/val/val"
BATCH_SIZE = 1
NUM_WORKERS = 0
MODEL_PATH = "models/helmet_detection/efficientnet_b0_combine_dataset.pt"
IMGSZ = 224


resize = torch.nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)


def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    # Transforms
    valform = T.Compose(
        [
            T.Resize([IMGSZ, IMGSZ]),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
        ]
    )

    # Dataloaders
    valset = torchvision.datasets.ImageFolder(VAL_DIR, transform=valform)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    names = valset.classes
    nc = len(valset.classes)

    # Load Model
    model = torch.load(MODEL_PATH, map_location=torch.device(device))["model"].float()

    # Initialize
    label_result = {}
    for label in names:
        label_result[label] = {"TP": 0, "FP": 0, "FN": 0}

    acc = Accuracy(num_classes=2)
    pred = []
    targets = []

    with torch.no_grad():
        for _, (image, label) in enumerate(tqdm(valloader)):
            image, label = image.to(device), label.to(device)

            # Inference
            preds = model(resize(image))
            p = F.softmax(preds, dim=1)  # convert to probabilty
            pred_label = p.argmax().unsqueeze(0)  # get index

            label_name = names[label.squeeze(0)]

            pred_label_name = names[pred_label.squeeze(0)]
            if label == pred_label:
                label_result[label_name]["TP"] += 1

            else:
                label_result[label_name]["FN"] += 1
                label_result[pred_label_name]["FP"] += 1

            acc(pred_label, label)

        print("Evaluation Result")

        for key in label_result.keys():
            TP = label_result[key]["TP"]
            FN = label_result[key]["FN"]
            FP = label_result[key]["FP"]

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = 2 * (precision * recall) / (precision + recall)

            print(f"Class : {key}")
            print(f"TP : {TP}")
            print(f"FN : {FN}")
            print(f"FP : {FP}")
            print(f"Precision : {precision}")
            print(f"Recall : {recall}")
            print(f"F1-Score : {f1_score}")
            print("\n")

        accuracy = acc.compute()
        print(f"Accuracy : {accuracy}")


if __name__ == "__main__":
    main()
