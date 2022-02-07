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

import sys

sys.path.append("yolov5_repo")

import os
import cv2
import xml.etree.ElementTree as ET
import xml.etree.ElementTree as gfg
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse

from xml.dom import minidom
from tqdm import tqdm
from yolov5_repo.utils.torch_utils import select_device
from yolov5_repo.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5_repo.utils.augmentations import letterbox
from yolov5_repo.models.common import DetectMultiBackend
from tvdr.utils.path import create_folder

IMGSZ = 640


def get_parser():
    parser = argparse.ArgumentParser(
        description="Preprocessing Helmet Violation Detection Dataset"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None, help="Path to dataset input directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to dataset output directory"
    )
    parser.add_argument(
        "--vehicle_detection_model",
        type=str,
        default=None,
        help="Path to vehilce detection model",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.25,
        help="Confidience threshold for croping 2 wheels vehicle",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.45,
        help="IOU threshold for croping 2 wheel vehicle",
    )

    return parser


def load_model(model_path, imgsz0):
    device_torch = select_device(device="cpu")
    model = DetectMultiBackend(model_path, device=device_torch)
    model_stride = model.stride
    imgsz = check_img_size(imgsz0, model_stride)
    return model, imgsz, model_stride


def predict(model, imgsz, stride, img0, conf_thres, iou_thres):
    im = letterbox(
        img0,
        new_shape=imgsz,
        stride=stride,
        auto=True,
        scaleFill=True,
    )[0]

    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to("cpu").float()
    im /= 255

    if len(im.shape) == 3:
        im = im[None]

    result = model(im)
    result = non_max_suppression(
        prediction=result,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        max_det=500,
    )

    result = result[0]
    result[:, :4] = scale_coords(im.shape[2:], result[:, :4], img0.shape).round()

    result = result.numpy()

    return result


def GenerateXML(
    output_path,
    bbox_data,
    image_size,
):
    root = gfg.Element("annotation")

    folder = gfg.Element("folder")
    folder.text = "images"

    size = gfg.Element("size")
    width = gfg.SubElement(size, "width")
    height = gfg.SubElement(size, "height")
    depth = gfg.SubElement(size, "depth")

    width.text = str(image_size[0])
    height.text = str(image_size[1])
    depth.text = str(3)

    root.append(folder)
    root.append(size)

    segmented = gfg.Element("segmented")
    segmented.text = str(0)
    root.append(segmented)

    for bbox in bbox_data:
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[2]
        ymax = bbox[3]
        label = bbox[4]

        object_data = gfg.Element("object")

        name_object = gfg.SubElement(object_data, "name")
        name_object.text = label

        pase_object = gfg.SubElement(object_data, "pose")
        pase_object.text = "Unspecified"

        truncated_object = gfg.SubElement(object_data, "truncated")
        truncated_object.text = str(0)

        occluded_object = gfg.SubElement(object_data, "occluded")
        occluded_object.text = str(0)

        difficult_object = gfg.SubElement(object_data, "difficult")
        difficult_object.text = str(0)

        bbox_data = gfg.Element("bndbox")
        xmin_bbox = gfg.SubElement(bbox_data, "xmin")
        xmin_bbox.text = str(xmin)
        ymin_bbox = gfg.SubElement(bbox_data, "ymin")
        ymin_bbox.text = str(ymin)

        xmax_bbox = gfg.SubElement(bbox_data, "xmax")
        xmax_bbox.text = str(xmax)
        ymax_bbox = gfg.SubElement(bbox_data, "ymax")
        ymax_bbox.text = str(ymax)

        object_data.append(bbox_data)

        root.append(object_data)

    tree = gfg.ElementTree(root)

    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="   ")
    with open(output_path, "w") as file_data:
        file_data.write(xmlstr)


def extract_info_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()

    # Initialise the info dict
    info_dict = {}
    info_dict["bboxes"] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict["filename"] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(subelem.text)

            info_dict["image_size"] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = subsubelem.text
            info_dict["bboxes"].append(bbox)

    return info_dict


def crop_image(img0, bbox):
    x, y = int(bbox[0]), int(bbox[1])
    w, h = int(abs(bbox[2] - bbox[0])), int(abs(bbox[3] - bbox[1]))
    img_crop = img0[y : y + h, x : x + w, :]
    return img_crop


def overlap_rectangle(bbox_1, bbox_2):
    bbox_1_area = abs(bbox_1[2] - bbox_1[0]) * abs(bbox_1[3] - bbox_1[1])
    bbox_2_area = abs(bbox_2[2] - bbox_2[0]) * abs(bbox_2[3] - bbox_2[1])

    intersecting_area = max(
        0, min(bbox_1[2], bbox_2[2]) - max(bbox_1[0], bbox_2[0])
    ) * max(0, min(bbox_1[3], bbox_2[3]) - max(bbox_1[1], bbox_2[1]))
    percent_coverage = intersecting_area / (
        bbox_1_area + bbox_2_area - intersecting_area
    )
    return percent_coverage


def calculate_new_frame_bbox(object_bbox, label_data):
    x_min = object_bbox[0]
    x_max = object_bbox[2]
    y_min = object_bbox[1]
    y_max = object_bbox[3]

    for label_bbox in label_data:
        if label_bbox[0] < x_min:
            x_min = label_bbox[0]
        if label_bbox[1] < y_min:
            y_min = label_bbox[1]
        if label_bbox[2] > x_max:
            x_max = label_bbox[2]
        if label_bbox[3] > y_max:
            y_max = label_bbox[3]

    return x_min, y_min, x_max, y_max


def generate_data(
    new_annotation_data, image_dir_out, annotation_dir_out, basename, img
):

    if len(new_annotation_data.keys()) > 0:
        for i, key in enumerate(new_annotation_data.keys()):
            image_path = os.path.join(image_dir_out, f"{basename}_{i}.jpg")
            annotation_path = os.path.join(annotation_dir_out, f"{basename}_{i}.xml")

            annotation_data = new_annotation_data[key]
            object_bbox = annotation_data["bbox_frame"].astype(int)

            xmin, ymin, xmax, ymax = calculate_new_frame_bbox(
                object_bbox, annotation_data["label_data"]
            )

            img_size = (
                int(object_bbox[2] - object_bbox[0]),
                int(object_bbox[3] - object_bbox[1]),
            )
            new_frame = img[ymin:ymax, xmin:xmax]
            cv2.imwrite(image_path, new_frame)

            GenerateXML(annotation_path, annotation_data["label_data"], img_size)


def annotation_combiner(predict_result, annotation_data):
    annotations_bbox = annotation_data["bboxes"]
    new_annotation_data = {}

    for i, object in enumerate(predict_result):
        object_class = object[5]
        if object_class != 1 and object_class != 4:
            continue

        bbox_object = object[0:4]
        list_label_data = []
        for annotation_bbox in annotations_bbox:
            bbox_helmet = [
                int(annotation_bbox["xmin"]),
                int(annotation_bbox["ymin"]),
                int(annotation_bbox["xmax"]),
                int(annotation_bbox["ymax"]),
            ]
            overlap = overlap_rectangle(bbox_object, bbox_helmet)
            if overlap > 0:

                bbox_helmet.append(annotation_bbox["class"])
                list_label_data.append(bbox_helmet)

        if len(list_label_data) > 0:
            dict_data = {"bbox_frame": bbox_object, "label_data": list_label_data}
            new_annotation_data[i] = dict_data

    return new_annotation_data


def annotation_image(img, list_bbox):
    img_new = img.copy()
    for bbox in list_bbox:
        img_new = cv2.rectangle(
            img=img_new,
            pt1=(int(bbox[0]), int(bbox[1])),
            pt2=(int(bbox[2]), int(bbox[3])),
            color=(0, 0, 255),
            thickness=2,
        )

    return img_new


def main():
    args = get_parser().parse_args()

    images_input_path = os.path.join(args.input_dir, "images")
    annotations_input_path = os.path.join(args.input_dir, "annotations")

    images_output_path = os.path.join(args.output_dir, "images")
    annotations_output_path = os.path.join(args.output_dir, "annotations")

    create_folder(images_output_path)
    create_folder(annotations_output_path)

    image_list = os.listdir(images_input_path)
    detector, imgsz, stride = load_model(args.vehicle_detection_model, IMGSZ)

    for image in tqdm(image_list):
        try:
            image_path = os.path.join(images_input_path, image)
            annotation_path = os.path.join(annotations_input_path, f"{image[:-4]}.xml")

            # Read image data frame
            frame_data = cv2.imread(image_path)

            # Detect vehicle
            predict_result = predict(
                detector, imgsz, stride, frame_data, args.conf_thres, args.iou_thres
            )

            annotation_data = extract_info_from_xml(annotation_path)

            # Annotate original image
            # image_annotated_out = os.path.join(IMAGES_PATH_OUT, f"{image[:-4]}.jpg")
            # predict_2_wheels = [
            #     data_predict
            #     for data_predict in predict_result
            #     if data_predict[5] == 1 or data_predict[5] == 4
            # ]
            # image_annotated_frame = annotation_image(frame_data, predict_2_wheels)

            # list_helmet_data = []
            # for i in range(0, len(annotation_data["bboxes"])):
            #     bbox_data = annotation_data["bboxes"][i]
            #     bbox = [
            #         int(bbox_data["xmin"]),
            #         int(bbox_data["ymin"]),
            #         int(bbox_data["xmax"]),
            #         int(bbox_data["ymax"]),
            #     ]
            #     list_helmet_data.append(bbox)

            # image_annotated_frame = annotation_image(
            #     image_annotated_frame, list_helmet_data
            # )

            new_annotation = annotation_combiner(predict_result, annotation_data)
            generate_data(
                new_annotation,
                images_output_path,
                annotations_output_path,
                image[:-4],
                frame_data,
            )
        except:
            pass


if __name__ == "__main__":
    main()
