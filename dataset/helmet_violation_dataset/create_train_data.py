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

import argparse
import os
import random
import shutil

from tqdm import tqdm
from tvdr.utils.path import create_folder
import xml.etree.ElementTree as ET

class_name_to_id_mapping = {
    "With Helmet": 0,
    "Without Helmet": 1,
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="Split dataset to val,train,test and convert annotation file to yolov5 format for training and evaluate process"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None, help="Path to dataset input directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to dataset output directory"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help="train ratio for spliting dataset",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.25,
        help="val ratio for spliting dataset",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.45,
        help="test ratio for spliting dataset",
    )
    return parser


def convert_to_yolov5(info_dict, output_path):
    print_buffer = []

    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = b["xmax"] - b["xmin"]
        b_height = b["ymax"] - b["ymin"]

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                class_id, b_center_x, b_center_y, b_width, b_height
            )
        )

    print("\n".join(print_buffer), file=open(output_path, "w"))


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
                image_size.append(int(subelem.text))

            info_dict["image_size"] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict["bboxes"].append(bbox)

    return info_dict


def main():
    args = get_parser().parse_args()
    print(f"Input Directory : {args.input_dir}")
    print(f"Output Directory : {args.output_dir}")

    input_images_dir = os.path.join(args.input_dir, "images")
    input_annotations_dir = os.path.join(args.input_dir, "annotations")

    output_train_images_dir = os.path.join(
        args.output_dir, os.path.join("images", "train")
    )
    output_val_images_dir = os.path.join(args.output_dir, os.path.join("images", "val"))
    output_test_images_dir = os.path.join(
        args.output_dir, os.path.join("images", "test")
    )

    output_train_labels_dir = os.path.join(
        args.output_dir, os.path.join("labels", "train")
    )
    output_val_labels_dir = os.path.join(args.output_dir, os.path.join("labels", "val"))
    output_test_labels_dir = os.path.join(
        args.output_dir, os.path.join("labels", "test")
    )

    create_folder(output_train_images_dir)
    create_folder(output_val_images_dir)
    create_folder(output_test_images_dir)

    create_folder(output_train_labels_dir)
    create_folder(output_val_labels_dir)
    create_folder(output_test_labels_dir)

    size_dataset = len(os.listdir(input_images_dir))
    print(f"Size Dataset Total : {size_dataset}")

    size_train = int(size_dataset * args.train_ratio)
    size_val = int(size_dataset * args.val_ratio)

    list_files = os.listdir(input_images_dir)
    random.shuffle(list_files)
    list_train = list_files[:size_train]
    list_val = list_files[size_train : size_train + size_val]
    list_test = list_files[size_train + size_val :]

    print(f"Size Train : {len(list_train)}")
    print(f"Size Validation : {len(list_val)}")
    print(f"Size Test : {len(list_test)}")

    print("Splitting Dataset ...")

    list_set = [list_train, list_val, list_test]
    list_name = ["train", "val", "test"]

    for i, list_files in enumerate(list_set):
        print(f"Create {list_name[i]} set")
        for file in tqdm(list_files):
            input_image_path = os.path.join(input_images_dir, file)
            input_annotation_path = os.path.join(
                input_annotations_dir, f"{file[:-4]}.xml"
            )
            if list_name[i] == "train":
                output_images_dir = output_train_images_dir
                output_labels_dir = output_train_labels_dir

            elif list_name[i] == "val":
                output_images_dir = output_val_images_dir
                output_labels_dir = output_val_labels_dir
            else:
                output_images_dir = output_test_images_dir
                output_labels_dir = output_test_labels_dir

            output_image_path = os.path.join(output_images_dir, file)
            output_label_path = os.path.join(output_labels_dir, f"{file[:-4]}.txt")

            shutil.copy(input_image_path, output_image_path)
            info_dict = extract_info_from_xml(input_annotation_path)
            convert_to_yolov5(info_dict, output_label_path)


if __name__ == "__main__":
    main()
