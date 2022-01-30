import cv2
import xml.etree.ElementTree as ET
import pathlib
import os
import numpy as np

from tvdr.utils.path import create_folder

INPUT_DIR = "datasets/helmet_violation"
OUTPUT_DIR = "outputs"
FILESNAME = "kaggle_datasets"


def main():
    input_annotations_dir = os.path.join(INPUT_DIR, "annotations")
    input_images_dir = os.path.join(INPUT_DIR, "images")

    output_helmet_dir = os.path.join(OUTPUT_DIR, "helmet")
    output_non_helmet_dir = os.path.join(OUTPUT_DIR, "non_helmet")

    create_folder(output_helmet_dir)
    create_folder(output_non_helmet_dir)

    count_helmet = 0
    count_non_helmet = 0

    images_path_list = os.listdir(input_images_dir)
    for image_path in images_path_list:
        annotation_path = os.path.join(
            input_annotations_dir, "{}.xml".format(image_path[:-4])
        )
        image_path = os.path.join(input_images_dir, image_path)

        image_frame = cv2.imread(image_path)

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for child in root:
            if child.tag == "object":
                for grand_child in child:
                    if grand_child.tag == "name":

                        if grand_child.text == "With Helmet":
                            filename_path = os.path.join(
                                output_helmet_dir,
                                "{}_helmet_{}.jpg".format(FILESNAME, count_helmet),
                            )
                            count_helmet += 1
                        else:
                            filename_path = os.path.join(
                                output_helmet_dir,
                                "{}_nonhelmet_{}.jpg".format(
                                    FILESNAME, count_non_helmet
                                ),
                            )
                            count_non_helmet += 1

                    if grand_child.tag == "bndbox":
                        bounding_box = []
                        for bnd_box in grand_child:
                            bounding_box.append(int(bnd_box.text))

                        crop_img = image_frame[
                            bounding_box[1] : bounding_box[3],
                            bounding_box[0] : bounding_box[2],
                        ]

                        cv2.imwrite(filename_path, crop_img)

    # img = cv2.imread("datasets/helmet_violation/BikesHelmets0.png")
    # tree = ET.parse("datasets/helmet_violation/BikesHelmet0.xml")

    # root = tree.getroot()
    # print(root.tag)
    # for child in root:
    #     if child.tag == "object":
    #         for grandchild in child:
    #             print(grandchild.tag, grandchild.text)
    #             for grand_grandchild in grandchild:
    #                 print(grand_grandchild.tag, grand_grandchild.text)


if __name__ == "__main__":
    main()
