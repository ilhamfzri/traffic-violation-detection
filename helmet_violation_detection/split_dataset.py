import os
import shutil
import sys

sys.path.append("/content/TrafficViolationDetection")

from sklearn.model_selection import train_test_split
from tvdr.utils.path import create_folder

INPUT_DIR = "/content/drive/MyDrive/Skripsi/Helmet Detector/Datasets/Kaggle"
OUTPUT_DIR = "/content/drive/MyDrive/Skripsi/Helmet Detector/Datasets/Kaggle/splited"


def move_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False


def main():
    # Read images and annotations
    input_images_dir = os.path.join(INPUT_DIR, "images")
    input_annotations_dir = os.path.join(INPUT_DIR, "annotations")

    images = [os.path.join(input_images_dir, x) for x in os.listdir(input_images_dir)]
    annotations = [
        os.path.join(input_annotations_dir, x)
        for x in os.listdir(input_annotations_dir)
        if x[-3:] == "txt"
    ]

    images.sort()
    annotations.sort()

    # Split the dataset into train-valid-test splits
    train_images, val_images, train_annotations, val_annotations = train_test_split(
        images, annotations, test_size=0.2, random_state=1
    )
    val_images, test_images, val_annotations, test_annotations = train_test_split(
        val_images, val_annotations, test_size=0.5, random_state=1
    )

    create_folder(os.path.join(OUTPUT_DIR, "images/train"))
    create_folder(os.path.join(OUTPUT_DIR, "images/val"))
    create_folder(os.path.join(OUTPUT_DIR, "images/test"))

    create_folder(os.path.join(OUTPUT_DIR, "annotations/train"))
    create_folder(os.path.join(OUTPUT_DIR, "annotations/val"))
    create_folder(os.path.join(OUTPUT_DIR, "annotations/test"))

    move_files_to_folder(train_images, os.path.join(OUTPUT_DIR, "images/train"))
    move_files_to_folder(val_images, os.path.join(OUTPUT_DIR, "images/val"))
    move_files_to_folder(test_images, os.path.join(OUTPUT_DIR, "images/test"))
    move_files_to_folder(
        train_annotations, os.path.join(OUTPUT_DIR, "annotations/train")
    )
    move_files_to_folder(val_annotations, os.path.join(OUTPUT_DIR, "annotations/val"))
    move_files_to_folder(test_annotations, os.path.join(OUTPUT_DIR, "annotations/test"))


if __name__ == "__main__":
    main()
