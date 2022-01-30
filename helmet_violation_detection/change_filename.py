import shutil
import os
import sys
from tvdr.utils.image import png_to_jpg
from tqdm import tqdm

INPUT_DIR = "/Users/hamz/Documents/Kuliah/Semester 7/Skripsi Bismillah/Dataset/Helmet Violation Detection"
OUTPUT_DIR = "/Users/hamz/Documents/Kuliah/Semester 7/Skripsi Bismillah/Dataset/helmet_violation_dataset"
FILENAME = "MotorcycleHelmet"


def main():
    list_images = os.listdir(INPUT_DIR)

    count = 0
    for i, filename in tqdm(enumerate(list_images)):
        if filename[-4:] == ".jpg":
            shutil.copy(
                src=os.path.join(INPUT_DIR, filename),
                dst=os.path.join(OUTPUT_DIR, f"{FILENAME}_{i}.jpg"),
            )
        elif filename[-4:] == ".png":
            print(f"{FILENAME}_{i}.jpg")
            png_to_jpg(
                image_input=os.path.join(INPUT_DIR, filename),
                image_output=os.path.join(OUTPUT_DIR, f"{FILENAME}_{i}.jpg"),
            )
        else:
            shutil.copy(
                src=os.path.join(INPUT_DIR, filename),
                dst=os.path.join(OUTPUT_DIR, f"{FILENAME}_{i}.jpg"),
            )


if __name__ == "__main__":
    main()
