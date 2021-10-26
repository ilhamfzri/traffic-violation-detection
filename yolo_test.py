import sys

import cv2
from PIL import Image
from tvdr.core import YOLOModel


def main():
    x = cv2.imread("samples/motor.jpeg")
    print(type(x))
    yolo_model = YOLOModel(device="cpu")
    yolo_model.load_model()
    img = yolo_model.inference_frame(x)
    cv2.imshow("test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
