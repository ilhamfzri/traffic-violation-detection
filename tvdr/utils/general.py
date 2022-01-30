import torch
import numpy as np
import cv2
import math

from scipy.spatial import distance


def sort_validity(sort: np.ndarray, frame_shape):
    new_sort = np.empty((0, 7))
    for i in range(0, sort.shape[0]):
        data = sort[i]
        if data[2] > data[0] and data[3] > data[1]:
            if (
                data[0] >= 0
                and data[2] <= frame_shape[0]
                and data[1] >= 0
                and data[3] <= frame_shape[1]
            ):
                new_sort = np.vstack((new_sort, data))

    return new_sort


def combine_yolo_sort_result(yolo: np.ndarray, sort: np.ndarray):

    new_sort = np.empty((0, 7))
    for sort_data in sort:
        distance_calculate = []
        for i in range(0, yolo.shape[0]):
            yolo_data = yolo[i]
            x = np.array(
                [yolo_data[0], yolo_data[1], yolo_data[2], yolo_data[3]],
                dtype=np.float32,
            )
            y = np.array(
                [
                    sort_data[0],
                    sort_data[1],
                    sort_data[2],
                    sort_data[3],
                ],
                dtype=np.float32,
            )
            similarity = 1 - distance.cosine(x, y)
            distance_calculate.append(similarity)

        index = np.argmax(distance_calculate)

        new_data = np.array(
            [
                sort_data[0],
                sort_data[1],
                sort_data[2],
                sort_data[3],
                yolo[index][4],
                yolo[index][5],
                sort_data[4],
            ]
        )
        new_sort = np.vstack((new_sort, new_data))

    return new_sort


def combine_yolo_deepsort_result(yolo: np.ndarray, deep_sort: np.ndarray):

    new_deep_sort = np.empty((0, 7))
    for deep_sort_data in deep_sort:
        distance_calculate = []
        for i in range(0, yolo.shape[0]):
            yolo_data = yolo[i]
            x = np.array(
                [yolo_data[0], yolo_data[1], yolo_data[2], yolo_data[3]],
                dtype=np.float32,
            )
            y = np.array(
                [
                    deep_sort_data[0],
                    deep_sort_data[1],
                    deep_sort_data[2],
                    deep_sort_data[3],
                ],
                dtype=np.float32,
            )
            similarity = 1 - distance.cosine(x, y)
            distance_calculate.append(similarity)

        index = np.argmax(distance_calculate)

        new_data = np.array(
            [
                deep_sort_data[0],
                deep_sort_data[1],
                deep_sort_data[2],
                deep_sort_data[3],
                yolo[index][4],
                deep_sort_data[5],
                deep_sort_data[4],
            ]
        )
        new_deep_sort = np.vstack((new_deep_sort, new_data))

    return new_deep_sort


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, ratio, (dw, dh)


def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else:  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(
            f"WARNING: --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}"
        )
    return new_size


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor
