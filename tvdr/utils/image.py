from hashlib import new
from PIL import Image
import cv2


def png_to_jpg(image_input: str, image_output: str):
    im = Image.open(image_input)
    im = im.convert("RGB")
    im.save(image_output, quality=95)


def resize(im, dst_shape):
    # if im.dtype == "uint8":
    #     print("here")
    #     im = im.astype("float32")
    src_shape = (im.shape[0], im.shape[1])
    print(src_shape)
    new_im = cv2.resize(im, dst_shape)
    r = (dst_shape[0] / src_shape[0], dst_shape[1] / dst_shape[1])
    print(new_im.dtype)
    return new_im, r
