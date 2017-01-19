from PIL import Image, ImageEnhance, ImageOps
import cv2
import numpy as np

# Shape of resized images
top = 60
bottom = 130
left = 0
right = 320
channels = 3

img_config = {
    'top': top,
    'bottom': bottom,
    'left': left,
    'right': right,
    'channels': channels,
    'img_height': bottom - top,
    'img_width': right - left,
    'resized_2Dshape': (right - left, bottom - top),
    'resized_3Dshape': (bottom - top, right - left, channels)
}

def process_img(img):
    img = img.crop(box=(img_config['left'], img_config['top'], img_config['right'], img_config['bottom']))
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)

    return img