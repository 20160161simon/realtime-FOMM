import cv2
import numpy as np

def load_img(img_path, img_size=(256, 256)):
    img = cv2.imread(img_path)
    if img.ndim == 2:
        img = np.tile(img[..., None], [1, 1, 3])
    img = img[..., :3][..., ::-1]
    img = cv2.resize(img, img_size)
    return img