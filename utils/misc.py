import cv2
import numpy as np
from glob import glob

def load_img(img_path, img_size=(256, 256)):
    source_imgs, source_imgs_names = [], []
    imgs_list = sorted(glob(img_path))
    for file_path in imgs_list:
        img = cv2.imread(file_path)
        if img.ndim == 2:
            img = np.tile(img[..., None], [1, 1, 3])
        # img = img[..., :3][..., ::-1]
        img = img[..., :3]
        img = cv2.resize(img, img_size)
        source_imgs.append(img)
        source_imgs_names.append(file_path.split('\\')[-1].split('.')[0])
    return source_imgs, source_imgs_names