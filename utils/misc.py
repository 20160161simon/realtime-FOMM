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
        img = img[..., :3][..., ::-1]
        # img = img[..., :3]
        img = cv2.resize(img, img_size)
        source_imgs.append(img)
        source_imgs_names.append(file_path.split('\\')[-1].split('.')[0])
    return source_imgs, source_imgs_names

# def cropping_frame(frame, target_size=(256, 256)):
    # frame.shape = (480, 640, 3)
    # FOMM need RGB format
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame, target_size)
    # frame = cv2.flip(frame, 1)  # flip horizontally
def cropping_frame(img, p=0.7, offset_x=0, offset_y=0):

    def clamp(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    h, w = img.shape[:2]
    x = int(min(w, h) * p)
    l = (w - x) // 2
    r = w - l
    u = (h - x) // 2
    d = h - u

    offset_x = clamp(offset_x, -l, w - r)
    offset_y = clamp(offset_y, -u, h - d)

    l += offset_x
    r += offset_x
    u += offset_y
    d += offset_y

    return img[u:d, l:r], (offset_x, offset_y)

    # return frame

def combine_frames(camera_input, FOMM_output, source_frame_name):

    camera_input = cv2.flip(camera_input, 1)  # flip horizontally
    FOMM_output = cv2.flip(FOMM_output, 1)  # flip horizontally

    # add border
    border_size = 6
    camera_input = cv2.copyMakeBorder(
        camera_input, border_size, border_size, border_size, border_size // 2,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    FOMM_output = cv2.copyMakeBorder(
        FOMM_output, border_size, border_size, border_size, border_size - border_size // 2,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    # add label
    label_height = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (0, 0, 0)
    thickness = 1
    text_left = "Camera Input"
    text_right = source_frame_name

    camera_labeled = cv2.copyMakeBorder(
        camera_input, 0, label_height, 0, 0,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    FOMM_labeled = cv2.copyMakeBorder(
        FOMM_output, 0, label_height, 0, 0,
        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    (text_w, text_h), _ = cv2.getTextSize(text_left, font, font_scale, thickness)
    x_left = (camera_labeled.shape[1] - text_w) // 2
    cv2.putText(
        camera_labeled, text_left, (x_left, 256 + 2 + label_height - 10),
        font, font_scale, font_color, thickness, cv2.LINE_AA)

    (text_w, text_h), _ = cv2.getTextSize(text_right, font, font_scale, thickness)
    x_right = (FOMM_labeled.shape[1] - text_w) // 2
    cv2.putText(
        FOMM_labeled, text_right, (x_right, 256 + 2 + label_height - 10),
        font, font_scale, font_color, thickness, cv2.LINE_AA)
    
    combined = cv2.hconcat([camera_labeled, FOMM_labeled])
    return combined
