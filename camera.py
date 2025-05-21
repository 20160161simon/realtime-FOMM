import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
from argparse import ArgumentParser

from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector

from utils import *

def make_animation(frame):

    animated_frame = cv2.flip(frame, 1)
    return animated_frame

def main_loop(kp_detector, generator):
    # open camera
    cap = VideoCaptureAsync(0)
    cap.start()
    if not cap.isOpened():
        print("cannot open camera")
        return

    # loop 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot receive frame (stream end?). Exiting ...")
            break

        camera_input = cropping_frame(frame)
        FOMM_output = make_animation(camera_input)

        combined = combine_frames(camera_input, FOMM_output)
        cv2.imshow('real-time FOMM (key \'q\' to exit)', combined)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            # key 'q' to exit
            break

    cap.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if torch.cuda.is_available() is False:
        print("CUDA is not available. Please check your CUDA installation.")
        exit(1)

    parser = ArgumentParser()
    parser.add_argument("--config", default='config\\vox-adv-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='checkpoints\\vox-adv-cpk.pth.tar', help="path to checkpoint to restore")
    
    parser.add_argument("--source_image", default='data\\potter.jpg', help="path to source image")

    opt = parser.parse_args()

    kp_detector, generator = load_checkpoints(
        config_path=opt.config, checkpoint_path=opt.checkpoint)
    
    source_image = load_img(opt.source_image)

    main_loop(kp_detector, generator)