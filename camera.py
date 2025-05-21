import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
from argparse import ArgumentParser

from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector

from utils import *

def main_loop(opt):
    # open camera
    cap = VideoCaptureAsync(0)
    cap.start()
    if not cap.isOpened():
        print("cannot open camera")
        return

    # load FOMM model
    generator, kp_detector = load_checkpoints(
        config_path=opt.config, checkpoint_path=opt.checkpoint)
    predictor = real_time_FOMM(generator, kp_detector)

    source_imgs, source_imgs_names = load_img(opt.source_image)
    predictor.set_source(source_imgs[0])

    # loop 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot receive frame (stream end?). Exiting ...")
            break

        camera_input = cropping_frame(frame)
        FOMM_output = predictor.predict(camera_input)

        combined = combine_frames(camera_input, FOMM_output, source_imgs_names[0])
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
    
    parser.add_argument("--source_image", default='data\\portraits\\*', help="path to source image")

    opt = parser.parse_args()

    main_loop(opt)