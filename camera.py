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

    frame_proportion = 0.9
    frame_offset_x = 0
    frame_offset_y = 0

    # loop 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("cannot receive frame (stream end?). Exiting ...")
            break

        frame = frame[..., ::-1]
        # camera_input = cropping_frame(frame)
        frame, (frame_offset_x, frame_offset_y) = cropping_frame(frame, p=frame_proportion, offset_x=frame_offset_x, offset_y=frame_offset_y)
        camera_input = cv2.resize(frame, (256, 256))

        if find_best_frame(camera_input, predictor):
            predictor.kp_driving_initial = None
            
        FOMM_output = predictor.predict(camera_input)

        combined = combine_frames(camera_input[..., ::-1], FOMM_output[..., ::-1], source_imgs_names[0])
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