import yaml
import torch
import numpy as np

from sync_batchnorm import DataParallelWithCallback
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp

def load_checkpoints(config_path, checkpoint_path):

    with open(config_path) as f:
        config = yaml.full_load(f)

    generator = OcclusionAwareGenerator(
        **config['model_params']['generator_params'],
        **config['model_params']['common_params']).cuda()
    kp_detector = KPDetector(
        **config['model_params']['kp_detector_params'],
        **config['model_params']['common_params']).cuda()
    checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    generator = DataParallelWithCallback(generator)
    kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

def convert_to_model_input(frame):
    return torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2) / 255

class real_time_FOMM:
    def __init__(self, generator, kp_detector):
        self.generator = generator
        self.kp_detector = kp_detector

        self.source_frame = None
        self.kp_source = None

        self.kp_driving_initial = None
    
    def set_source(self, source_frame):
        with torch.no_grad():
            self.source_frame = convert_to_model_input(source_frame).cuda()
            self.kp_source = self.kp_detector(self.source_frame)

    def predict(self, driving_frame):
        with torch.no_grad():
            driving = convert_to_model_input(driving_frame).cuda()
            kp_driving = self.kp_detector(driving)
            if self.kp_driving_initial is None:
                self.kp_driving_initial = kp_driving
            
            kp_norm = normalize_kp(
                kp_source=self.kp_source, kp_driving=kp_driving,
                kp_driving_initial=self.kp_driving_initial, use_relative_movement=True,
                use_relative_jacobian=True, adapt_movement_scale=True)
            out = self.generator(self.source_frame, kp_source=self.kp_source, kp_driving=kp_norm)
        
            out = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
        return out


