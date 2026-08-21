from AI_Script.preprocess.base_preprocess import BasePreprocessor
from AI_Script.preprocess.registry_preprocess import PreprocessRegistry
from AI_Script.core.utils import check_file
import cv2
import numpy as np
from PIL import Image

@PreprocessRegistry.register("lenet5")
class LeNet5Preprocessor(BasePreprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.target_size = tuple(self.config.get("target_size", (32, 32)))
        self.mean = np.array(self.config.get("mean", 0), dtype=np.float32)
        self.std = np.array(self.config.get("std", 1), dtype=np.float32)

    def preprocess(self, input) -> np.ndarray:
        # Load image
        if check_file(input) == 'npy_path':
            processed = np.load(input)
        elif check_file(input) == 'image_path':
            processed = cv2.imread(input)
        elif check_file(input) == 'numpy_array':
            processed = input
        else:
            print("==ERROR: Input format is invalid==")
        # Grayscale
        if processed.ndim == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        # Resize
        processed = cv2.resize(processed, self.target_size)
        # Add channel
        processed = processed[..., np.newaxis]
        # Normalize to [0,1]
        processed = processed.astype(np.float32) / 255.0
        # Normalize ImageNet
        processed = (processed - self.mean) / self.std
        # HWC -> CHW
        processed = np.transpose(processed, (2, 0, 1))
        # Add batch dim
        processed = np.expand_dims(processed, axis=0)

        return processed