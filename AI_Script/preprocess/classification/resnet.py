# preprocess/processors/resnet.py
from AI_Script.preprocess.base_preprocess import BasePreprocessor
from AI_Script.preprocess.registry_preprocess import PreprocessRegistry
from AI_Script.core.utils import check_file
import cv2
import numpy as np
from PIL import Image

# decorator = register_preprocessor("resnet")
# ResnetPreprocessor = decorator(ResnetPreprocessor)
@PreprocessRegistry.register("resnet")
@PreprocessRegistry.register("resnet16")
@PreprocessRegistry.register("resnet50")
class ResnetPreprocessor(BasePreprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.target_size = tuple(self.config.get("target_size", (224, 224)))
        self.mean = np.array([0.3445, 0.3803, 0.4077], dtype=np.float32)
        self.std = np.array([0.0915, 0.0652, 0.0553], dtype=np.float32)

    def preprocess(self, input) -> np.ndarray:
        # Load image
        if check_file(input) == 'npy_path':
            processed = np.load(input)
        elif check_file(input) == 'image_path':
            processed = cv2.imread(input)
        elif check_file(input) == 'numpy_array':
            processed = input
        else:
            raise ValueError("Invalid input format")

        # Resize to 224x224
        processed = cv2.resize(processed, self.target_size)

        # BGR -> RGB
        if processed.shape[-1] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # Normalize to [0,1]
        processed = processed.astype(np.float32) / 255.0

        # Normalize EuroSAT
        processed = (processed - self.mean) / self.std

        # HWC -> CHW
        processed = np.transpose(processed, (2, 0, 1))

        # Add batch dim
        processed = np.expand_dims(processed, axis=0)

        return processed