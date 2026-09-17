from AI_Script.preprocess.base_preprocess import BasePreprocessor
from AI_Script.preprocess.registry_preprocess import PreprocessRegistry
from AI_Script.core.utils import check_file
import cv2
import numpy as np


@PreprocessRegistry.register("detr")
class DETRPreprocessor(BasePreprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.target_size = tuple(self.config.get("target_size", (640, 640)))
        self.auto_pad_color = (114, 114, 114)

    def letterbox(self, img: np.ndarray):
        shape = img.shape[:2]  # (h, w)
        new_shape = self.target_size
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.auto_pad_color)
        return img, r, (dw, dh)

    def preprocess(self, input) -> np.ndarray:
        if check_file(input) == 'npy_path':
            processed = np.load(input)
        elif check_file(input) == 'image_path':
            processed = cv2.imread(input)
        elif check_file(input) == 'numpy_array':
            processed = input
        else:
            raise ValueError("Invalid input format")

        processed, ratio, (dw, dh) = self.letterbox(processed)
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        processed = processed.astype(np.float32) / 255.0
        processed = np.transpose(processed, (2, 0, 1))
        processed = np.expand_dims(processed, axis=0)
        return processed
