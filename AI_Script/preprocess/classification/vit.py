from AI_Script.preprocess.base_preprocess import BasePreprocessor
from AI_Script.preprocess.registry_preprocess import PreprocessRegistry
from AI_Script.core.utils import check_file
import cv2
import numpy as np
from PIL import Image


@PreprocessRegistry.register("vit")
class ViTPreprocessor(BasePreprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.target_size = tuple(self.config.get("target_size", (224, 224)))

    def _load(self, input):
        if check_file(input) == 'npy_path':
            arr = np.load(input)
            # npy is CHW float or HWC uint8 -> convert to PIL RGB
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.0 else (arr * 255).astype(np.uint8)
            return Image.fromarray(arr[..., ::-1] if arr.shape[2] == 3 else arr)
        elif check_file(input) == 'image_path':
            return Image.open(input).convert("RGB")
        elif check_file(input) == 'numpy_array':
            arr = input
            if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
                arr = np.transpose(arr, (1, 2, 0))
            # BGR (cv2) -> RGB
            if arr.ndim == 3 and arr.shape[2] == 3:
                arr = arr[..., ::-1]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.0 else (arr * 255).astype(np.uint8)
            return Image.fromarray(arr)
        else:
            raise ValueError("Invalid input format")

    def _preprocess_manual(self, image: Image.Image) -> np.ndarray:
        size = self.target_size[0]
        image = image.resize((size, size), Image.BILINEAR)
        arr = np.array(image, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def _preprocess_torchvision(self, image: Image.Image) -> np.ndarray:
        size = self.target_size[0]
        w, h = image.size
        if h < w:
            new_h, new_w = 256, int(w * 256 / h)
        else:
            new_w, new_h = 256, int(h * 256 / w)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        w, h = image.size
        left, top = (w - size) // 2, (h - size) // 2
        image = image.crop((left, top, left + size, top + size))
        arr = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def preprocess(self, input) -> np.ndarray:
        image = self._load(input)
        # INT8 model uses torchvision-style preprocess, else manual
        if "int8" in str(self.precision_format).lower():
            return self._preprocess_torchvision(image)
        return self._preprocess_manual(image)
