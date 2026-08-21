"""YOLOv5 preprocessing - letterbox, normalize, tensor conversion."""
import cv2
import numpy as np
from typing import Tuple, Union, Any, Optional
from pathlib import Path


class YOLOv5Preprocessor:
    """YOLOv5-specific preprocessing: letterbox resize, normalize, CHW format."""
    
    def __init__(self, config: dict):
        self.config = config
        target_size = config.get("target_size")
        if target_size is None:
            target_size = [640, 640]
        self.target_size = tuple(target_size)
        
        mean = config.get("mean")
        if mean is None:
            mean = [0.0, 0.0, 0.0]
        self.mean = np.array(mean, dtype=np.float32)
        
        std = config.get("std")
        if std is None:
            std = [1.0, 1.0, 1.0]
        self.std = np.array(std, dtype=np.float32)
        
        self.auto_pad_color = (114, 114, 114)
    
    def letterbox(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resize and pad image with letterbox to target size."""
        shape = img.shape[:2]  # (h, w)
        new_shape = self.target_size
        
        # Ratio scale (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Calculate new size after scale
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        
        # Calculate padding
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # Add padding (top, bottom, left, right)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=self.auto_pad_color
        )
        
        return img, r, (dw, dh)
    
    def load_image(self, input_data: Union[str, np.ndarray, Path]) -> np.ndarray:
        """Load image from path or return array directly."""
        if isinstance(input_data, (str, Path)):
            path = str(input_data)
            if path.lower().endswith('.npy'):
                return np.load(path)
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Could not load image: {path}")
            return img
        elif isinstance(input_data, np.ndarray):
            return input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
    
    def __call__(self, input_data: Union[str, np.ndarray, Path]) -> np.ndarray:
        """Preprocess image to model input tensor."""
        # Load image
        img = self.load_image(input_data)
        
        # Letterbox resize
        img, _, _ = self.letterbox(img)
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Normalize with mean/std (ImageNet style if configured)
        if not np.allclose(self.mean, 0) or not np.allclose(self.std, 1):
            img = (img - self.mean) / self.std
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img


def create_preprocessor(config: dict) -> YOLOv5Preprocessor:
    """Factory function to create YOLOv5 preprocessor."""
    return YOLOv5Preprocessor(config)