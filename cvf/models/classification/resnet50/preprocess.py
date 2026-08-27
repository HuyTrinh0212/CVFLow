"""ResNet50 preprocessing - standard ImageNet preprocessing."""
import cv2
import numpy as np
from typing import Tuple, Union, Any, Optional
from pathlib import Path


class ResNet50Preprocessor:
    """ResNet50 preprocessing: EuroSAT style (matches main: AI_Script/preprocess/classification/resnet.py)."""
    
    def __init__(self, config: dict):
        self.config = config
        self.target_size = tuple(config.get("target_size", (224, 224)))
        # Correct EuroSAT mean/std from main branch (not ImageNet)
        self.mean = np.array(config.get("mean", [0.3445, 0.3803, 0.4077]), dtype=np.float32)
        self.std = np.array(config.get("std", [0.0915, 0.0652, 0.0553]), dtype=np.float32)
    
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
        
        # Resize to target size
        img = cv2.resize(img, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Normalize with EuroSAT mean/std (main branch)
        img = (img - self.mean) / self.std
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img


def create_preprocessor(config: dict) -> ResNet50Preprocessor:
    """Factory function to create ResNet50 preprocessor."""
    return ResNet50Preprocessor(config)