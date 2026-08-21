"""LeNet5 preprocessing - grayscale MNIST-style preprocessing."""
import cv2
import numpy as np
from typing import Tuple, Union, Any, Optional
from pathlib import Path


class LeNet5Preprocessor:
    """LeNet5 preprocessing: grayscale, resize to 32x32, normalize."""
    
    def __init__(self, config: dict):
        self.config = config
        self.target_size = tuple(config.get("target_size", (32, 32)))
        mean = config.get("mean", 0)
        std = config.get("std", 1)
        self.mean = np.array(mean, dtype=np.float32) if np.isscalar(mean) else np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32) if np.isscalar(std) else np.array(std, dtype=np.float32)
    
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
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to target size
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        # Add channel dimension (H, W) -> (H, W, 1)
        img = img[..., np.newaxis]
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Normalize with mean/std
        img = (img - self.mean) / self.std
        
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img


def create_preprocessor(config: dict) -> LeNet5Preprocessor:
    """Factory function to create LeNet5 preprocessor."""
    return LeNet5Preprocessor(config)