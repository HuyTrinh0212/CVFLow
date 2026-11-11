import numpy as np
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# check file
def check_file(path):
    if isinstance(path, np.ndarray):
        return 'numpy_array'
    if isinstance(path, str):
        if os.path.isdir(path):
            return 'folder_path'
        if not os.path.exists(path):
            return 'invalid_path'
        if path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return 'video_path'
        if path.lower().endswith(('.npy')):
            return 'npy_path'
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            return 'image_path'
    return 'unknown_type'

