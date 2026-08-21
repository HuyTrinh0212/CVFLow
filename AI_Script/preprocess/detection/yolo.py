# preprocess/processors/yolo.py
from AI_Script.preprocess.base_preprocess import BasePreprocessor
from AI_Script.preprocess.registry_preprocess import PreprocessRegistry
from AI_Script.core.utils import check_file
import cv2
import numpy as np

@PreprocessRegistry.register("yolo")
@PreprocessRegistry.register("yolov5")
class YOLOPreprocessor(BasePreprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.target_size = tuple(self.config.get("target_size", (640, 640)))
        self.mean = np.array(self.config.get("mean", [0.0, 0.0, 0.0]), dtype=np.float32)
        self.std = np.array(self.config.get("std", [1.0, 1.0, 1.0]), dtype=np.float32)
        # color padding
        self.auto_pad_color = (114, 114, 114)

    def letterbox(self, img: np.ndarray):
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
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=self.auto_pad_color)

        return img, r, (dw, dh)

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

        # Resize
        processed, ratio, (dw, dh) = self.letterbox(processed)
        # BGR -> RGB
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        # Normalize to [0,1]
        processed = processed.astype(np.float32) / 255.0
        # Normalize ImageNet
        # processed = (processed - self.mean) / self.std
        # HWC -> CHW
        processed = np.transpose(processed, (2, 0, 1))
        # Add batch dim
        processed = np.expand_dims(processed, axis=0)
        return processed


# Show image are pre-procesed
def show_img(intput):
    # 1. Loại bỏ batch dimension
    display_image = intput.squeeze()
    # 2. CHW -> HWC
    display_image = np.transpose(display_image, (1, 2, 0))
    # 3. Hủy chuẩn hóa (đưa pixel về [0, 1])
    display_image = display_image * 1.0 + 0.0
    # 4. Giới hạn giá trị trong [0, 1] và chuyển về [0, 255]
    display_image = np.clip(display_image, 0, 1) * 255.0
    # 5. Chuyển dtype về uint8
    display_image = display_image.astype(np.uint8)
    # 6. RGB -> BGR để hiển thị với cv2
    display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

    # Hiển thị ảnh
    cv2.imshow("Processed Image", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()