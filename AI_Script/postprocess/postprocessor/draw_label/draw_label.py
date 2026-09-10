import os
import os.path

import numpy as np
from AI_Script.postprocess.registry_postprocess import PostProcessorRegistry
from AI_Script.postprocess.base_postprocess import BasePostprocessor
import cv2
from AI_Script.core.utils import check_file
from PIL import Image
from datetime import datetime
from AI_Script.postprocess.postprocessor.get_labels.get_labels import GetLabel
from AI_Script.postprocess.postprocessor.decode_crnn.decode_crnn import DecodeCRNN

@PostProcessorRegistry.register("draw_label")
class DrawLabel(BasePostprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        if not self.config.get("output_path") or not str(self.config.get("output_path")).strip():
            raise ValueError("'output_path' is required in config (no default).")
        self.output_path = os.path.abspath(str(self.config.get("output_path")).strip())
        os.makedirs(self.output_path, exist_ok=True)
        self.name_model = str(self.config.get("model_name"))
        self.task = str(self.config.get("task"))
        self._adapters = {
            'classification': self._classification,
            'htr': self._htr
        }
        if self.task not in self._adapters:
            raise ValueError(f"Task '{self.task}' is not supported. "
                             f"Supported tasks are: {list(self._adapters.keys())}")

        self.adapter_function = self._adapters[self.task]

    def postprocess(self, raw_output, raw_input):
        return self.adapter_function(raw_output, raw_input)

    def draw(self, text, raw_input):
        if check_file(raw_input) == "image_path":
            img = cv2.imread(raw_input)
            if img is None:
                raise FileNotFoundError(f"Could not load {raw_input}")
        else:
            raise ValueError("Format input is invalid")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        font_thickness = 3
        text_color = (0, 255, 0)
        bg_color = (0, 0, 0)  # background

        # Resize
        img = cv2.resize(img, (1200, int(1200 * img.shape[0] / img.shape[1])))

        # === GET TEXT SIZE ===
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        padding = text_h + 30

        # === CREATE NEW CANVAS WITH EXTRA TOP SPACE ===
        h, w, c = img.shape
        new_h = h + padding
        canvas = np.zeros((new_h, w, c), dtype=np.uint8)

        # Fill top region background
        canvas[:padding] = bg_color

        # Copy original image below top region
        canvas[padding:] = img

        # === CALCULATE TEXT POSITION ===
        x = (w - text_w) // 2
        y = (padding + text_h) // 2

        # === DRAW TEXT ===
        cv2.putText(canvas, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Save result
        save_path = os.path.join(self.output_path, f"{self.name_model} {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}.jpg")
        cv2.imwrite(save_path, canvas)
        print(f"Output are saved in {save_path}")

    def _classification(self, raw_output, raw_input):
        # Postprocess to get text label
        get_label = GetLabel(self.config)
        text = f"Predicted: {get_label(raw_output)}"

        # Draw
        self.draw(text, raw_input)


    def _htr(self, raw_output, raw_input):
        # Postprocess to get text label
        get_label = DecodeCRNN(self.config)
        text = f"Predicted: {get_label(raw_output)}"

        # Draw
        self.draw(text, raw_input)
