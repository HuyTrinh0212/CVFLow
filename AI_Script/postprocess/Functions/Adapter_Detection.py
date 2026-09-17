import numpy as np
from AI_Script.postprocess.Functions.Boxes_Steps import xywh_to_xyxy

class Adapter:
    def __init__(self, name_model: str, transpose=False):
        self.name_model = name_model.lower()
        self.transpose = transpose
        self._adapters = {
            'yolov5': self._adapter_yolov5,
            'detr': self._adapter_detr,
        }

        if self.name_model not in self._adapters:
            raise ValueError(f"Model '{name_model}' is not supported. "
                             f"Supported models are: {list(self._adapters.keys())}")

        self.adapter_function = self._adapters[self.name_model]

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def __call__(self, raw_outputs):
        return self.adapter_function(raw_outputs)

    def _adapter_yolov5(self, raw_outputs):
        if self.transpose == True:
            outputs = np.transpose(raw_outputs[0], (0, 2, 1))
        else:
            outputs = raw_outputs[0]

        prediction = outputs.squeeze(0)

        # Boxes: (x_center, y_center, width, height)
        boxes = prediction[:, :4]
        boxes_xyxy = xywh_to_xyxy(boxes)

        objectness = prediction[:, 4:5]
        class_scores = prediction[:, 5:]

        max_class_scores = np.max(class_scores, axis=1, keepdims=True)
        class_ids = np.argmax(class_scores, axis=1, keepdims=True)
        conf = objectness * max_class_scores

        output_dict = {
            'boxes': boxes_xyxy,
            'conf': conf.squeeze(1),
            'class': class_ids.squeeze(1)
        }
        return output_dict

    def _adapter_detr(self, raw_outputs):
        # RT-DETR: [1, 300, 84] = [cx, cy, w, h (normalized 0-1), c0..c79]
        # No objectness score (unlike YOLOv5)
        outputs = raw_outputs[0]
        prediction = outputs.squeeze(0)

        boxes = prediction[:, :4]
        boxes_xyxy = xywh_to_xyxy(boxes)

        class_scores = prediction[:, 4:]
        conf = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        output_dict = {
            'boxes': boxes_xyxy,
            'conf': conf,
            'class': class_ids
        }
        return output_dict