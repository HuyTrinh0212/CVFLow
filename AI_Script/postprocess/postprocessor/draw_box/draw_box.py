import os
import numpy as np
import json
from AI_Script.postprocess.registry_postprocess import PostProcessorRegistry
from AI_Script.postprocess.base_postprocess import BasePostprocessor
from AI_Script.postprocess.Functions.Adapter_Detection import Adapter
from AI_Script.postprocess.Functions.Boxes_Steps import NMS
from AI_Script.core.utils import check_file, PROJECT_ROOT
import cv2
from datetime import datetime
from AI_Script.postprocess.Functions.Boxes_Steps import unletterbox

@PostProcessorRegistry.register("draw_box")
class DrawBox(BasePostprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        if not self.config.get("output_path") or not str(self.config.get("output_path")).strip():
            raise ValueError("'output_path' is required in config (no default).")
        self.output_path = os.path.abspath(str(self.config.get("output_path")).strip())
        os.makedirs(self.output_path, exist_ok=True)
        self.dataset = str(self.config.get("dataset"))
        self.model_name = str(self.config.get("model_name"))
        self.target_size = tuple(self.config.get("target_size"))
        self.precision_format = str(self.config.get("precision_format"))
        self.conf_threshold = float(self.config.get("conf_threshold"))
        self.iou_threshold = float(self.config.get("iou_threshold"))
        self.transpose_output = bool(self.config.get("transpose_output"))
        self.json_dataset = self.get_json() # to get text label

        self.adapter = Adapter(name_model=self.model_name, transpose=self.transpose_output)

    @staticmethod
    def get_json():
        json_path = os.path.join(PROJECT_ROOT, "configs/datasets.json")
        with open(json_path, "r") as f:
            jsons = json.load(f)
        return jsons

    def postprocess(self, raw_outputs, raw_input=None, save=True):
        # === Load image (cv2) ===
        if check_file(raw_input) == "image_path":
            image = cv2.imread(raw_input)  # BGR
            if image is None:
                raise FileNotFoundError(f"Could not load {raw_input}")
        elif check_file(raw_input) == 'npy_path':
            img_array = np.load(raw_input)
            arr = img_array.astype(np.uint8)
            # if array is RGB, convert to BGR for cv2 display
            if arr.ndim == 3 and arr.shape[2] == 3:
                image = arr[..., ::-1]  # RGB -> BGR
            elif arr.ndim == 2:
                image = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            else:
                image = arr
        elif check_file(raw_input) == "numpy_array":
            image = raw_input.astype(np.uint8)
        else:
            raise ValueError("Format input is invalid")
        # Note: cv2 shape is (h, w, c)
        h, w = image.shape[:2]

        #======
        # from ultralytics.utils import nms, ops
        # import torch
        # predss = torch.from_numpy(raw_outputs[0])
        # predss = predss.transpose(1, 2)
        # preds = nms.non_max_suppression(
        #     predss,
        #     0.6,
        #     0.3,
        #     None,
        #     False,
        #     max_det=100,
        #     nc=0,
        #     end2end=False,
        #     rotated=False,
        #     return_idxs=False,
        # )
        # print(preds[0])
        # print(preds[0].shape)
        # boxes = preds[0][:, 0:4].to(torch.int32).cpu().numpy()
        # scores = preds[0][:, 4].cpu().numpy()
        # class_ids = preds[0][:, 5].to(torch.int32).cpu().numpy()
        #======

        # Load raw outputs
        dicts = self.adapter(raw_outputs)
        boxes = dicts["boxes"]
        conf_scores = dicts["conf"]
        class_ids = dicts["class"]

        mask = conf_scores > self.conf_threshold
        # Remove low confidence
        boxes = boxes[mask]
        scores = conf_scores[mask]
        class_ids = class_ids[mask]

        # Apply NMS
        if len(boxes) > 0:
            keep = NMS(boxes, scores, iou_threshold=self.iou_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]

        # unletterbox
        boxes = unletterbox(boxes_xyxy=boxes, original_shape=(h, w), target_size=self.target_size)

        # Draw
        cv_font = cv2.FONT_HERSHEY_SIMPLEX
        desired_px = 8
        test_size = cv2.getTextSize("Ag", cv_font, 1.0, 1)[0][1]
        font_thickness = 1
        font_scale = (desired_px / test_size) if test_size > 0 else 1.0

        # Color conversions: original PIL used RGB tuples. OpenCV uses BGR.
        rect_color = (255, 0, 0)
        label_color = (0, 255, 0)
        prob_color = (255, 0, 255)

        # Sử dụng `boxes_corrected` đã được tính toán lại
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            # convert coords to int
            x1_i, y1_i, x2_i, y2_i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

            # Draw rectangle (thickness=1)
            cv2.rectangle(image, (x1_i, y1_i), (x2_i, y2_i), rect_color, 1)

            # Prepare label strings
            label = f"Label: {self.json_dataset[self.dataset][str(class_id)]}"
            prob = f"Score: {score:.2f}"

            (label_w, label_h), label_baseline = cv2.getTextSize(label, cv_font, font_scale, font_thickness)
            (prob_w, prob_h), prob_baseline = cv2.getTextSize(prob, cv_font, font_scale, font_thickness)

            label_org = (x1_i + 5, y1_i + 5 + label_h)              # bottom-left for first line
            prob_org = (x1_i + 5, y1_i + 5 + label_h + prob_h + 2)  # second line below first

            cv2.putText(image, label, label_org, cv_font, font_scale, label_color, font_thickness, cv2.LINE_AA)
            cv2.putText(image, prob, prob_org, cv_font, font_scale, prob_color, font_thickness, cv2.LINE_AA)


        # Save and display the result
        if save:
            save_path = os.path.join(self.output_path, f"{self.model_name} | {self.precision_format} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}.jpg")
            cv2.imwrite(save_path, image)
            print(f"Image are saved as {save_path}")
        return image