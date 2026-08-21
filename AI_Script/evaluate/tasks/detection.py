import numpy
from PIL import Image
import numpy as np
from AI_Script.evaluate.base_evaluate import BaseEvaluate
from AI_Script.evaluate.registry_evaluate import EvaluateRegistry
from AI_Script.postprocess.Functions.Adapter_Detection import Adapter
from AI_Script.postprocess.Functions.Boxes_Steps import unletterbox, xywh_to_xyxy
import pickle
import os
import cv2
from AI_Script.core.utils import PROJECT_ROOT
from sklearn.metrics import average_precision_score
import warnings

@EvaluateRegistry.register("object_detection")
class detection_benchmark(BaseEvaluate):
    def __init__(self, config):
        super().__init__(config)
        self.model_name = str(self.config.get("model_name"))
        self.target_size = tuple(self.config.get("target_size"))
        self.iou_threshold = float(self.config.get("iou_threshold"))
        self.conf_threshold = float(self.config.get("conf_threshold"))

        self.adapter = Adapter(name_model=self.model_name)

    def _adapter_boxes(self, predicts, images_path):
        adap = self.adapter(predicts)
        img = cv2.imread(images_path)
        height, width = img.shape[:2]
        # unletterbox
        adap['boxes'] = unletterbox(boxes_xyxy=adap['boxes'], original_shape=(height, width), target_size=self.target_size)
        return adap

    def _load_labels(self, labels_path, images_path):
        labels = np.loadtxt(labels_path)
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)
        classes = labels[:, 0].astype(int)
        boxes = labels[:, 1:]
        # Convert boxes xywh -> xyxy
        boxes = xywh_to_xyxy(boxes=boxes)
        # get real image shape
        img = cv2.imread(images_path)
        height, width = img.shape[:2]
        # scale to real size image
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        label = {
            "boxes": boxes.astype('int'),
            "class": classes
        }
        return label

    def _calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_g, y1_g, x2_g, y2_g = box2

        xi1 = max(x1, x1_g)
        yi1 = max(y1, y1_g)
        xi2 = min(x2, x2_g)
        yi2 = min(y2, y2_g)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_g - x1_g) * (y2_g - y1_g)

        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def _calculate_metrics(self, preds, grt):
        iou_thrs = np.arange(0.5, 1.0, 0.05)  # 0.5 – 0.95 step 0.05
        cls_ids = np.unique(grt['class'])  # Chỉ lấy class từ ground truth
        total_missed = 0  # Tổng số object không được dự đoán

        # Lọc theo ngưỡng conf
        mask = preds['conf'] > self.conf_threshold
        p_boxes = preds['boxes'][mask]
        p_confs = preds['conf'][mask]
        p_clss = preds['class'][mask]

        aps = {t: [] for t in iou_thrs}  # AP cho từng IoU

        for cls_id in cls_ids:
            # GT & pred của class này
            gt_m = grt['class'] == cls_id
            pd_m = p_clss == cls_id
            gt_b = grt['boxes'][gt_m]
            pd_b = p_boxes[pd_m]
            pd_c = p_confs[pd_m]

            n_gt = len(gt_b)
            if n_gt == 0:
                for t in iou_thrs:
                    aps[t].append(0.)
                continue
            if len(pd_b) == 0:
                for t in iou_thrs:
                    aps[t].append(0.)
                total_missed += n_gt  # Tất cả object đều không được dự đoán
                continue

            # Sắp xếp theo conf giảm dần
            order = np.argsort(-pd_c)
            pd_b = pd_b[order]
            pd_c = pd_c[order]

            for t in iou_thrs:
                matched = np.zeros(n_gt, bool)
                tp = np.zeros(min(len(pd_b), n_gt), dtype=float)  # Chỉ xét số box tối đa bằng số gt

                for j, pb in enumerate(pd_b[:n_gt]):  # Giới hạn số box dự đoán
                    best_iou = 0
                    best_idx = -1
                    for k, gb in enumerate(gt_b):
                        if matched[k]:
                            continue
                        iou = self._calculate_iou(pb, gb)
                        if iou > best_iou:
                            best_iou, best_idx = iou, k
                    if best_iou >= t and best_idx >= 0:
                        matched[best_idx] = True
                        tp[j] = 1

                # Tính AP
                if tp.sum() == 0:
                    aps[t].append(0.)
                    continue

                tp_cum = np.cumsum(tp)
                rc = tp_cum / n_gt
                pr = tp_cum / (np.arange(len(tp)) + 1)

                # COCO-style: 101-point interpolation
                recalls = np.linspace(0, 1, 101)
                precisions = np.array([pr[rc >= r].max() if np.any(rc >= r) else 0.
                                       for r in recalls])
                ap = precisions.mean()
                aps[t].append(ap)

            # Đếm số object không được dự đoán cho class này
            total_missed += n_gt - min(len(pd_b), n_gt)  # Số object thiếu so với ground truth

        # mAP
        map50 = np.mean(aps[0.50])
        map5095 = np.mean([np.mean(aps[t]) for t in iou_thrs])
        return {"mAP@0.5": map50, "mAP@[0.5:0.95]": map5095, "total_missed_objects": total_missed}

    def benchmark_single(self, predictions, images_path, labels_path):
        preds = self._adapter_boxes(predictions, images_path)
        labels = self._load_labels(labels_path, images_path)

        benchmark = {
            "image_path": str,
            "metrics": None
        }
        out = self._calculate_metrics(preds=preds, grt=labels)
        # Append
        benchmark["image_path"] = images_path
        benchmark["metrics"] = out
        return benchmark

    def benchmark_toltal(self, benchmark_dict):
        benchmark = {
            "avg_mAP@0.5": float,
            "avg_mAP@[0.5:0.95]": float,
        }

        # average
        map_05 = []
        map_5095 = []
        for img_id, img_data in benchmark_dict["images"].items():
            map_05.append(img_data['metrics']['mAP@0.5'])
            map_5095.append(img_data['metrics']['mAP@[0.5:0.95]'])
        benchmark["avg_mAP@0.5"] = sum(map_05) / len(map_05) if map_05 else 0
        benchmark["avg_mAP@[0.5:0.95]"] = sum(map_5095) / len(map_5095) if map_5095 else 0

        return benchmark