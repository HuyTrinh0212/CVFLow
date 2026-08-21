import numpy as np
import torch
from torchvision.ops import nms

# Convert bounding xywh to xyxy
def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy_boxes

def NMS(boxes, scores, iou_threshold=0.3):
    boxes = torch.tensor(boxes, dtype=torch.float32)
    scores = torch.tensor(scores, dtype=torch.float32)
    keep = nms(boxes, scores, iou_threshold)
    return keep.numpy()

def unletterbox(boxes_xyxy, original_shape, target_size):
    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.size == 0:
        return np.zeros((0, 4), dtype=np.int32)

    # normalize shape
    if boxes.ndim == 1 and boxes.shape[0] == 4:
        boxes = boxes.reshape(1, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N,4) or (4,)")

    orig_h, orig_w = original_shape
    tgt_h, tgt_w = target_size

    # nếu boxes normalized (0..1), chuyển về pixel space của ảnh letterbox (tgt)
    if boxes.max() <= 1.0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * tgt_w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * tgt_h

    # tỉ lệ scale theo YOLO
    r = min(tgt_h / orig_h, tgt_w / orig_w)

    # new_unpad = round(orig * r)  (lưu ý rounding trước)
    new_unpad_w = int(round(orig_w * r))
    new_unpad_h = int(round(orig_h * r))

    # tổng padding (pixel)
    pad_w = tgt_w - new_unpad_w
    pad_h = tgt_h - new_unpad_h

    # padding nửa (float), Ultralytics tính left/top bằng round(dw - 0.1)
    pad_w_half = pad_w / 2.0
    pad_h_half = pad_h / 2.0
    left = int(round(pad_w_half - 0.1))
    top = int(round(pad_h_half - 0.1))

    # gỡ padding
    boxes_out = boxes.copy().astype(np.float64)
    boxes_out[:, [0, 2]] -= left
    boxes_out[:, [1, 3]] -= top
    boxes_out[:, [0, 2]] /= r
    boxes_out[:, [1, 3]] /= r

    # clip
    boxes_out[:, [0, 2]] = boxes_out[:, [0, 2]].clip(0, orig_w - 1)
    boxes_out[:, [1, 3]] = boxes_out[:, [1, 3]].clip(0, orig_h - 1)

    return boxes_out.round().astype(np.int32)





