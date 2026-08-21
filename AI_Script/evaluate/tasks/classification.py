from PIL import Image
import numpy as np
from AI_Script.evaluate.base_evaluate import BaseEvaluate
from AI_Script.evaluate.registry_evaluate import EvaluateRegistry
import pickle
import os
from AI_Script.core.utils import PROJECT_ROOT
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    matthews_corrcoef
)

@EvaluateRegistry.register("classification")
class classification_benchmark(BaseEvaluate):
    def __init__(self, config):
        super().__init__(config)

    def _get_max_index(self, raw_output):
        return np.argmax(raw_output[0])

    def _get_labels(self, label_path):
        with open(label_path, "r") as f:
            return str(f.read().strip())

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def benchmark_single(self, predictitons, images_path, labels_path):
        preds = int(self._get_max_index(predictitons))
        labels = int(self._get_labels(labels_path))

        benchmark = {
            "image_path": images_path,
            "metrics": {
                "correct": bool(preds == labels),
                "predict": preds,
                "ground_truth": labels,
                # "confidence_score": float(self._softmax(predictitons[0])[0][preds]),
            }
        }
        return benchmark

    def benchmark_toltal(self, benchmark_dict):
        preds = [data["metrics"]["predict"] for key, data in benchmark_dict["images"].items()]
        labels = [data["metrics"]["ground_truth"] for key, data in benchmark_dict["images"].items()]

        # Check length
        if len(preds) != len(labels):
            raise ValueError(f"Shape mismatch: preds.shape={len(preds)}, labels.shape={len(labels)}")

        # Accuracy
        acc = float(accuracy_score(labels, preds))

        # Macro / micro / weighted precision/recall/f1
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
        prec_weight, rec_weight, f1_weight, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

        # Per-class report & confusion matrix
        per_class = classification_report(labels, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds).tolist()

        # Additional metrics that do not require probabilities
        mcc = float(matthews_corrcoef(labels, preds))

        benchmark = {
            "accuracy": acc,
            "macro_avg": {
                "precision": float(prec_macro),
                "recall": float(rec_macro),
                "f1-score": float(f1_macro)
            },
            "micro_avg": {
                "precision": float(prec_micro),
                "recall": float(rec_micro),
                "f1-score": float(f1_micro)
            },
            "weighted_avg": {
                "precision": float(prec_weight),
                "recall": float(rec_weight),
                "f1-score": float(f1_weight)
            },
            "mcc": mcc,
            "Per_class": per_class,
            "Confusion_matrix": cm,
        }
        return benchmark