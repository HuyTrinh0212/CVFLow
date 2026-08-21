"""Classification task - schema, postprocess, metrics."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
from scipy.special import softmax

from cvf.core.contracts.runtime.classification import ClassificationOutput
from cvf.core.contracts.config.task import TaskConfig


@dataclass
class ClassificationConfig:
    """Classification-specific configuration."""
    top_k: int = 5
    threshold: float = 0.0
    class_names: Optional[List[str]] = None


class ClassificationPostprocessor:
    """Classification postprocessing: softmax, top-k, thresholding."""
    
    def __init__(self, config: Optional[ClassificationConfig] = None):
        self.config = config or ClassificationConfig()
    
    def __call__(self, output: ClassificationOutput, task_config: TaskConfig) -> ClassificationOutput:
        """Apply postprocessing to classification output."""
        # Get probabilities (apply softmax if needed)
        probs = output.probabilities
        if np.any(probs < 0) or np.any(probs > 1) or abs(probs.sum() - 1.0) > 1e-5:
            probs = softmax(output.logits)
        
        # Get top-k
        top_k = min(self.config.top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_scores = probs[top_indices]
        top_classes = top_indices
        
        # Apply threshold
        mask = top_scores >= self.config.threshold
        top_scores = top_scores[mask]
        top_classes = top_classes[mask]
        
        return ClassificationOutput(
            logits=output.logits,
            probabilities=probs,
            class_ids=top_classes,
            scores=top_scores,
            class_names=self.config.class_names,
        )


class ClassificationMetrics:
    """Classification evaluation metrics: accuracy, F1, top-k."""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
    
    def compute(self, preds: List[ClassificationOutput], targets: List[int]) -> Dict[str, float]:
        """Compute classification metrics."""
        if not preds or not targets:
            return {}
        
        n = len(preds)
        top1_correct = 0
        topk_correct = 0
        
        for pred, target in zip(preds, targets):
            if pred.top1_id == target:
                top1_correct += 1
            if target in pred.class_ids:
                topk_correct += 1
        
        accuracy = top1_correct / n
        topk_accuracy = topk_correct / n
        
        return {
            "accuracy": accuracy,
            f"top{self.top_k}_accuracy": topk_accuracy,
            "total_samples": n,
            "correct_top1": top1_correct,
            f"correct_top{self.top_k}": topk_correct,
        }


def create_classification_postprocess(config: Optional[Dict] = None) -> ClassificationPostprocessor:
    """Factory for classification postprocessor."""
    if config:
        cls_config_dict = {k: v for k, v in config.items() 
                          if k in ['top_k', 'threshold', 'class_names']}
        cls_config = ClassificationConfig(**cls_config_dict)
    else:
        cls_config = ClassificationConfig()
    return ClassificationPostprocessor(cls_config)


def create_classification_metrics(config: Optional[Dict] = None) -> ClassificationMetrics:
    """Factory for classification metrics."""
    top_k = config.get("top_k", 5) if config else 5
    return ClassificationMetrics(top_k=top_k)