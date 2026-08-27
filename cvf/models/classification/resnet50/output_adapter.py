"""ResNet50 output adapter - converts raw ONNX output to canonical ClassificationOutput."""
import numpy as np
from typing import Any, Dict, Optional
from scipy.special import softmax

from cvf.core.contracts.runtime.classification import ClassificationOutput


class ResNet50ClassificationAdapter:
    """Adapter for ResNet50 raw output -> canonical ClassificationOutput."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Resolve class_names from merged model+task config (AdapterStage forwards task_class_names/dataset)
        # Priority: model.class_names > task.task_class_names > dataset registry
        if not self.config.get("class_names") and self.config.get("task_class_names"):
            self.config["class_names"] = self.config["task_class_names"]
        # Dataset registry fallback (correct order from main: AI_Script/configs/datasets.json eurosat)
        if not self.config.get("class_names") and self.config.get("dataset") == "eurosat":
            self.config["class_names"] = [
                "Forest", "River", "Highway", "AnnualCrop", "SeaLake",
                "HerbaceousVegetation", "Industrial", "Residential", "PermanentCrop", "Pasture"
            ]
    
    def adapt(self, raw_output: Any) -> ClassificationOutput:
        """
        Convert raw ResNet50 output to canonical ClassificationOutput.
        
        Args:
            raw_output: List of numpy arrays from ONNX Runtime, typically [1, num_classes]
            
        Returns:
            ClassificationOutput with logits, probabilities, top-k class_ids and scores
        """
        # Handle different output formats
        if isinstance(raw_output, (list, tuple)):
            outputs = raw_output[0]
        else:
            outputs = raw_output
        
        # Squeeze batch dimension
        logits = outputs.squeeze(0)
        if isinstance(logits, np.ndarray) and logits.dtype == np.float16:
            logits = logits.astype(np.float32)
        
        # Apply softmax to get probabilities
        probs = softmax(logits)
        
        # Get top-k
        top_k = self.config.get("top_k", 5)
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_scores = probs[top_indices]
        top_classes = top_indices
        
        return ClassificationOutput(
            logits=logits,
            probabilities=probs,
            class_ids=top_classes.astype(np.int32),
            scores=top_scores,
            class_names=self.config.get("class_names"),
        )


def create_adapter(config: Optional[Dict] = None) -> ResNet50ClassificationAdapter:
    """Factory function to create ResNet50 classification adapter."""
    return ResNet50ClassificationAdapter(config)