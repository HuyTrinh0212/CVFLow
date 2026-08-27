"""LeNet5 output adapter - converts raw ONNX output to canonical ClassificationOutput."""
import numpy as np
from typing import Any, Dict, Optional
from scipy.special import softmax

from cvf.core.contracts.runtime.classification import ClassificationOutput


class LeNet5ClassificationAdapter:
    """Adapter for LeNet5 raw output -> canonical ClassificationOutput."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        # Support task_class_names forwarded by AdapterStage (task ↔ model separation)
        if not self.config.get("class_names") and self.config.get("task_class_names"):
            self.config["class_names"] = self.config["task_class_names"]
    
    def adapt(self, raw_output: Any) -> ClassificationOutput:
        """
        Convert raw LeNet5 output to canonical ClassificationOutput.
        
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
        # Cast FP16 to FP32 to avoid softmax precision issues
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


def create_adapter(config: Optional[Dict] = None) -> LeNet5ClassificationAdapter:
    """Factory function to create LeNet5 classification adapter."""
    return LeNet5ClassificationAdapter(config)