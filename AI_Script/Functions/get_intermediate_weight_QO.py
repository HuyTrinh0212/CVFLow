import onnx
import numpy as np
from pathlib import Path
import os
import pickle
import json

parent_path = Path(__file__)
head_path = parent_path.parent

def to_serializable(obj):
    """Recursively convert NumPy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj

def dump_all_weights(model_path):
    model = onnx.load(model_path)
    graph = model.graph

    # print(f"Model: {model_path}")
    out = {}
    for initializer in graph.initializer:
        arr = onnx.numpy_helper.to_array(initializer)
        name = initializer.name
        # print(f"\n\n\n\n\n=== Tensor: {name} ===")
        # print(f"Shape: {arr.shape}")
        # print(f"Dtype: {arr.dtype}")
        # print(f"Values: {arr}")
        out[str(name)] = {
            "name": str(name),
            "shape": arr.shape,
            "dtype": str(arr.dtype),
            "values": arr
        }
    # output_pickle = os.path.join(head_path, "weights/weight_QO.pkl")
    output_json = os.path.join(head_path, "weights", "intermediate_weight_lenet5_47_labels_int8.json")
    serializable_out = to_serializable(out)
    # Save JSON
    with open(output_json, "w") as f:
        json.dump(serializable_out, f, indent=2)
    return serializable_out

# Example usage
file_path = "/home/bht/CODE/ai-script/pretrained_models/INT8/lenet5_47labels_int8.onnx"
out = dump_all_weights(file_path)

# Load
def load_weights_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    restored = {}
    for name, info in data.items():
        restored[name] = {
            "name": info["name"],
            "shape": info["shape"],
            "dtype": info["dtype"],
            "values": info["values"]
        }

    print(f"✅ Loaded weights from: {json_path}")
    return restored

save_path = "weights/intermediate_weight_lenet5_47_labels_int8.json"
weights = load_weights_from_json(save_path)
for name, tensor in weights.items():
    print(name, tensor["values"])
