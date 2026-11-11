import json
import numpy as np
from typing import Any, Dict, Tuple

def to_numpy(x: Any) -> np.ndarray:
    """Convert list to numpy array (float)"""
    arr = np.array(x, dtype=float)
    return arr

def mse_between_arrays(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate MSE between 2 numpy arrays."""
    if a.shape != b.shape:
        a_flat = a.ravel()
        b_flat = b.ravel()
        if a_flat.size != b_flat.size:
            raise ValueError(f"Can't compare, A & B differ shape: {a.shape} vs {b.shape}")
        diff = a_flat - b_flat
    else:
        diff = (a - b).ravel()
    return float(np.mean(diff ** 2))

def Compare_Ouput_Intermediate(json_a, json_b, require_same_keys: bool = True) -> Tuple[Dict[str, float], float]:
    keys_a = set(json_a.keys())
    keys_b = set(json_b.keys())

    if require_same_keys and keys_a != keys_b:
        missing_in_b = keys_a - keys_b
        missing_in_a = keys_b - keys_a
        raise KeyError(f"Differ key. Lake B: {missing_in_b}. Lake A: {missing_in_a}")

    keys = sorted(keys_a & keys_b)

    per_layer_mse: Dict[str, float] = {}
    total_se = 0.0
    total_count = 0

    for k in keys:
        a_arr = to_numpy(json_a[k]["values"])
        b_arr = to_numpy(json_b[k]["values"])

        if a_arr.size != b_arr.size:
            raise ValueError(f"Layer '{k}' have differ shape: {a_arr.shape} vs {b_arr.shape}")

        diff = (a_arr.ravel() - b_arr.ravel())
        se = float((diff ** 2).sum())   # sum squared errors
        count = diff.size

        layer_mse = se / count if count > 0 else float('nan')
        per_layer_mse[k] = layer_mse

        total_se += se
        total_count += count

    overall_mse = total_se / total_count if total_count > 0 else float('nan')
    return per_layer_mse, overall_mse


# if __name__ == "__main__":
#
#     json_A = {"layer 0": [0.01, 0.02, 0.03], "layer 1": [[0.02,0.01], [0.05,0.03]]}
#     json_B = {"layer 0": [0.01, 0.02, 0.03], "layer 1": [[0.02,0.04], [0.01,0.03]]}
#
#     per_layer, overall = Compare_Ouput_Intermediate(json_A, json_B)
#     print("MSE by layer:")
#     for k, v in per_layer.items():
#         print(f"  {k}: {v:.12f}")
#     print(f"MSE overall: {overall:.12f}")
