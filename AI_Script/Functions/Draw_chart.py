import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- User input: set these paths ----
json_path_a = "/home/bht/CODE/xaip/AI_Script/outputs/evaluate - lenet5 - int8 - RTL - 2025-11-02 10:56:00.json"
json_path_b = "/home/bht/CODE/xaip/AI_Script/outputs/evaluate - lenet5 - int8 - CPU - 2025-11-02 10:52:41.json"
# ------------------------------------

OUT_DIR = Path("/home/bht/CODE/xaip/AI_Script/outputs/charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_report(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_numeric_size(size_str: str) -> float:
    """
    Parse strings like "0.07 MB" or "1024 KB" into MB (float).
    If input already numeric string, try to convert.
    """
    if size_str is None:
        return np.nan
    if isinstance(size_str, (int, float)):
        return float(size_str)
    s = str(size_str).strip()
    try:
        if "mb" in s.lower():
            return float(s.split()[0])
        if "kb" in s.lower():
            return float(s.split()[0]) / 1024.0
        if "b" == s.lower()[-1]:
            # naive fallback
            return float(s.split()[0]) / (1024.0 * 1024.0)
        # fallback numeric
        return float(s)
    except Exception:
        return np.nan


def extract_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    mi = report.get("Model_Information", {})
    perf = report.get("Performance", {})
    eval_ = report.get("Evaluate", {}).get("overall_metrics", {})

    model_size_mb = parse_numeric_size(mi.get("model_size_mb", None))
    total_time_s = perf.get("total_time_s", np.nan)
    throughput_qps = perf.get("throughput_qps", np.nan)

    accuracy = eval_.get("accuracy", np.nan)
    mcc = eval_.get("mcc", np.nan)

    # macro/micro/weighted group
    macro = eval_.get("macro_avg", {})
    micro = eval_.get("micro_avg", {})
    weighted = eval_.get("weighted_avg", {})

    summary = {
        "model_name": mi.get("model_name", "model"),
        "run_mode": mi.get("run_mode", ""),
        "precision_format": mi.get("precision_format", ""),
        "model_size_mb": model_size_mb,
        "total_time_s": total_time_s,
        "throughput_qps": throughput_qps,
        "accuracy": accuracy,
        "mcc": mcc,
        "macro": macro,
        "micro": micro,
        "weighted": weighted,
        # include per-class dictionary if present
        "per_class": eval_.get("Per_class", {}),
        "confusion_matrix": eval_.get("Confusion_matrix", None),
    }
    return summary


def plot_summary_bar(a: Dict[str, Any], b: Dict[str, Any], out_path: Path):
    """
    Plot summary metrics side-by-side for the two models:
    - model size (MB), total_time_s, throughput_qps, accuracy, mcc
    """
    labels = ["Model size (MB)", "Total time (s)", "Throughput (QPS)", "Accuracy", "MCC"]
    a_vals = [
        a["model_size_mb"],
        a["total_time_s"],
        a["throughput_qps"],
        a["accuracy"],
        a["mcc"],
    ]
    b_vals = [
        b["model_size_mb"],
        b["total_time_s"],
        b["throughput_qps"],
        b["accuracy"],
        b["mcc"],
    ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, a_vals, width, label=a["model_name"])
    rects2 = ax.bar(x + width/2, b_vals, width, label=b["model_name"])

    ax.set_ylabel("Value")
    ax.set_title(f"Model summary comparison: {a['model_name']} vs {b['model_name']}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="upper right", frameon=True)

    # annotate bar values
    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.3g}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_prf_grouped(a: Dict[str, Any], b: Dict[str, Any], out_path: Path):
    """
    Plot grouped bars for precision / recall / f1 for macro, micro, weighted.
    Robust to missing keys and mismatched lengths.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    categories = ["macro", "micro", "weighted"]
    metrics = ["precision", "recall", "f1-score"]

    # helper to safely fetch metric value (float) or np.nan
    def safe_get_metric(model_summary, cat, metric):
        cat_dict = model_summary.get(cat, {}) or {}
        # Some JSONs might use slightly different keys like "macro_avg" etc.
        # try fallback keys
        if not cat_dict:
            fallback_keys = {
                "macro": ["macro_avg", "macro_avg", "macro"],
                "micro": ["micro_avg", "micro"],
                "weighted": ["weighted_avg", "weighted"]
            }
            for fk in fallback_keys.get(cat, []):
                cat_dict = model_summary.get(fk, {}) or {}
                if cat_dict:
                    break
        try:
            val = cat_dict.get(metric, np.nan)
            return float(val) if val is not None else np.nan
        except Exception:
            return np.nan

    # Build arrays for each metric
    for metric in metrics:
        x = np.arange(len(categories))
        a_vals = [safe_get_metric(a, cat, metric) for cat in categories]
        b_vals = [safe_get_metric(b, cat, metric) for cat in categories]

        # debug guard: lengths must match x
        if len(a_vals) != len(x) or len(b_vals) != len(x):
            print("DEBUG: mismatch lengths", len(a_vals), len(b_vals), "expected", len(x))
            # pad or trim to len(x)
            def fix_len(arr):
                arr = list(arr)
                if len(arr) < len(x):
                    arr += [np.nan] * (len(x) - len(arr))
                if len(arr) > len(x):
                    arr = arr[:len(x)]
                return arr
            a_vals = fix_len(a_vals)
            b_vals = fix_len(b_vals)

        fig, ax = plt.subplots(figsize=(8, 4))
        width = 0.35

        rects1 = ax.bar(x - width/2, a_vals, width, label=a.get("model_name", "modelA"))
        rects2 = ax.bar(x + width/2, b_vals, width, label=b.get("model_name", "modelB"))

        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} by averaging method")
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in categories])
        ax.set_ylim(0.0, 1.05)  # PRF are in [0,1]
        ax.legend(loc="upper right", frameon=True)

        # annotate bar values (handles nan)
        def annotate_rects(rects):
            for rect in rects:
                h = rect.get_height()
                if np.isnan(h):
                    txt = "n/a"
                    va = "bottom"
                    y = 0
                else:
                    txt = f"{h:.3f}"
                    va = "bottom"
                    y = h
                ax.annotate(txt,
                            xy=(rect.get_x() + rect.get_width() / 2, y),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va=va, fontsize=8)

        annotate_rects(rects1)
        annotate_rects(rects2)

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        # save per-metric file
        out_file = out_path.parent / f"{out_path.stem}_{metric.replace(' ', '_')}.png"
        fig.savefig(out_file, dpi=200)
        plt.close(fig)


def plot_per_class_comparison(a: Dict[str, Any], b: Dict[str, Any], out_path: Path):
    """
    Plot per-class f1 (or precision/recall) for both models as grouped bars.
    """
    # per_class is a dict keyed by class label strings; some keys include "accuracy" or "macro avg"—filter numeric keys only
    def get_ordered_per_class(pc_dict):
        numeric_keys = [k for k in pc_dict.keys() if k.isdigit()]
        numeric_keys_sorted = sorted(numeric_keys, key=lambda x: int(x))
        classes = numeric_keys_sorted
        f1s = [pc_dict[k].get("f1-score", np.nan) for k in classes]
        precisions = [pc_dict[k].get("precision", np.nan) for k in classes]
        recalls = [pc_dict[k].get("recall", np.nan) for k in classes]
        return classes, precisions, recalls, f1s

    a_pc = a.get("per_class", {})
    b_pc = b.get("per_class", {})

    classes_a, a_prec, a_rec, a_f1 = get_ordered_per_class(a_pc)
    classes_b, b_prec, b_rec, b_f1 = get_ordered_per_class(b_pc)

    # Use intersection of classes to compare
    classes = [c for c in classes_a if c in classes_b]
    if not classes:
        return  # nothing to plot

    a_f1 = [a_pc[c].get("f1-score", np.nan) for c in classes]
    b_f1 = [b_pc[c].get("f1-score", np.nan) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(classes)*0.5 + 2), 4))
    rects1 = ax.bar(x - width/2, a_f1, width, label=a["model_name"])
    rects2 = ax.bar(x + width/2, b_f1, width, label=b["model_name"])

    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1-score comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend(loc="upper right")

    for rect in rects1 + rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrices(a: Dict[str, Any], b: Dict[str, Any], out_path: Path):
    """
    Plot confusion matrices side-by-side (improved).
    Uses constrained_layout to avoid tight_layout warnings and adds better annotations,
    colorbar, axis labels, and optional class tick labels (if classes available).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors

    cm_a = a.get("confusion_matrix")
    cm_b = b.get("confusion_matrix")
    if cm_a is None or cm_b is None:
        return

    cm_a = np.array(cm_a)
    cm_b = np.array(cm_b)

    # ensure same square shape
    n = max(cm_a.shape[0], cm_b.shape[0])

    def pad_cm(cm, n):
        out = np.zeros((n, n), dtype=int)
        out[:cm.shape[0], :cm.shape[1]] = cm
        return out

    cm_a = pad_cm(cm_a, n)
    cm_b = pad_cm(cm_b, n)

    # optional: try to derive class labels from per_class keys if present
    classes = None
    a_pc = a.get("per_class") or {}
    if a_pc:
        numeric_keys = sorted([k for k in a_pc.keys() if k.isdigit()], key=lambda x: int(x))
        if len(numeric_keys) == n:
            classes = numeric_keys

    # Choose colormap and normalization (shared)
    cmap = plt.cm.Greys_r  # good for counts; change if you prefer
    vmin = min(cm_a.min(), cm_b.min())
    vmax = max(cm_a.max(), cm_b.max())
    # avoid zero-range normalization
    if vmax == vmin:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(cm_a, aspect="equal", cmap=cmap, norm=norm)
    axes[0].set_title(f"{a.get('model_name', 'Model A')} — Confusion Matrix", fontsize=10)
    axes[0].set_xlabel("Predicted", fontsize=9)
    axes[0].set_ylabel("Actual", fontsize=9)

    im1 = axes[1].imshow(cm_b, aspect="equal", cmap=cmap, norm=norm)
    axes[1].set_title(f"{b.get('model_name', 'Model B')} — Confusion Matrix", fontsize=10)
    axes[1].set_xlabel("Predicted", fontsize=9)
    axes[1].set_ylabel("Actual", fontsize=9)

    # ticks and optional class labels
    ticks = np.arange(n)
    axes[0].set_xticks(ticks)
    axes[0].set_yticks(ticks)
    axes[1].set_xticks(ticks)
    axes[1].set_yticks(ticks)

    if classes is not None:
        # use provided class labels (strings)
        axes[0].set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        axes[0].set_yticklabels(classes, fontsize=7)
        axes[1].set_xticklabels(classes, rotation=45, ha="right", fontsize=7)
        axes[1].set_yticklabels(classes, fontsize=7)
    else:
        # numeric labels
        axes[0].set_xticklabels(ticks, fontsize=7)
        axes[0].set_yticklabels(ticks, fontsize=7)
        axes[1].set_xticklabels(ticks, fontsize=7)
        axes[1].set_yticklabels(ticks, fontsize=7)

    # annotate counts: adjust fontsize relative to figure size / n
    annot_fontsize = max(6, int(80 / max(1, n)))  # simple heuristic
    for ax, cm in zip(axes, (cm_a, cm_b)):
        for i in range(n):
            for j in range(n):
                count = int(cm[i, j])
                ax.text(j, i, str(count),
                        ha="center", va="center",
                        fontsize=annot_fontsize,
                        color="black" if norm(count) < 0.6 else "white")  # contrast

    # shared colorbar (use the second image mappable)
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.7, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Count", fontsize=9)

    # save and close
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(json_a: str, json_b: str):
    r_a = load_report(json_a)
    r_b = load_report(json_b)

    s_a = extract_summary(r_a)
    s_b = extract_summary(r_b)

    # attach model_name field inside summary for convenience
    s_a["model_name"] = s_a.get("model_name") or r_a.get("Model_Information", {}).get("model_name", "A")
    s_b["model_name"] = s_b.get("model_name") or r_b.get("Model_Information", {}).get("model_name", "B")

    # 1) summary bar
    plot_summary_bar(s_a, s_b, OUT_DIR / "summary_comparison.png")

    # 2) precision/recall/f1 grouped charts (saved per metric)
    plot_prf_grouped(s_a, s_b, OUT_DIR / "prf_grouped.png")

    # 3) per-class comparison (F1)
    plot_per_class_comparison(s_a, s_b, OUT_DIR / "per_class_f1_comparison.png")

    # 4) confusion matrices side-by-side
    plot_confusion_matrices(s_a, s_b, OUT_DIR / "confusion_matrices.png")

    print(f"Charts saved to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    # validate paths
    for p in (json_path_a, json_path_b):
        if not os.path.exists(p):
            print(f"ERROR: JSON file not found: {p}")
    # run
    main(json_path_a, json_path_b)
