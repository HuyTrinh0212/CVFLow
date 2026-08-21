import os.path
import time
import json
from datetime import datetime
import os
from AI_Script.core.utils import PROJECT_ROOT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools


class Pipeline_Benchmark:
    def _annotate_bars(self, ax, rects, fmt="{:.3f}", y_offset=0.01):
        """Annotate a list of bar containers (rects) with their heights."""
        for rect in rects:
            for r in rect:
                h = r.get_height()
                if np.isnan(h):
                    txt = fmt.format(0.0)
                else:
                    txt = fmt.format(h)
                ax.annotate(txt,
                            xy=(r.get_x() + r.get_width() / 2, h),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

    def _plot_overall_metrics(self, m_a, m_b, save_path):
        keys = ["accuracy", "mcc"]
        a_vals = [m_a['Evaluate']["overall_metrics"].get(k, np.nan) for k in keys]
        b_vals = [m_b['Evaluate']["overall_metrics"].get(k, np.nan) for k in keys]
        x = np.arange(len(keys))
        width = 0.35

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - width / 2, a_vals, width, label=m_a.get("name", "Model A"))
        ax.bar(x + width / 2, b_vals, width, label=m_b.get("name", "Model B"))
        ax.set_xticks(x);
        ax.set_xticklabels(keys)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Value")
        ax.set_title("Overall metrics (accuracy & mcc)")
        ax.legend()
        for i, (av, bv) in enumerate(zip(a_vals, b_vals)):
            ax.text(i - width / 2, av + 0.01, f"{av:.3f}", ha='center', va='bottom', fontsize=9)
            ax.text(i + width / 2, bv + 0.01, f"{bv:.3f}", ha='center', va='bottom', fontsize=9)

        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_avg_metrics(self, m_a, m_b, save_path):
        groups = ["macro_avg", "micro_avg", "weighted_avg"]
        metrics = ["precision", "recall", "f1-score"]
        rows, labels = [], []
        for g in groups:
            a_vals = m_a['Evaluate']["overall_metrics"].get(g, {})
            b_vals = m_b['Evaluate']["overall_metrics"].get(g, {})
            rows.append([a_vals.get(m, np.nan) for m in metrics])
            rows.append([b_vals.get(m, np.nan) for m in metrics])
            labels.append(g + " (A)")
            labels.append(g + " (B)")
        df = pd.DataFrame(rows, index=labels, columns=metrics)

        x = np.arange(len(labels))
        total_width = 0.75
        width = total_width / len(metrics)
        offsets = np.linspace(-total_width / 2 + width / 2, total_width / 2 - width / 2, len(metrics))

        fig, ax = plt.subplots(figsize=(9, 4))
        for i, met in enumerate(metrics):
            vals = df[met].values
            ax.bar(x + offsets[i], vals, width, label=met)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Value")
        ax.set_title("Macro / Micro / Weighted averages (precision, recall, f1)")
        ax.legend()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_per_class_metrics(self, m_a, m_b, save_path, top_n=None):
        per_a = m_a['Evaluate']["overall_metrics"]["Per_class"]
        per_b = m_b['Evaluate']["overall_metrics"]["Per_class"]
        classes = sorted(k for k in per_a.keys() if k.isdigit())
        if top_n:
            classes = classes[:top_n]
        metrics = ["precision", "recall", "f1-score"]
        n_classes = len(classes)
        ind = np.arange(n_classes)
        width_group = 0.25  # width per metric group

        fig, ax = plt.subplots(figsize=(max(8, n_classes * 0.45), 4))
        # We'll plot A and B side-by-side per metric for clarity
        # For each class, stack groups like: [A.prec, B.prec] [A.rec, B.rec] [A.f1, B.f1] offset horizontally
        for i, met in enumerate(metrics):
            a_vals = [per_a[c][met] for c in classes]
            b_vals = [per_b[c][met] for c in classes]
            offset = (i - 1) * (width_group + 0.02)  # center group
            ax.bar(ind + offset - width_group / 2, a_vals, width_group / 2,
                   label=f"{m_a.get('name', 'A')} {met}" if i == 0 else None)
            ax.bar(ind + offset + width_group / 2, b_vals, width_group / 2,
                   label=f"{m_b.get('name', 'B')} {met}" if i == 0 else None, alpha=0.75)

        ax.set_xticks(ind)
        ax.set_xticklabels(classes)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Value")
        ax.set_title("Per-class metrics — Precision / Recall / F1 (A vs B)")
        # custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='C0', label=m_a.get('name', 'Model A')),
            Patch(facecolor='C1', label=m_b.get('name', 'Model B')),
        ]
        ax.legend(handles=legend_elements)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_confusion_matrices(self, m_a, m_b, save_path,
                                normalize=False,
                                cmap="viridis",
                                figsize=(10, 5),
                                dpi=200):
        # lấy matrix
        cm_a = np.array(m_a['Evaluate']["overall_metrics"]["Confusion_matrix"], dtype=float)
        cm_b = np.array(m_b['Evaluate']["overall_metrics"]["Confusion_matrix"], dtype=float)
        n_classes = cm_a.shape[0]
        classes = [str(i) for i in range(n_classes)]

        name_a = str(m_a.get("name", "Model A"))
        name_b = str(m_b.get("name", "Model B"))

        # normalize nếu cần
        if normalize:
            cm_a = cm_a / cm_a.sum(axis=1, keepdims=True)
            cm_b = cm_b / cm_b.sum(axis=1, keepdims=True)
            fmt = "{:.2%}"
        else:
            fmt = "{:d}"

        # xử lý đường dẫn
        if os.path.isdir(save_path) or save_path.endswith(os.sep):
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, "confusion_matrices.png")
        else:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # tạo figure 2 subplot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        for ax, cm, title in zip(axes, [cm_a, cm_b], [name_a, name_b]):
            im = ax.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")
            ax.set_title(f"Confusion matrix — {title}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(classes)
            ax.set_yticklabels(classes)

            thresh = cm.max() / 2.0
            for i, j in itertools.product(range(n_classes), range(n_classes)):
                val = cm[i, j]
                txt = fmt.format(val if normalize else int(val))
                ax.text(j, i, txt,
                        ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontsize=7)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        fig.subplots_adjust(wspace=0.3, right=0.92)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_performance_metrics(self, m_a, m_b, save_path, dpi=200, fmt="{:.3f}"):
        # Extract values
        perf_a = m_a.get("Performance", {})
        perf_b = m_b.get("Performance", {})

        metrics = ["total_time_s", "throughput_fps"]
        a_vals = [perf_a.get(k, np.nan) for k in metrics]
        b_vals = [perf_b.get(k, np.nan) for k in metrics]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle("Performance Comparison (Time & Throughput)", fontsize=12, weight="bold")

        # --- Plot total time (lower = better) ---
        ax = axes[0]
        x = np.arange(1)
        width = 0.35
        bars_a = ax.bar(x - width / 2, [a_vals[0]], width, label=m_a.get("name", "Model A"))
        bars_b = ax.bar(x + width / 2, [b_vals[0]], width, label=m_b.get("name", "Model B"))
        self._annotate_bars(ax, [bars_a, bars_b], fmt=fmt)

        ax.set_xticks([])
        ax.set_ylabel("Seconds")
        ax.set_title("Total time per batch (lower is better)")
        ax.legend(fontsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # --- Plot throughput (higher = better) ---
        ax = axes[1]
        x = np.arange(1)
        bars_a = ax.bar(x - width / 2, [a_vals[1]], width, label=m_a.get("name", "Model A"))
        bars_b = ax.bar(x + width / 2, [b_vals[1]], width, label=m_b.get("name", "Model B"))
        self._annotate_bars(ax, [bars_a, bars_b], fmt=fmt)

        ax.set_xticks([])
        ax.set_ylabel("Frame per second (FPS)")
        ax.set_title("Throughput (higher is better)")
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Footer note
        fig.text(0.5, -0.02,
                 "Left: total_time_s (execution time). Right: throughput_fps (frames/sec).",
                 ha='center', fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def run(self, input_source):
        input_source_1, input_source_2 = input_source

        # Init parameter
        output_folder = os.path.join(PROJECT_ROOT, f"outputs/Benchmark_Chart_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
        os.makedirs(output_folder, exist_ok=True)

        # Load
        with open(input_source_1, 'rb') as f:
            data1 = json.load(f)
        with open(input_source_2, 'rb') as f:
            data2 = json.load(f)

        # Performance AI Model
        self._plot_overall_metrics(data1, data2, os.path.join(output_folder, "overall_metrics.png"))
        self._plot_avg_metrics(data1, data2, os.path.join(output_folder, "avg_metrics.png"))
        self._plot_per_class_metrics(data1, data2, os.path.join(output_folder, "per_class_metrics.png"))
        self._plot_confusion_matrices(data1, data2, os.path.join(output_folder, "confusion_matrices.png"))

        # Performance Hardware
        self._plot_performance_metrics(data1, data2, os.path.join(output_folder, "performance_hardware.png"))

    def __call__(self, input_source):
        return self.run(input_source)