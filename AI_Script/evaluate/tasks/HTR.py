from AI_Script.evaluate.base_evaluate import BaseEvaluate
from AI_Script.evaluate.registry_evaluate import EvaluateRegistry
from AI_Script.postprocess.postprocessor.decode_crnn.decode_crnn import DecodeCRNN
from typing import List, Tuple

@EvaluateRegistry.register("htr")
class htr_benchmark(BaseEvaluate):
    def __init__(self, config):
        super().__init__(config)
        self.DecodeCRNN = DecodeCRNN(self.config)

    def _get_text_decode(self, raw_output):
        return str((self.DecodeCRNN(raw_output)))

    def _get_labels(self, label_path):
        with open(label_path, "r") as f:
            return str(f.read().strip())

    @staticmethod
    def edit_distance(pred, gt) -> int:
        m, n = len(pred), len(gt)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if pred[i - 1] == gt[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # delete
                    dp[i][j - 1] + 1,  # insert
                    dp[i - 1][j - 1] + cost  # substitute
                )
        return dp[m][n]

    def _cer(self, pred, gt) -> Tuple[int, int]:
        """Character Error Rate raw values: (edits, gt_chars)"""
        edits = self.edit_distance(list(pred), list(gt))
        return edits, len(gt)

    def _wer(self, pred, gt) -> Tuple[int, int]:
        """Word Error Rate raw values: (edits_on_tokens, gt_tokens)"""
        pred_tokens = pred.split()
        gt_tokens = gt.split()
        edits = self.edit_distance(pred_tokens, gt_tokens)
        return edits, len(gt_tokens)

    def benchmark_single(self, predictitons, images_path, labels_path):
        preds = self._get_text_decode(predictitons)
        labels = self._get_labels(labels_path)

        benchmark = {
            "image_path": images_path,
            "metrics": {
                "correct": bool(preds == labels),
                "predict": str(preds),
                "ground_truth": str(labels),
            }
        }
        return benchmark

    def benchmark_toltal(self, benchmark_dict):
        preds = [data["metrics"]["predict"] for key, data in benchmark_dict["images"].items()]
        labels = [data["metrics"]["ground_truth"] for key, data in benchmark_dict["images"].items()]

        assert len(preds) == len(labels), "preds and gts must have same length"

        total = len(preds)
        correct = 0

        for i, (p, g) in enumerate(zip(preds, labels)):
            if p.strip().lower() == g.strip().lower():
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
        }