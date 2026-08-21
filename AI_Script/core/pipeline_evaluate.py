import os
import time
from pathlib import Path
from AI_Script.models.factory_model import ModelFactory
from AI_Script.preprocess.factory_preprocess import PreprocessorFactory
from AI_Script.postprocess.factory_postprocess import PostprocessorFactory
from AI_Script.evaluate.factory_evaluate import EvaluateFactory
from AI_Script.core.utils import check_file
import json
from datetime import datetime
from AI_Script.core.utils import PROJECT_ROOT


class Pipeline_Evaluate:
    def __init__(self, config):
        self.config = config
        self.model_name = str(self.config.get("model_name"))
        self.precision_format = str(self.config.get("precision_format"))
        self.weight_path = self._get_weight_path()
        self.outputs_path = os.path.join(PROJECT_ROOT, "outputs")

        # pre-process -> AI inference -> evaluate
        self.preprocessor = PreprocessorFactory.create(config=config)
        self.model = ModelFactory.create(config=config)
        self.evaluate = EvaluateFactory.create(config=config)

    def _get_weight_path(self):
        if self.config.get("weight_path"):
            return str(self.config.get("weight_path"))
        else:
            if self.precision_format == 'fp32':
                return str(self.config.get("weight_path_fp32"))
            elif self.precision_format == 'fp16':
                return str(self.config.get("weight_path_fp16"))
            elif self.precision_format == 'int8':
                return str(self.config.get("weight_path_int8"))
            else:
                raise FileNotFoundError(f"No support precision format: {self.precision_format}")

    def _process_single_item(self, item_source):
        preprocessed_data = self.preprocessor(item_source)
        return self.model(preprocessed_data)

    def run(self, input_source):
        input_type = check_file(input_source)
        if input_type == 'folder_path':
            print("Start evaluating...")
            # Create dict
            final_results = {
                "Model_Information": {
                    "model_name": str(self.config.get("model_name")),
                    "run_target": "CPU",
                    "task": str(self.config.get("task")),
                    "weight_path": self.weight_path,
                    "precision_format": self.precision_format,
                    "model_size_mb": f"{os.path.getsize(self.weight_path) / (1024 * 1024):.2f} MB",
                    "target_size": list(self.config.get("target_size")),
                },
                "Dataset_Information": {
                    "dataset_path": str(input_source),
                    "number_images": int
                },
                "Performance": {
                    "total_time_s": float,
                    "throughput_fps": float
                },
                "Evaluate": {
                    "images": {},
                    "overall_metrics": {}
                },
            }

            # Step 1: Predict all images (pre-process -> AI inference)
            start_time = time.time()
            if input_type == 'folder_path':
                # Images
                images_path = os.path.join(input_source, "images")
                list_images = os.listdir(images_path)
                # Labels
                labels_path = os.path.join(input_source, "labels")
                for idx, img in enumerate(list_images):
                    # Predict
                    img_path = os.path.join(images_path, img)
                    pred = self._process_single_item(img_path)
                    # extract name and convert to txt
                    name, _ = os.path.splitext(img)
                    label_path = os.path.join(labels_path, f"{name}.txt")
                    # Evaluate
                    s_evaluate = self.evaluate.benchmark_single(pred, img_path, label_path)
                    # Update dict
                    final_results["Evaluate"]["images"][str(img)] = s_evaluate

            end_time = time.time()
            elapsed = end_time - start_time

            # Update dict
            final_results["Dataset_Information"]["number_images"] = int(len(list_images))
            final_results["Performance"]["total_time_s"] = float(elapsed)
            final_results["Performance"]["throughput_fps"] = float(len(list_images) / elapsed) if elapsed > 0.0 else 0.0

            # Calculate average mAP
            evaluate_total = self.evaluate.benchmark_toltal(final_results["Evaluate"])
            final_results["Evaluate"]["overall_metrics"].update(evaluate_total)

            # Save evaluate
            save_path = os.path.join(PROJECT_ROOT, f"outputs/evaluate - {self.model_name} - {self.precision_format} - CPU - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=5)
            print(f"Saved evaluate to {save_path}")

        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    def __call__(self, input_source):
        return self.run(input_source)