import os
import yaml
from AI_Script.core.utils import PROJECT_ROOT
from AI_Script.core.pipeline_inference import Pipeline_Inference
from AI_Script.core.pipeline_evaluate import Pipeline_Evaluate
from AI_Script.core.pipeline_tracking import Pipeline_Tracking
from AI_Script.core.pipeline_debug import Pipeline_Debug
from AI_Script.core.pipeline_benchmark import Pipeline_Benchmark
from AI_Script.core.pipeline_compare_intermediate import Pipeline_Compare_Intermediate

class factory_pipeline:
    def __init__(self,
                 model_name,
                 weight_path=None,
                 precision_format=None,
                 output_path=os.path.join(PROJECT_ROOT, "outputs"),
                 inference_mode=True,
                 debug_mode=False,
                 evaluate_mode=False,
                 tracking_option=False,
                 target_device="cpu"):
        # init parameters
        self.model_name = model_name
        self.weight_path = weight_path
        self.precision_format = precision_format
        self.output_path = output_path
        self.tracking_option = tracking_option
        self.target_device = target_device

        # mode
        self.inference_mode = inference_mode
        self.debug_mode = debug_mode
        self.evaluate_mode = evaluate_mode

        # Load model config
        self.config = self.load_config(self.model_name)
        self.pipe_benchmark = Pipeline_Benchmark()
        self.pipe_compare_intermediate = Pipeline_Compare_Intermediate()

        # Overwrite
        if self.weight_path:
            self.config["weight_path"] = str(self.weight_path)
        if self.precision_format:
            self.config['precision_format'] = str(self.precision_format)
        if self.target_device:
            self.config['target_device'] = str(self.target_device)

        # Init pipeline
        self.pipeline = self._get_pipeline()

    @staticmethod
    def load_config(model_name):
        config = {}
        config_path = os.path.join(PROJECT_ROOT, f"configs/{model_name}.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return config

    def _get_pipeline(self):
        # Inference
        if self.inference_mode:
            if self.tracking_option:
                return Pipeline_Tracking(config=self.config)
            else:
                return Pipeline_Inference(config=self.config)
        # Debug
        elif self.debug_mode:
            return Pipeline_Debug(config=self.config)
        # Evaluate
        elif self.evaluate_mode:
            return Pipeline_Evaluate(config=self.config)
        # Invalid
        else:
            raise ValueError("Invalid pipeline, the cause may be parameter conflicts.")

    def benchmark_dashboard(self, input_source):
        self.pipe_benchmark(input_source)

    def compare_intermediate(self, input_source):
       return self.pipe_compare_intermediate(input_source)

    def __call__(self, input_source):
        self.pipeline(input_source)
