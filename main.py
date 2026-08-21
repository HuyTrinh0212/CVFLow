import os
import sys
import yaml
from AI_Script.core.utils import PROJECT_ROOT
from AI_Script.core.factory import factory_pipeline

# Load config from command line argument
if len(sys.argv) < 2:
    print("Usage: python main.py <config.yaml>")
    sys.exit(1)

config_path = sys.argv[1]
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Handle output_path - default to cwd/outputs/
output_path = config.get("output_path", os.path.join(os.getcwd(), "outputs"))
os.makedirs(output_path, exist_ok=True)

# Target device config
target_device = config.get("target", {}).get("device", "cpu")

# INIT
factory = factory_pipeline(
    model_name=config["model_name"],
    precision_format=config["precision_format"],
    # mode
    inference_mode=config["inference_mode"],
    debug_mode=config["debug_mode"],
    evaluate_mode=config["evaluate_mode"],
    # option
    tracking_option=config["tracking_option"],
    # overwrite
    weight_path=config["weight_path"],
    output_path=output_path,
    target_device=target_device
)

# RUN
factory(config["input_path"])


# compare_intermediate
# outputs = factory.compare_intermediate(("/home/bht/CODE/xaip/AI_Script/outputs/Output intermediate layer - lenet5 - 2025-11-05 15:02:21.json",
#                               "/home/bht/CODE/xaip/AI_Script/outputs/Output intermediate layer - lenet5 - 2025-11-05 15:02:21.json"))
# print(outputs)


# benchmark_dashboard
# factory.benchmark_dashboard(("/home/bht/CODE/xaip/AI_Script/outputs/evaluate - lenet5 - int8 - CPU - 2025-11-06 11:46:18.json",
#                              "/home/bht/CODE/xaip/AI_Script/outputs/evaluate - lenet5 - int8 - RTL - 2025-11-06 11:43:04.json"))
