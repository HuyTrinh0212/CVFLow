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

# Handle output_path - required, no default
if "output_path" not in config or config["output_path"] is None or not str(config["output_path"]).strip():
    print("Error: 'output_path' is required in config.yaml (no default). Please set output_path: /path/to/output/dir")
    sys.exit(1)
output_path = os.path.abspath(str(config["output_path"]).strip())
if os.path.exists(output_path) and not os.path.isdir(output_path):
    print(f"Error: output_path is a file, not a directory: {output_path}")
    sys.exit(1)
try:
    os.makedirs(output_path, exist_ok=True)
except PermissionError as e:
    print(f"Error: Cannot create output_path '{output_path}': {e}")
    sys.exit(1)

# Target device config
target_device = config.get("target", {}).get("device", "cpu")

# Display option config (default False for headless)
display = config.get("display", config.get("display_option", False))
# Ensure boolean type
display = bool(display) if isinstance(display, bool) else str(display).lower() in ("true", "1", "yes")

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
    display=display,
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
