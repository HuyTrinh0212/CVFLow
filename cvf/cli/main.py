"""CVF CLI - command line interface."""
import sys
import argparse
from pathlib import Path

from cvf.core import load_config, Pipeline, CompatibilityResolver
from cvf.core.artifacts import artifacts_manager
from cvf.plugins.builtin import register_builtin


def run_inference(config_path: str, input_path: str) -> None:
    """Run inference pipeline."""
    # Register builtin components
    register_builtin()
    
    print(f"Loading config: {config_path}")
    config = load_config(config_path)
    
    # Validate compatibility
    CompatibilityResolver.validate(config)
    print("Config validation passed")
    
    # Initialize artifacts manager
    artifacts_manager.output_dir = Path(config.output_dir or "outputs")
    artifacts_manager.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create and run pipeline
    pipeline = Pipeline(config)
    print(f"Running pipeline with stages: {[s.name for s in pipeline.stages]}")
    
    context = pipeline(input_path)
    
    print("\n--- Pipeline Complete ---")
    print(f"Task result: {context.task_result}")
    if context.benchmark_metrics:
        print(f"Benchmark: {context.benchmark_metrics}")
    if context.evaluation_metrics:
        print(f"Evaluation: {context.evaluation_metrics}")
    
    print(f"\nStage timings:")
    for stage, timing in context.stage_timings.items():
        print(f"  {stage}: {timing*1000:.2f}ms")


def run_benchmark(config_path: str) -> None:
    """Run benchmark only."""
    register_builtin()
    
    print(f"Loading benchmark config: {config_path}")
    config = load_config(config_path)
    CompatibilityResolver.validate(config)
    
    artifacts_manager.output_dir = Path(config.output_dir or "outputs/benchmark")
    artifacts_manager.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run with dummy input for benchmark
    import numpy as np
    dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
    
    pipeline = Pipeline(config)
    context = pipeline(dummy_input)
    
    print("\n--- Benchmark Complete ---")
    if context.benchmark_metrics:
        print(f"Benchmark: {context.benchmark_metrics}")


def main():
    parser = argparse.ArgumentParser(description="CVF - Computer Vision Flow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run inference pipeline")
    run_parser.add_argument("config", help="Path to config YAML")
    run_parser.add_argument("input", help="Path to input image/video")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark")
    bench_parser.add_argument("config", help="Path to config YAML")
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_inference(args.config, args.input)
    elif args.command == "benchmark":
        run_benchmark(args.config)


if __name__ == "__main__":
    main()