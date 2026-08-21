"""Benchmark stage - measures performance metrics."""
from cvf.core.pipeline.stage import Stage
from cvf.core.pipeline.context import PipelineContext
from cvf.benchmark.runner import BenchmarkRunner


class BenchmarkStage(Stage):
    """Run benchmark measurements."""
    
    name = "benchmark"
    
    def __init__(self, config):
        super().__init__(config)
        self.runner = BenchmarkRunner(config)
    
    def run(self, context: PipelineContext) -> PipelineContext:
        # Run benchmark using the runner
        metrics = self.runner.run(context)
        context.benchmark_metrics = metrics
        return context