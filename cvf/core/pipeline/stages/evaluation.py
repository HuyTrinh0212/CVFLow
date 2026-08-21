"""Evaluation stage - evaluates model accuracy against ground truth."""
from cvf.core.pipeline.stage import Stage
from cvf.core.pipeline.context import PipelineContext
from cvf.evaluation.evaluator import Evaluator


class EvaluationStage(Stage):
    """Run evaluation against ground truth."""
    
    name = "evaluation"
    
    def __init__(self, config):
        super().__init__(config)
        self.evaluator = Evaluator(config)
    
    def run(self, context: PipelineContext) -> PipelineContext:
        metrics = self.evaluator.run(context)
        context.evaluation_metrics = metrics
        return context