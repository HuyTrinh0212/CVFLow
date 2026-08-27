"""Adapter stage - converts raw model output to canonical format."""
from cvf.core.pipeline.stage import Stage
from cvf.core.pipeline.context import PipelineContext
from cvf.core.contracts.runtime.output import ModelOutput
from cvf.core.registry import adapter_registry


class AdapterStage(Stage):
    """Convert raw model output to canonical output using model+task adapter."""
    
    name = "adapter"
    
    def run(self, context: PipelineContext) -> PipelineContext:
        if context.raw_output is None:
            raise ValueError("Raw output not available. Run inference stage first.")
        
        # Merge model + task config so adapter is config-driven (model ↔ task separation)
        # Adapter is per model+task (e.g. classification.resnet50.classification), needs both dims
        model_cfg = context.config.model.model_dump()
        task_cfg = context.config.task.model_dump() if hasattr(context.config, "task") and context.config.task else {}
        # Forward task-level dataset/class_names without mutating model namespace
        if task_cfg:
            # Prefer explicit class_names from task; fallback to dataset registry lookup later
            if task_cfg.get("class_names"):
                model_cfg.setdefault("class_names", task_cfg["class_names"])
                model_cfg["task_class_names"] = task_cfg["class_names"]
            if task_cfg.get("dataset"):
                model_cfg.setdefault("dataset", task_cfg["dataset"])
            # Forward top_k/threshold for adapters that need it
            for k in ("top_k", "threshold", "conf_threshold"):
                if k in task_cfg and task_cfg[k] is not None:
                    model_cfg.setdefault(k, task_cfg[k])
        # Get model+task specific adapter with merged config
        adapter = adapter_registry.create(
            f"{context.model_family}.{context.model_variant}.{context.task_type}",
            model_cfg
        )
        
        # Convert to canonical format
        canonical = adapter.adapt(context.raw_output.data)
        
        context.canonical_output = canonical
        
        return context