import os
import yaml
from AI_Script.evaluate.registry_evaluate import EvaluateRegistry
from pathlib import Path

current_path = Path(__file__) # Current dir
parent_path = current_path.parent.parent

class EvaluateFactory:
    @staticmethod
    def create(config):
        # get class from registry
        task = str(config.get("task"))
        PreprocessorClass = EvaluateRegistry.get(task)
        return PreprocessorClass(config)
