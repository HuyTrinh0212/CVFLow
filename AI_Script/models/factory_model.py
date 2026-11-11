import os
import yaml
from AI_Script.models.registry_model import ModelRegistry
from pathlib import Path

current_path = Path(__file__) # Current dir
parent_path = current_path.parent.parent
configs_dir = os.path.join(parent_path, "configs") # Config dir

class ModelFactory:
    @staticmethod
    def create(config, debug_mode=False):
        # get class from registry
        model_name = str(config.get("model_name"))
        PreprocessorClass = ModelRegistry.get(model_name)
        return PreprocessorClass(config, debug_mode=debug_mode)
