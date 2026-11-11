import os
import yaml
from AI_Script.preprocess.registry_preprocess import PreprocessRegistry
from pathlib import Path

current_path = Path(__file__) # Current dir
parent_path = current_path.parent.parent
configs_dir = os.path.join(parent_path, "configs") # Config dir

class PreprocessorFactory:
    @staticmethod
    def create(config):
        # get class from registry
        model_name = str(config.get("model_name"))
        PreprocessorClass = PreprocessRegistry.get(model_name)
        return PreprocessorClass(config)
