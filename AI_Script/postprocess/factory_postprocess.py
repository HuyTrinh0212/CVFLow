import os
import yaml
from AI_Script.postprocess.registry_postprocess import PostProcessorRegistry
from pathlib import Path

current_path = Path(__file__) # Current dir
parent_path = current_path.parent.parent
configs_dir = os.path.join(parent_path, "configs") # Config dir

class PostprocessorFactory:
    @staticmethod
    def create(config):
        # get class from registry
        default_postprocess = str(config.get("default_postprocess"))
        PostProcessorClass = PostProcessorRegistry.get(default_postprocess)
        return PostProcessorClass(config)
