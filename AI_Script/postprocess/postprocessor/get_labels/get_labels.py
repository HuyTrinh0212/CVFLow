import os
import json
import pickle
import numpy as np
from pathlib import Path
from AI_Script.postprocess.registry_postprocess import PostProcessorRegistry
from AI_Script.postprocess.base_postprocess import BasePostprocessor

current_path = Path(__file__)
parent_path = current_path.parent.parent.parent.parent

@PostProcessorRegistry.register("get_label")
class GetLabel(BasePostprocessor):
    """
    This function will return the text label corresponding with dataset
    Input: numpy array with shape (Batch, Num class)
    Output: list of text label (Batch)
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.dataset = str(self.config.get("dataset"))
        self.json_dataset = self.get_json()

    @staticmethod
    def get_json():
        json_path = os.path.join(parent_path, "configs/datasets.json")
        with open(json_path, "r") as f:
            jsons = json.load(f)
        return jsons

    def postprocess(self, raw_output, raw_input=None):
        max_index = np.argmax(raw_output[0])
        text_label = self.json_dataset[self.dataset][str(max_index)]
        return text_label
