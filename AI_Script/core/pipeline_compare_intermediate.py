import os.path
import time
import json
from AI_Script.postprocess.Functions.Compare_Output_Intermediate import Compare_Ouput_Intermediate


class Pipeline_Compare_Intermediate:

    def run(self, input_source):
        input_source_1, input_source_2 = input_source

        # Load
        with open(input_source_1, 'rb') as f:
            data1 = json.load(f)
        with open(input_source_2, 'rb') as f:
            data2 = json.load(f)

        return Compare_Ouput_Intermediate(data1, data2)

    def __call__(self, input_source):
        return self.run(input_source)