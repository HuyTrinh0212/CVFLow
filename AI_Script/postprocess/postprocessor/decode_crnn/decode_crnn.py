from pathlib import Path
import torch
from AI_Script.postprocess.registry_postprocess import PostProcessorRegistry
from AI_Script.postprocess.base_postprocess import BasePostprocessor
from AI_Script.postprocess.postprocessor.decode_crnn.ctc_decode import CTC_DECODE
from AI_Script.postprocess.postprocessor.decode_crnn.dataset_crnn import Synth90kDataset

@PostProcessorRegistry.register("decode_crnn")
class DecodeCRNN(BasePostprocessor):
    def __init__(self, config=None):
        super().__init__(config)
        self.method = str(self.config.get("method", "beam_search"))
        self.beam_size = int(self.config.get("beam_size", 10))

    def postprocess(self, raw_outputs, raw_input=None):
        logits_np = raw_outputs[0]
        tensor_torch = torch.from_numpy(logits_np)
        # tensor_torch = tensor_torch.float() #fp16
        log_probs = torch.nn.functional.log_softmax(tensor_torch, dim=2)
        preds = CTC_DECODE(log_probs, method=self.method, beam_size=self.beam_size, label2char=Synth90kDataset.LABEL2CHAR)
        output_text = ''.join(preds[0])
        return output_text