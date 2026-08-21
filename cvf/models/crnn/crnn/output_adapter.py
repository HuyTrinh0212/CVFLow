"""CRNN output adapter - converts raw ONNX output to text via CTC decoding."""
import numpy as np
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from scipy.special import logsumexp
from collections import defaultdict

from cvf.core.contracts.runtime.output import CanonicalOutput


# Character set for Synth90k (same as original)
CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}
BLANK_LABEL = 0
NINF = -1 * float('inf')
DEFAULT_EMISSION_THRESHOLD = 0.01


def _reconstruct(labels, blank=BLANK_LABEL):
    """Merge same labels and remove blanks."""
    new_labels = []
    previous = None
    for l in labels:
        if l != previous:
            new_labels.append(l)
            previous = l
    new_labels = [l for l in new_labels if l != blank]
    return new_labels


def greedy_decode(emission_log_prob, blank=BLANK_LABEL, **kwargs):
    """Greedy CTC decoding."""
    labels = np.argmax(emission_log_prob, axis=-1)
    return _reconstruct(labels, blank=blank)


def beam_search_decode(emission_log_prob, blank=BLANK_LABEL, **kwargs):
    """Beam search CTC decoding."""
    beam_size = kwargs.get('beam_size', 10)
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [([], 0)]  # (prefix, accumulated_log_prob)
    for t in range(length):
        new_beams = []
        for prefix, accumulated_log_prob in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue
                new_prefix = prefix + [c]
                new_accu_log_prob = accumulated_log_prob + log_prob
                new_beams.append((new_prefix, new_accu_log_prob))

        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_size]

    total_accu_log_prob = {}
    for prefix, accu_log_prob in beams:
        labels = tuple(_reconstruct(prefix, blank))
        total_accu_log_prob[labels] = logsumexp([accu_log_prob, total_accu_log_prob.get(labels, NINF)])

    labels_beams = [(list(labels), accu_log_prob) for labels, accu_log_prob in total_accu_log_prob.items()]
    labels_beams.sort(key=lambda x: x[1], reverse=True)
    return labels_beams[0][0]


def prefix_beam_decode(emission_log_prob, blank=BLANK_LABEL, **kwargs):
    """Prefix beam search CTC decoding."""
    beam_size = kwargs.get('beam_size', 10)
    emission_threshold = kwargs.get('emission_threshold', np.log(DEFAULT_EMISSION_THRESHOLD))

    length, class_count = emission_log_prob.shape

    beams = [(tuple(), (0.0, NINF))]  # (prefix, (blank_log_prob, non_blank_log_prob))

    for t in range(length):
        new_beams_dict = {}
        
        for prefix, (lp_b, lp_nb) in beams:
            for c in range(class_count):
                log_prob = emission_log_prob[t, c]
                if log_prob < emission_threshold:
                    continue

                end_t = prefix[-1] if prefix else None

                if prefix not in new_beams_dict:
                    new_beams_dict[prefix] = (NINF, NINF)
                
                new_lp_b, new_lp_nb = new_beams_dict[prefix]

                if c == blank:
                    new_beams_dict[prefix] = (
                        logsumexp([new_lp_b, lp_b + log_prob, lp_nb + log_prob]),
                        new_lp_nb
                    )
                    continue
                if c == end_t:
                    new_beams_dict[prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_nb + log_prob])
                    )
                else:
                    new_prefix = prefix + (c,)
                    if new_prefix not in new_beams_dict:
                        new_beams_dict[new_prefix] = (NINF, NINF)
                    new_lp_b, new_lp_nb = new_beams_dict[new_prefix]
                    new_beams_dict[new_prefix] = (
                        new_lp_b,
                        logsumexp([new_lp_nb, lp_b + log_prob, lp_nb + log_prob])
                    )

        beams = sorted(new_beams_dict.items(), key=lambda x: logsumexp(x[1]), reverse=True)
        beams = beams[:beam_size]

    return list(beams[0][0])


DECODERS = {
    'greedy': greedy_decode,
    'beam_search': beam_search_decode,
    'prefix_beam_search': prefix_beam_decode,
}


def ctc_decode(log_probs: np.ndarray, label2char=None, blank=BLANK_LABEL, 
               method='beam_search', beam_size=10) -> List[str]:
    """
    CTC decoding for batched log probabilities.
    
    Args:
        log_probs: Log probabilities of shape (batch, length, class) or (length, class)
        label2char: Mapping from label to character
        blank: Blank label index
        method: Decoding method ('greedy', 'beam_search', 'prefix_beam_search')
        beam_size: Beam size for beam search
        
    Returns:
        List of decoded strings
    """
    # Ensure 3D: (batch, length, class)
    if log_probs.ndim == 2:
        log_probs = log_probs[np.newaxis, ...]
    
    # Transpose to (batch, length, class) if needed
    # Original expects (batch, length, class)
    if log_probs.shape[0] == 1 and log_probs.shape[1] > log_probs.shape[2]:
        log_probs = np.transpose(log_probs, (0, 2, 1))
    
    batch_size = log_probs.shape[0]
    decoded_list = []
    
    decoder = DECODERS.get(method, beam_search_decode)
    
    for b in range(batch_size):
        emission_log_prob = log_probs[b]  # (length, class)
        decoded = decoder(emission_log_prob, blank=blank, beam_size=beam_size)
        if label2char:
            decoded = [label2char[l] for l in decoded]
        decoded_list.append(''.join(decoded))
    
    return decoded_list


@dataclass
class HTROutput(CanonicalOutput):
    """Canonical HTR output."""
    text: str = ""
    confidence: float = 0.0


class CRNNHTRAdapter:
    """Adapter for CRNN raw output -> canonical HTROutput."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.method = self.config.get("method", "beam_search")
        self.beam_size = self.config.get("beam_size", 10)
        self.label2char = self.config.get("label2char", LABEL2CHAR)
    
    def adapt(self, raw_output: Any) -> HTROutput:
        """
        Convert raw CRNN output to text via CTC decoding.
        
        Args:
            raw_output: List of numpy arrays from ONNX Runtime, typically [length, batch, num_classes]
            
        Returns:
            HTROutput with decoded text
        """
        # Handle different output formats
        if isinstance(raw_output, (list, tuple)):
            outputs = raw_output[0]
        else:
            outputs = raw_output
        
        # CRNN ONNX output is typically (length, batch, class) = (24, 1, 37)
        # Transpose to (batch, length, class) then squeeze batch
        if outputs.ndim == 3:
            # Check if second dim is batch (size 1)
            if outputs.shape[1] == 1:
                outputs = outputs.transpose(1, 0, 2).squeeze(0)  # (length, class)
            elif outputs.shape[0] == 1:
                outputs = outputs.squeeze(0)  # (length, class)
        
        # Apply log_softmax (axis=-1 for class dimension)
        log_probs = self._log_softmax(outputs, axis=-1)
        
        # CTC decode
        texts = ctc_decode(
            log_probs,
            label2char=self.label2char,
            method=self.method,
            beam_size=self.beam_size
        )
        
        text = texts[0] if texts else ""
        
        return HTROutput(
            text=text,
            confidence=1.0,  # CTC doesn't directly give confidence
        )
    
    @staticmethod
    def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute log softmax."""
        max_val = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - max_val)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        return x - max_val - np.log(sum_exp)


def create_adapter(config: Optional[Dict] = None) -> CRNNHTRAdapter:
    """Factory function to create CRNN HTR adapter."""
    return CRNNHTRAdapter(config)