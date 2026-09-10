import os
import os.path
import glob
import json
from AI_Script.postprocess.Functions.Compare_Output_Intermediate import Compare_Ouput_Intermediate


def _load_intermediate_source(source):
    """
    Load intermediate data from either:
    - a single monolithic JSON file (legacy: {layer_name: {dtype, shape, values}})
    - a directory of per-layer JSONs (new: each file {layer_name, index, dtype, shape, values})
    Returns dict {layer_name: {dtype, shape, values}}
    """
    # Directory -> per-layer files
    if os.path.isdir(source):
        data = {}
        json_files = sorted(glob.glob(os.path.join(source, "*.json")))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in directory: {source}")
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # New per-layer format has layer_name at top level
            if isinstance(payload, dict) and "layer_name" in payload and "values" in payload:
                layer_name = str(payload["layer_name"])
                data[layer_name] = {
                    "dtype": payload.get("dtype", ""),
                    "shape": payload.get("shape", ""),
                    "values": payload["values"],
                }
            elif isinstance(payload, dict):
                # Might be a legacy monolithic file inside directory (unlikely)
                # If payload looks like {layer: {values:...}}, merge it
                is_legacy = all(isinstance(v, dict) and "values" in v for v in payload.values())
                if is_legacy:
                    data.update(payload)
                else:
                    raise ValueError(f"Unrecognized JSON format in {jf}")
            else:
                raise ValueError(f"Unrecognized JSON format in {jf}")
        return data

    # Single file -> legacy or single per-layer file
    if os.path.isfile(source):
        with open(source, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # New per-layer single file
        if isinstance(payload, dict) and "layer_name" in payload and "values" in payload:
            layer_name = str(payload["layer_name"])
            return {layer_name: {"dtype": payload.get("dtype", ""), "shape": payload.get("shape", ""), "values": payload["values"]}}
        # Legacy monolithic file
        if isinstance(payload, dict):
            return payload
        raise ValueError(f"Unrecognized JSON format in {source}")

    raise FileNotFoundError(f"Intermediate source not found: {source}")


class Pipeline_Compare_Intermediate:

    def run(self, input_source):
        input_source_1, input_source_2 = input_source

        data1 = _load_intermediate_source(input_source_1)
        data2 = _load_intermediate_source(input_source_2)

        return Compare_Ouput_Intermediate(data1, data2)

    def __call__(self, input_source):
        return self.run(input_source)