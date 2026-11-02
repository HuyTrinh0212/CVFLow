import os
import re
import hashlib
from AI_Script.models.factory_model import ModelFactory
from AI_Script.preprocess.factory_preprocess import PreprocessorFactory
from AI_Script.postprocess.factory_postprocess import PostprocessorFactory
from AI_Script.core.utils import check_file
import numpy as np
import onnx
from onnx import helper, shape_inference
import json
import glob

class Pipeline_Debug:
    def __init__(self, config):
        self.config = config
        if not self.config.get("output_path") or not str(self.config.get("output_path")).strip():
            raise ValueError("'output_path' is required in config (no default).")
        self.output_path = os.path.abspath(str(self.config.get("output_path")).strip())
        os.makedirs(self.output_path, exist_ok=True)
        self.model_name = str(self.config.get("model_name"))
        self.precision_format = str(self.config.get("precision_format"))
        self.model_path = self._get_model_path()
        # dump model -> pre-process -> AI inference -> post-process
        self._dump_model()
        self.preprocessor = PreprocessorFactory.create(config=self.config)
        self.model = ModelFactory.create(config=self.config, debug_mode=True)
        self.postprocessor = PostprocessorFactory.create(config=self.config)

    def _process_single_item(self, item_source):
        preprocessed_data = self.preprocessor(item_source)
        (model_output, model_profiling) = self.model(preprocessed_data)
        return (model_output, model_profiling)

    def _get_model_path(self):
        if self.config.get("weight_path"):
            return str(self.config.get("weight_path"))
        else:
            if self.precision_format == 'fp32':
                return str(self.config.get("weight_path_fp32"))
            elif self.precision_format == 'fp16':
                return str(self.config.get("weight_path_fp16"))
            elif self.precision_format == 'int8':
                return str(self.config.get("weight_path_int8"))
            elif self.precision_format == 'bf16':
                return str(self.config.get("weight_path_bf16"))
            else:
                raise FileNotFoundError(f"No support precision format: {self.precision_format}")

    def _get_shape(self, obj):
        if isinstance(obj, np.ndarray):
            return str(obj.shape)
        if isinstance(obj, (list, tuple)):
            return str(len(obj))
        return "Not array-like"

    def _dump_model(self):
        # Load model
        model = onnx.load(self.model_path)
        inferred_model = shape_inference.infer_shapes(model)

        # Collect value_info
        vi_info = {
            vi.name: (
                vi.type.tensor_type.elem_type,
                [d.dim_value for d in vi.type.tensor_type.shape.dim]
            )
            for vi in inferred_model.graph.value_info
        }

        # Add initializers to vi_info
        for init in inferred_model.graph.initializer:
            if init.name not in vi_info:
                vi_info[init.name] = (init.data_type, list(init.dims))

        # Collect all input/output tensors.
        # Optimization (config key "debug_dump_initializers", default False): do NOT expose
        # initializers (weights/bias) as graph outputs. They are already stored inside the
        # .onnx file on disk; dumping them as JSON bloats RAM (tolist ~8x expansion) and disk
        # for no debug value. Set the key to true to restore the old behavior.
        # Empty tensor names ("") are optional inputs and cannot be dumped - skip them.
        include_init = bool(self.config.get("debug_dump_initializers", False))
        initializer_names = {init.name for init in model.graph.initializer}
        tensors_to_dump = set()
        for node in model.graph.node:
            for name in node.input:
                if name and (include_init or name not in initializer_names):
                    tensors_to_dump.add(name)
            for name in node.output:
                if name:
                    tensors_to_dump.add(name)

        def get_dtype_for_tensor(name):
            dtype = next((vi.type.tensor_type.elem_type for vi in inferred_model.graph.value_info if vi.name == name),
                         None)
            if dtype is None:
                dtype = next((i.type.tensor_type.elem_type for i in inferred_model.graph.input if i.name == name), None)
            if dtype is None:
                dtype = next((o.type.tensor_type.elem_type for o in inferred_model.graph.output if o.name == name),
                             None)
            if dtype is None:
                for init in inferred_model.graph.initializer:
                    if init.name == name:
                        dtype = init.data_type
                        break
            if dtype is None and name.endswith('_quantized'):
                producing_node = next((node for node in model.graph.node if name in node.output), None)
                if producing_node:
                    op_type = producing_node.op_type
                    y_zp_idx = None
                    if op_type == 'QuantizeLinear':
                        y_zp_idx = 2
                    elif op_type == 'QLinearConv':
                        y_zp_idx = 7
                    elif op_type == 'QGemm':
                        y_zp_idx = 8 if len(producing_node.input) == 9 else None
                    if y_zp_idx is not None and y_zp_idx < len(producing_node.input):
                        y_zp_name = producing_node.input[y_zp_idx]
                        for init in inferred_model.graph.initializer:
                            if init.name == y_zp_name:
                                dtype = init.data_type
                                break
            if dtype is None and name.endswith('_quantized'):
                dtype = onnx.TensorProto.INT8  # Fallback to INT8
            return dtype

        for name in tensors_to_dump:
            if not any(o.name == name for o in model.graph.output):
                dtype = get_dtype_for_tensor(name)
                if dtype is not None:
                    model.graph.output.append(helper.make_tensor_value_info(name, dtype, None))
                else:
                    print(f"Warning: dtype for {name} not found")

        # Release the explicit binding to the shape-inference copy after its last use.
        # (The dtype-check closure still holds a reference, so the copy is reaped when
        # this method returns - i.e. before the session is created and inference runs.)
        del inferred_model

        # Precompute the topological layer order ONCE from this graph (first output of each
        # node, in node order). Replaces _sort_right_index, which re-loaded the ONNX file on
        # every run just to recover the same order.
        self._layer_order = [str(list(node.output)[0]) for node in model.graph.node if node.output]

        # Lưu lại model tạm
        temp_model_path = os.path.join(self.output_path, "dump_model.onnx")
        onnx.save(model, temp_model_path)
        self.config["weight_path"] = str(temp_model_path)
        print(f"==Model are dump in {temp_model_path}==")

    def _sanitize_layer_name(self, name: str) -> str:
        # Replace illegal characters, truncate to 200 chars
        sanitized = re.sub(r'[^A-Za-z0-9._-]', '_', name)
        sanitized = sanitized.strip('_') or 'unnamed'
        if len(sanitized) > 200:
            sanitized = sanitized[:200]
        return sanitized

    def run(self, input_source):
        input_type = check_file(input_source)

        # Debug mode only supports single image input
        if input_type not in ['image_path', 'npy_path', 'numpy_array']:
            raise ValueError(
                f"Debug mode only supports single image input (image_path/npy_path/numpy_array), got '{input_type}'. "
                f"Folder and video inputs are not supported in debug_mode."
            )

        print("Detected single input. Processing...")
        # Step 1: pre-process -> ai-inference
        (result_1, profiling) = self._process_single_item(input_source)
        # Step 2: post-process
        result_2 = self.postprocessor(result_1, input_source)

        # Stream intermediate outputs to disk layer-by-layer instead of holding every
        # layer's tolist() in memory at once (the main cause of OOM on large models).
        output_names = [o.name for o in self.model.session.get_outputs()]
        name_to_result = dict(zip(output_names, result_1))
        del result_1

        if self.model_name == 'crnn':
            # Keep session output order as before (no topological re-sort for crnn)
            layer_names = list(output_names)
        else:
            # Topological order computed once in _dump_model; leftover tensors (e.g.,
            # secondary node outputs, original graph outputs) appended sorted, matching
            # the old _sort_right_index result without re-loading the model.
            ordered = [n for n in self._layer_order if n in name_to_result]
            leftover = sorted(set(name_to_result) - set(ordered))
            layer_names = ordered + leftover

        # Save each layer as separate JSON in intermediate_outputs/
        intermediate_dir = os.path.join(self.output_path, "intermediate_outputs")
        os.makedirs(intermediate_dir, exist_ok=True)

        # Clean previous intermediate outputs to avoid stale files
        for old_file in glob.glob(os.path.join(intermediate_dir, "*.json")):
            try:
                os.remove(old_file)
            except OSError:
                pass

        seen_sanitized = {}
        saved_count = 0
        for idx, layer_name in enumerate(layer_names):
            value = name_to_result.pop(layer_name, None)
            if value is None:
                continue
            payload = {
                "layer_name": layer_name,
                "index": idx,
                "dtype": str(value.dtype),
                "shape": self._get_shape(value),
                "values": value.tolist(),
            }
            # Free this layer's numpy array and Python-list representation immediately
            del value

            sanitized = self._sanitize_layer_name(layer_name)
            # Handle collisions after sanitization
            if sanitized in seen_sanitized:
                short_hash = hashlib.md5(layer_name.encode()).hexdigest()[:6]
                sanitized = f"{sanitized}_{short_hash}"
                # Truncate again if needed
                if len(sanitized) > 200:
                    sanitized = sanitized[:200]
            seen_sanitized[sanitized] = layer_name

            file_name = f"{idx:04d}__{sanitized}.json"
            file_path = os.path.join(intermediate_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            del payload
            saved_count += 1

        print(f"Output profiling are saved in {profiling}")
        print(f"Output intermediate layers are saved in {intermediate_dir} ({saved_count} files)")

        # Clean up temp dump model
        dump_path = str(self.config.get("weight_path", ""))
        if dump_path and os.path.basename(dump_path) == "dump_model.onnx" and os.path.exists(dump_path):
            try:
                os.remove(dump_path)
                print(f"Cleaned temp dump model: {dump_path}")
            except OSError as e:
                print(f"Warning: could not remove temp dump model {dump_path}: {e}")


    def __call__(self, input_source):
        return self.run(input_source)