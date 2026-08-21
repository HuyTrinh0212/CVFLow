import os
from AI_Script.models.factory_model import ModelFactory
from AI_Script.preprocess.factory_preprocess import PreprocessorFactory
from AI_Script.postprocess.factory_postprocess import PostprocessorFactory
from AI_Script.core.utils import check_file, PROJECT_ROOT
import numpy as np
import onnx
from onnx import helper, shape_inference
from datetime import datetime
import pickle
import json

class Pipeline_Debug:
    def __init__(self, config):
        self.config = config
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

        # Collect all input/output tensors
        tensors_to_dump = set()
        for node in model.graph.node:
            tensors_to_dump.update(node.input)
            tensors_to_dump.update(node.output)

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

        # Lưu lại model tạm
        temp_model_path = os.path.join(PROJECT_ROOT, "outputs/dump_model.onnx")
        onnx.save(model, temp_model_path)
        self.config["weight_path"] = str(temp_model_path)
        print(f"==Model are dump in {temp_model_path}==")

    def _sort_right_index(self, data):
        model = onnx.load(self.model_path)
        onnx.checker.check_model(model)
        graph = model.graph

        dict = {}
        for i, node in enumerate(graph.node):
            output_name = str(list(node.output)[0])
            dict[output_name] = data[output_name]
        return dict

    def run(self, input_source):
        input_type = check_file(input_source)

        # --- Case 1: single image ---
        if input_type in ['image_path', 'npy_path', 'numpy_array']:
            print("Detected single input. Processing...")
            # Step 1: pre-process -> ai-inference
            (result_1, profiling) = self._process_single_item(input_source)
            # Step 2: post-process
            result_2 = self.postprocessor(result_1, input_source)

            # Preprocessing Profiler


            # Save intermediate output into pickle file
            out = {}
            output_names = [o.name for o in self.model.session.get_outputs()]
            for name, value in zip(output_names, result_1):
                out[str(name)] = {
                    "dtype": str(value.dtype),
                    "shape": self._get_shape(value),
                    "values": value.tolist(),
                }

            # Sort to right index
            if self.model_name != 'crnn':
                out = self._sort_right_index(out)

            # Save
            output_path = os.path.join(PROJECT_ROOT, f"outputs/Output intermediate layer - {self.model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.json")
            # Save intermediate
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=4)
            # Save profiling
            print(f"Output profiling are saved in {profiling}")
            print(f"Output intermediate layer are saved in {output_path}")

        # --- Case 2: folder images ---
        elif input_type == 'folder_path':
            print(f"Detected batch input (folder). Processing each item...")
            try:
                image_files = sorted(
                    [f for f in os.listdir(input_source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                if not image_files:
                    print(f"Warning: No images found in folder {input_source}")
                    return []
            except FileNotFoundError:
                print(f"Error: Folder not found at {input_source}")
                return []

            out = {}
            for filename in image_files:
                item_path = os.path.join(input_source, filename)
                try:
                    dict_format = {}
                    # Step 1: pre-process -> ai-inference
                    result_1 = self._process_single_item(item_path)
                    # Step 2: post-process
                    result_2 = self.postprocessor(result_1, item_path)
                    # Intermediate
                    output_names = [o.name for o in self.model.session.get_outputs()]
                    for name, value in zip(output_names, result_1):
                        dict_format[str(name)] = {
                            "dtype": str(value.dtype),
                            "shape": self._get_shape(value),
                            "values": value.tolist(),
                        }
                    out[str(filename)] = dict_format

                except Exception as e:
                    print(f"    ! Failed to process {filename}. Error: {e}")
            # SAVE
            output_pickle = os.path.join(PROJECT_ROOT, f"outputs/Output intermediate layer - {self.model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pkl")
            with open(output_pickle, "wb") as f:
                pickle.dump(out, f)

        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        # # Clean memory
        # if os.path.exists(str(self.config["weight_path"])):
        #     os.remove(str(self.config["weight_path"]))


    def __call__(self, input_source):
        return self.run(input_source)