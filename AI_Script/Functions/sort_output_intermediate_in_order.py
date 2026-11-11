import onnx
import json

# Đường dẫn tới file JSON
json_file = "/home/bht/CODE/xaip/AI_Script/outputs/Output intermediate layer - lenet5 - 2025-11-05 14:37:13.json"

# Load nội dung file JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Đường dẫn tới file ONNX
model_path = "/home/bht/CODE/xaip/AI_Script/pretrained_models/INT8/lenet5_47labels_int8_QO_QUInt8.onnx"

# Load mô hình
model = onnx.load(model_path)

# Kiểm tra mô hình hợp lệ
onnx.checker.check_model(model)

# Truy cập vào graph
graph = model.graph

print("===== Danh sách các layer (node) trong mô hình ONNX =====")
dict = {}
for i, node in enumerate(graph.node):
    output_name = str(list(node.output)[0])
    dict[output_name] = data[output_name]
    print(output_name)

with open("/home/bht/CODE/xaip/AI_Script/outputs/sorted.json", "w", encoding="utf-8") as f:
    json.dump(dict, f, ensure_ascii=False, indent=4)
print("Saved")
