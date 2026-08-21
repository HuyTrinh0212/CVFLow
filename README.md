# XAIP - AI Inference Pipeline

## Quick Start

### Option 1: Using run.sh (Recommended)

```bash
# Run with config file (auto-creates conda env 'xaip' if missing)
./run.sh config.yaml
```

The script will:
- Check for conda environment `xaip`, create from `environment.yml` if not found
- Activate the environment
- Run `python3 main.py config.yaml`

### Option 2: Manual Setup

#### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate xaip
```

#### 2. Install Dependencies

```bash
pip install onnxruntime opencv-python numpy pyyaml torch torchvision
```

For GPU support (CUDA):
```bash
pip install onnxruntime-gpu torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Run Main Script

```bash
python main.py config.yaml
```

---

## Project Structure

```
xaip/
├── main.py                    # Entry point (accepts config.yaml as argument)
├── run.sh                     # Auto-setup script (conda env + run)
├── environment.yml            # Conda environment (python=3.12)
├── config.yaml                # Pipeline configuration (paths, model, modes)
├── configs/                   # Model configurations (YAML)
├── inputs/                    # Input data (images, videos)
├── outputs/                   # Output results (created automatically)
└── AI_Script/                 # Core pipeline
    ├── core/                  # Pipelines (inference, tracking, debug, evaluate, benchmark)
    ├── models/                # Model implementations (YOLOv5, CRNN, ResNet, LeNet)
    ├── preprocess/            # Preprocessing (YOLO, CRNN, Classification)
    ├── postprocess/           # Postprocessing (draw boxes, decode CRNN, etc.)
    ├── tracker/               # ByteTrack implementation
    └── evaluate/              # Evaluation tasks
```

---

## Modes (in main.py)

| Mode | Description |
|------|-------------|
| `inference_mode=True` | Standard inference |
| `tracking_option=True` | Enable ByteTrack object tracking |
| `debug_mode=True` | Profile model performance |
| `evaluate_mode=True` | Run evaluation benchmarks |

---

## Input Types Supported

- Single image: `image_path`, `npy_path`, `numpy_array`
- Folder of images: `folder_path`
- Video file: `video_path`

---

## Requirements

- Python 3.12
- ONNX Runtime (CPU or GPU)
- OpenCV
- NumPy
- PyYAML
- PyTorch (for ByteTrack tracker)

---

## Config.yaml Options

| Key | Description | Default |
|-----|-------------|---------|
| `input_path` | Path to input (image, folder, video, npy) | Required |
| `weight_path` | Path to ONNX model weights | Required |
| `output_path` | Output directory for results | `cwd/outputs/` |
| `model_name` | Model name (yolov5, crnn, resnet, lenet) | Required |
| `precision_format` | Precision (int8, fp32, fp16) | Required |
| `inference_mode` | Standard inference | `true` |
| `debug_mode` | Profile model performance | `false` |
| `evaluate_mode` | Run evaluation benchmarks | `false` |
| `tracking_option` | Enable ByteTrack tracking | `false` |

> **Note**: Only one mode can be active at a time (inference/debug/evaluate).
> `output_path` directory is created automatically if it doesn't exist.
