#!/bin/bash

# Check if config file argument provided
if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <config.yaml> [input] [command]"
    echo "  config.yaml : path to YAML config (required, must contain 'input' or pass [input])"
    echo "  input       : optional input image/video (overrides 'input' in config)"
    echo "  command     : run (default) or benchmark"
    echo ""
    echo "Examples:"
    echo "  ./run.sh cvf/configs/test_yolo.yaml"
    echo "  ./run.sh cvf/configs/test_yolo.yaml /path/to/image.jpg"
    echo "  ./run.sh cvf/configs/test_yolo.yaml benchmark"
    echo "  ./run.sh cvf/configs/test_yolo.yaml /path/to/image.jpg benchmark"
    exit 1
fi

CONFIG_FILE=$1
IMAGE_PATH=""
COMMAND="run"

# Parse remaining args: [input] [command]
if [ $# -eq 2 ]; then
    if [ "$2" = "benchmark" ] || [ "$2" = "run" ]; then
        COMMAND=$2
    else
        IMAGE_PATH=$2
    fi
elif [ $# -ge 3 ]; then
    IMAGE_PATH=$2
    COMMAND=$3
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

# Check if image exists (only if provided)
if [ -n "$IMAGE_PATH" ] && [ ! -f "$IMAGE_PATH" ]; then
    echo "Error: Image file '$IMAGE_PATH' not found"
    exit 1
fi

# Initialize conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if conda env 'xaip' exists
if ! conda env list | grep -q "^xaip "; then
    echo "Conda environment 'xaip' not found. Creating from environment.yml..."
    if [ ! -f "environment.yml" ]; then
        echo "Error: environment.yml not found in project root"
        exit 1
    fi
    conda env create -f environment.yml
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create conda environment"
        exit 1
    fi
    echo "Environment 'xaip' created successfully"
fi

# Activate conda env and run
echo "Activating conda environment 'xaip'..."
conda activate xaip

echo "Running CVF with config: $CONFIG_FILE"
if [ -n "$IMAGE_PATH" ]; then
    echo "Input (override): $IMAGE_PATH"
fi
echo "Command: $COMMAND"

if [ "$COMMAND" = "benchmark" ]; then
    python -m cvf.cli.main benchmark "$CONFIG_FILE"
else
    if [ -n "$IMAGE_PATH" ]; then
        python -m cvf.cli.main run "$CONFIG_FILE" "$IMAGE_PATH"
    else
        python -m cvf.cli.main run "$CONFIG_FILE"
    fi
fi
