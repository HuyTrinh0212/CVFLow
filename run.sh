#!/bin/bash

# Check if config file argument provided
if [ $# -lt 1 ]; then
    echo "Usage: ./run.sh <config.yaml>"
    exit 1
fi

CONFIG_FILE=$1

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
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

echo "Running with config: $CONFIG_FILE"
python3 main.py "$CONFIG_FILE"