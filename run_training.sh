#!/bin/bash

# Example script to run training with different resolutions

# Train with 32x32 resolution
echo "Running training with 32x32 resolution"
accelerate launch train.py resolution=32x32

# Train with 64x64 resolution
echo "Running training with 64x64 resolution"
accelerate launch train.py resolution=64x64

# Train with 128x128 resolution
echo "Running training with 128x128 resolution"
accelerate launch train.py resolution=128x128

# Train with 256x256 resolution
echo "Running training with 256x256 resolution"
accelerate launch train.py resolution=256x256

# You can also specify the model type (dit or tread)
echo "Running training with 64x64 resolution and TREAD model"
accelerate launch train.py resolution=64x64 model=tread-64x64 