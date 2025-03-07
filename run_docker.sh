#!/bin/bash

# Exit on error
set -e

# Build the Docker image
echo "Building Docker image..."
docker build -t pytorch-gh200 .

# Get the command to run, or default to bash
CMD=${@:-/bin/bash}

# Run the Docker container with GPU support
echo "Running Docker container with GPU support..."
docker run \
  --gpus all \
  -it \
  --rm \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e __NV_PRIME_RENDER_OFFLOAD=1 \
  -e __GLX_VENDOR_LIBRARY_NAME=nvidia \
  -v "$(pwd)":/workspace \
  pytorch-gh200 \
  $CMD

echo "Docker container exited." 