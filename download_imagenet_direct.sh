#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p ./data

# Download the dataset with progress indicator
echo "Downloading downsampled ImageNet 64x64 dataset..."
curl -L -o ./data/downsampled-imagenet-64x64.zip \
  https://www.kaggle.com/api/v1/datasets/download/ayaroshevskiy/downsampled-imagenet-64x64 \
  --progress-bar

# Check if download was successful
if [ $? -eq 0 ]; then
  echo "Download completed successfully!"
  echo "Dataset saved to: ./data/downsampled-imagenet-64x64.zip"
  
  # Extract the files with progress indicator
  echo "Extracting files..."
  unzip -q ./data/downsampled-imagenet-64x64.zip -d ./data/imagenet-64x64
  
  echo "Extraction completed!"
  echo "Dataset extracted to: ./data/imagenet-64x64"
else
  echo "Download failed!"
fi 