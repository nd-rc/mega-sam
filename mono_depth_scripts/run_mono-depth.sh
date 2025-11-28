#!/bin/bash

DATA_PATH="/mnt/d/work/input/my_apartment"
IMAGES_PATH="$DATA_PATH/images"
SEQ_PATH="$DATA_PATH/depth"

echo "=== Запуск на RTX 5070 ==="
echo "Папка: $DATA_PATH"

# Включаем GPU
export CUDA_VISIBLE_DEVICES=0

# Запускаем DepthAnything
python Depth-Anything/run_videos.py --encoder vitl \
  --load-from Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  --img-path "$IMAGES_PATH" \
  --outdir "$SEQ_PATH" 

# UniDepth временно отключен из-за несовместимости xFormers с CUDA 12.8

echo "=== Готово! ==="
echo "Результат: "$SEQ_PATH" 