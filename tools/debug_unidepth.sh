#!/bin/bash

DATA_PATH="/mnt/d/work/images/my_apartment"
SEQ_NAME="my_apartment"

# Добавляем UniDepth в путь Python
export PYTHONPATH="${PYTHONPATH}:$(pwd)/UniDepth"

echo "=== Тест запуска UniDepth на RTX 5070 ==="

# Принудительно отключаем xformers через переменную окружения (иногда это помогает в новых либах)
export XFORMERS_DISABLED=1 

# Запуск
python UniDepth/scripts/demo_mega-sam.py \
  --scene-name "$SEQ_NAME" \
  --img-path "$DATA_PATH/images" \
  --outdir "UniDepth/outputs"