#!/bin/bash

SCENE_NAME="my_apartment"
DATA_PATH="/mnt/d/work/input/$SCENE_NAME"
IMAGES_PATH="$DATA_PATH/images"
SEQ_PATH="$DATA_PATH/depth"
#METRIC_DEPTH_PATH="$DATA_PATH/uni_depth"
METRIC_DEPTH_PATH="$DATA_PATH/depth"

CKPT_PATH=checkpoints/megasam_final.pth


   CUDA_VISIBLE_DEVICES=0 python camera_tracking_scripts/test_demo.py \
   --datapath=$IMAGES_PATH \
   --weights=$CKPT_PATH \
   --scene_name $SCENE_NAME \
   --mono_depth_path $SEQ_PATH \
   --metric_depth_path $METRIC_DEPTH_PATH \
   --disable_vis $@

