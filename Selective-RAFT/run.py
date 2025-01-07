#!/bin/bash

#To evaluate SceneFlow, run
# python evaluate_stereo.py \
#     --restore_ckpt ./weighs/sceneflow/sceneflow.pth 


model_name=eth3d
model_name=sceneflow  # 优
#model_name=middlebury # 优
model_name=kitti2015 # 优
python demo_imgs.py \
   --restore_ckpt ./weights/${model_name}/${model_name}_finetune.pth \
   --left_imgs    '/mnt/Data2/depth/depth20250102/left1/*.png' \
   --right_imgs   '/mnt/Data2/depth/depth20250102/right1/*.png' \
   --output_directory /mnt/Data2/depth/depth20250102/result/Selective-RAFT/${model_name}
